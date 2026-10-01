"""STEP 5.12N2 — Training-only mean-imputed five-method PU screen.

Run after Steps 5.12N0 and 5.12N1 in the redevelopment notebook.  This step
uses only the frozen Training cohort and its five locked folds.  Every method
receives the same fold-fitted mean-imputed feature matrix, without missingness
indicators and without native-NaN routing.  Validation and Test are not read.

Compared methods
----------------
1. Naive documented-positive versus unlabeled (reference)
2. Standard bagging PU
3. Repeated-spy reliable-negative PU
4. Official PULSCAR discovery followed by XGBoost scoring
5. Official PULSNAR discovery followed by XGBoost scoring

KI-CP-PU is not repeated because its renal-evidence promotion is circular for
the intended clinical utility audit.  AdaSampling is not repeated because the
faithful implementation failed its convergence criterion in all prior folds.
"""

from google.colab import drive
drive.mount("/content/drive")

from datetime import datetime, timezone
from pathlib import Path
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

from packaging.version import Version


# Keep the environment compatible with the already-audited official package.
try:
    sklearn_version_before = importlib.metadata.version("scikit-learn")
except importlib.metadata.PackageNotFoundError:
    sklearn_version_before = "0"
if Version(sklearn_version_before) < Version("1.9.0"):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "scikit-learn>=1.9,<1.10"]
    )
for package, requirement in {
    "xgboost": "xgboost",
    "catboost": "catboost==1.2.10",
}.items():
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", requirement])

import numpy as np
import pandas as pd
import sklearn
import xgboost
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


VERSIONS = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
}
if Version(VERSIONS["scikit_learn"]) < Version("1.9.0"):
    raise RuntimeError(
        "scikit-learn was upgraded on disk but an older version remains loaded. "
        "Restart the Colab runtime once and rerun this complete cell."
    )


EXPECTED_PATIENTS = 69_701
EXPECTED_POSITIVES = 5_042
EXPECTED_UNLABELED = 64_659
EXPECTED_FEATURES = 20
EXPECTED_FOLDS = 5
EXPECTED_TRANSFORMED_FEATURES = 21
WEIGHT_COL = "TEMPORAL_BALANCING_WEIGHT"

SEED = 71_000_042
PUL_SEED = 123
BOOTSTRAP_REPLICATES = 500
BAGS = 20
UNLABELED_TO_POSITIVE_RATIO = 1.0
SPY_REPETITIONS = 10
SPY_FRACTION = 0.15
SPY_LOWER_QUANTILE = 0.05
RN_VOTES_REQUIRED = 8

METHODS = [
    "naive_pn",
    "standard_bagging_pu",
    "spy_consensus_rn",
    "official_pulscar",
    "official_pulsnar",
]

LOCKED_XGB_PARAMETERS = {
    "n_estimators": 500,
    "learning_rate": 0.04570147939250174,
    "reg_lambda": 10.204527160227807,
    "colsample_bytree": 0.7400473792766912,
    "max_depth": 5,
    "min_child_weight": 7.323551441473847,
    "subsample": 0.8377920943702721,
}

ROOT = Path(
    "/content/drive/MyDrive/CKD_Project/outputs/"
    "section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
N1_ROOT = ROOT / "05_12N1_training_only_mean_vs_multiple_mice_screen"
OUTPUT_ROOT = ROOT / "05_12N2_training_only_mean_imputed_five_method_pu_screen"
SOFTWARE_LOCK = ROOT / (
    "05_6D1_pulsnar_software_and_parameter_lock/"
    "a7bf680d472f693279074683/pulsnar_software_and_parameter_lock.json"
)


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str
    ).encode("utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                json_safe(payload), stream, indent=2, ensure_ascii=False,
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_oof(path, ids, probability):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probability = np.asarray(probability, dtype=float)
    if len(ids) != len(probability) or not np.isfinite(probability).all():
        raise ValueError(f"Invalid probability vector: {path}")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"Probability outside [0,1]: {path}")
    temporary = path.with_suffix(".tmp.parquet")
    pd.DataFrame({ID_COL: np.asarray(ids), "probability": probability}).to_parquet(
        temporary, index=False
    )
    os.replace(temporary, path)


def read_oof(path, expected_ids):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        table = pd.read_parquet(path)
    except Exception:
        return None
    if list(table.columns) != [ID_COL, "probability"]:
        return None
    if not np.array_equal(table[ID_COL].to_numpy(), np.asarray(expected_ids)):
        return None
    probability = pd.to_numeric(table["probability"], errors="coerce").to_numpy(float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        return None
    return probability


def make_model(seed):
    return XGBClassifier(
        **LOCKED_XGB_PARAMETERS,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=int(seed),
        missing=np.nan,
        verbosity=0,
    )


def metric_row(method, target, probability):
    probability = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    return {
        "Method": method,
        "Average_precision": float(average_precision_score(target, probability)),
        "ROC_AUC": float(roc_auc_score(target, probability)),
        "Brier_score": float(brier_score_loss(target, probability)),
        "Log_loss": float(log_loss(target, probability, labels=[0, 1])),
    }


# -----------------------------------------------------------------------------
# Restore and verify the completed N1 decision basis.
# -----------------------------------------------------------------------------
n1_candidates = []
for completion_path in N1_ROOT.rglob("completion.json"):
    payload = load_json(completion_path)
    if payload and payload.get("status") == "TRAINING_ONLY_MEAN_VS_MULTIPLE_MICE_SCREEN_COMPLETE":
        summary_path = completion_path.parent / "training_oof_imputation_summary.csv"
        bootstrap_path = completion_path.parent / "paired_bootstrap_ap_difference.csv"
        if summary_path.is_file() and bootstrap_path.is_file():
            n1_candidates.append((completion_path, payload, summary_path, bootstrap_path))
if not n1_candidates:
    raise FileNotFoundError("Run Step 5.12N1 before Step 5.12N2.")

N1_FILE, n1, N1_SUMMARY_FILE, N1_BOOTSTRAP_FILE = max(
    n1_candidates, key=lambda item: item[0].stat().st_mtime
)
if "temporal balancing weights" not in str(n1.get("training_weight_policy", "")).lower():
    raise ValueError(
        "The latest N1 run predates the temporal-weight correction. Rerun the "
        "updated Step 5.12N1 before running N2."
    )
n1_summary = pd.read_csv(N1_SUMMARY_FILE)
n1_bootstrap = pd.read_csv(N1_BOOTSTRAP_FILE)
required_n1_candidates = {"mean_imputation", "multiple_mice_5"}
if set(n1_summary["Candidate"]) != required_n1_candidates:
    raise ValueError("Unexpected N1 candidate set.")
mean_ap = float(
    n1_summary.loc[n1_summary["Candidate"].eq("mean_imputation"), "Average_precision"].iloc[0]
)
mice_ap = float(
    n1_summary.loc[n1_summary["Candidate"].eq("multiple_mice_5"), "Average_precision"].iloc[0]
)
mice_ci_upper = float(n1_bootstrap["Bootstrap_95CI_upper"].iloc[0])
if not (mean_ap > mice_ap and mice_ci_upper < 0):
    raise ValueError(
        "N1 does not support the prespecified Mean lock: Mean must have higher AP "
        "and the paired MICE-minus-Mean AP interval must be below zero."
    )

N0_FILE = Path(n1["source_policy_lock"])
n0 = load_json(N0_FILE)
if n0 is None or n0.get("status") != "TRAINING_ONLY_HIGH_MISSINGNESS_POLICY_LOCKED":
    raise ValueError("The N0 source policy lock recorded by N1 is invalid.")
SNAPSHOT_FILE = Path(n0["frozen_training_snapshot"])
EXPECTED_SNAPSHOT_HASH = n0["frozen_training_snapshot_sha256"]
FEATURES = list(n0["retained_features"])
EXCLUDED_FEATURES = list(n0["excluded_features"])
if len(FEATURES) != EXPECTED_FEATURES or "GENDER" not in FEATURES:
    raise ValueError("The locked N0 retained feature panel is incompatible.")
if not SNAPSHOT_FILE.is_file() or sha256_file(SNAPSHOT_FILE) != EXPECTED_SNAPSHOT_HASH:
    raise ValueError("The frozen Training snapshot is missing or changed.")

d0 = load_json(n0["source_protocol"])
if d0 is None:
    raise ValueError("The source protocol recorded by N0 is unreadable.")
ID_COL = d0["cohort"]["patient_id_column"]
LABEL_COL = d0["cohort"]["observed_label_column"]
FOLD_COL = d0["cohort"]["locked_fold_column"]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature != "GENDER"]

training = pd.read_parquet(
    SNAPSHOT_FILE, columns=[ID_COL, LABEL_COL, FOLD_COL, WEIGHT_COL, *FEATURES]
)
if len(training) != EXPECTED_PATIENTS:
    raise ValueError(f"Training row count changed: {len(training):,}")
if training[ID_COL].isna().any() or training[ID_COL].duplicated().any():
    raise ValueError("Training identifiers are missing or duplicated.")

labels = pd.to_numeric(training[LABEL_COL], errors="raise").astype(int).to_numpy()
if int(labels.sum()) != EXPECTED_POSITIVES or int((labels == 0).sum()) != EXPECTED_UNLABELED:
    raise ValueError("Training class counts changed.")
weights = pd.to_numeric(training[WEIGHT_COL], errors="raise").to_numpy(float)
if not np.isfinite(weights).all() or np.any(weights <= 0):
    raise ValueError("Temporal balancing weights are invalid.")
weights = weights / weights.mean()
folds = training[FOLD_COL].to_numpy()
fold_values = sorted(pd.Series(folds).dropna().unique().tolist())
if len(fold_values) != EXPECTED_FOLDS:
    raise ValueError(f"Expected five locked folds, found {fold_values}.")
if training["GENDER"].isna().any():
    raise ValueError("GENDER unexpectedly contains missing values.")

numeric_frame = training[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan)
gender_frame = training[["GENDER"]].astype("string")


# -----------------------------------------------------------------------------
# Verify and load the archived official PULSCAR/PULSNAR implementation.
# -----------------------------------------------------------------------------
software_lock = load_json(SOFTWARE_LOCK)
if software_lock is None or software_lock.get("status") != "PULSNAR_SOFTWARE_AND_PARAMETERS_LOCKED":
    raise ValueError("The audited official PULSNAR software lock is unavailable.")
shared_parameters = dict(software_lock["shared_PULSNAR_parameters"])
expected_shared = {
    "scar": False,
    "csrdata": False,
    "n_clusters": 0,
    "max_clusters": 25,
    "covar_type": "full",
    "top50p_covars": True,
    "bin_method": "scott",
    "bw_method": "hist",
    "lowerbw": 0.01,
    "upperbw": 0.5,
    "optim": "local",
    "calibration": True,
    "calibration_data": "PU",
    "calibration_method": "isotonic",
    "calibration_n_bins": 100,
    "smooth_isotonic": False,
    "classification_metrics": False,
    "n_iterations": 1,
    "kfold": 5,
    "kflips": 1,
}
if shared_parameters != expected_shared:
    raise ValueError("Official PULSNAR parameters differ from the audited protocol.")
official = software_lock["official_implementation"]
WHEEL = Path(official["archived_wheel"])
if not WHEEL.is_file() or sha256_file(WHEEL) != official["archived_wheel_sha256"]:
    raise ValueError("The archived official PULSNAR wheel is missing or changed.")


context = {
    "step": "5.12N2",
    "status": "TRAINING_ONLY_MEAN_IMPUTED_FIVE_METHOD_PU_SCREEN",
    "n0_policy_lock": str(N0_FILE),
    "n0_policy_lock_sha256": sha256_file(N0_FILE),
    "n1_completion": str(N1_FILE),
    "n1_completion_sha256": sha256_file(N1_FILE),
    "training_snapshot": str(SNAPSHOT_FILE),
    "training_snapshot_sha256": EXPECTED_SNAPSHOT_HASH,
    "software_lock": str(SOFTWARE_LOCK),
    "software_lock_sha256": sha256_file(SOFTWARE_LOCK),
    "features": FEATURES,
    "excluded_features": EXCLUDED_FEATURES,
    "imputation_policy": (
        "fold-fitted arithmetic mean for every retained numeric feature; "
        "no missingness indicators; no NaN reaches discovery or scoring"
    ),
    "categorical_policy": "fold-fitted one-hot encoding; unknown ignored",
    "engine": "XGBoost",
    "engine_parameters": LOCKED_XGB_PARAMETERS,
    "training_weight_policy": (
        "previously locked temporal balancing weights normalized to mean one; "
        "applied to every downstream XGBoost fit"
    ),
    "methods": METHODS,
    "methodological_exclusions": {
        "ki_cp_pu": "not repeated: renal-evidence promotion is circular for renal-evidence utility",
        "faithful_adasampling_xgb": "not repeated: failed epsilon convergence in all prior folds",
    },
    "official_pu_parameters": expected_shared,
    "folds": EXPECTED_FOLDS,
    "evaluation_label": "documented CKD versus unlabeled; not verified latent CKD",
    "versions": VERSIONS,
    "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
CACHE_DIR = PRIVATE_DIR / "fold_checkpoints"
DISCOVERY_DIR = PRIVATE_DIR / "official_pu_discovery"
for directory in [OUTPUT_DIR, PRIVATE_DIR, CACHE_DIR, DISCOVERY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONTEXT_FILE = OUTPUT_DIR / "context.json"
if CONTEXT_FILE.is_file():
    existing_context = load_json(CONTEXT_FILE)
    if existing_context != json_safe(context):
        raise ValueError("Existing N2 context is incompatible.")
else:
    atomic_json(CONTEXT_FILE, context)

subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
        "--force-reinstall", str(WHEEL),
    ]
)
from PULSNAR import PULSNAR  # noqa: E402


def transform_fold(fit_positions, eval_positions):
    """Fit Mean and gender transforms on the outer fitting fold only."""
    imputer = SimpleImputer(
        strategy="mean", add_indicator=False, keep_empty_features=True
    )
    fit_numeric = imputer.fit_transform(
        numeric_frame.iloc[fit_positions]
    ).astype(np.float32)
    eval_numeric = imputer.transform(
        numeric_frame.iloc[eval_positions]
    ).astype(np.float32)

    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=np.float32
    )
    fit_gender = encoder.fit_transform(gender_frame.iloc[fit_positions])
    eval_gender = encoder.transform(gender_frame.iloc[eval_positions])
    gender_names = encoder.get_feature_names_out(["GENDER"]).tolist()

    feature_names = [*NUMERIC_FEATURES, *gender_names]
    fit_x = pd.DataFrame(
        np.column_stack([fit_numeric, fit_gender]).astype(np.float32),
        columns=feature_names,
    )
    eval_x = pd.DataFrame(
        np.column_stack([eval_numeric, eval_gender]).astype(np.float32),
        columns=feature_names,
    )
    if fit_x.shape[1] != EXPECTED_TRANSFORMED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_TRANSFORMED_FEATURES} transformed features, "
            f"found {fit_x.shape[1]}."
        )
    if not np.isfinite(fit_x.to_numpy()).all() or not np.isfinite(eval_x.to_numpy()).all():
        raise ValueError("Mean-imputed scorer matrix contains a non-finite value.")

    # Official discovery gets the same completed values, standardized within
    # the fitting fold.  Standardization is required only by discovery.
    scaler = StandardScaler()
    discovery_x = scaler.fit_transform(fit_x).astype(np.float64)
    if not np.isfinite(discovery_x).all():
        raise ValueError("Official PU discovery matrix contains a non-finite value.")
    return fit_x, eval_x, discovery_x, feature_names


def run_official_discovery(method, fold_number, discovery_x, y_fit, fit_ids):
    if method not in {"official_pulscar", "official_pulsnar"}:
        raise ValueError(method)
    fold_dir = DISCOVERY_DIR / method / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    audit_path = fold_dir / "discovery_audit.json"
    selected_path = fold_dir / "selected_probable_positive.parquet"

    if audit_path.is_file() and selected_path.is_file():
        audit = load_json(audit_path)
        selected_table = pd.read_parquet(selected_path)
        if list(selected_table.columns) != [ID_COL, "selected_probable_positive"]:
            raise ValueError(f"Invalid saved discovery selection: {selected_path}")
        if not np.array_equal(selected_table[ID_COL].to_numpy(), np.asarray(fit_ids)):
            raise ValueError(f"Saved discovery IDs changed: {selected_path}")
        return selected_table["selected_probable_positive"].astype(bool).to_numpy(), audit

    parameters = dict(shared_parameters)
    parameters["scar"] = method == "official_pulscar"
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix=f"{method}_fold{fold_number}_", dir=str(fold_dir)
    ) as temporary_text:
        temporary = Path(temporary_text)
        prior_cwd = Path.cwd()
        try:
            os.chdir(temporary)
            estimator = PULSNAR.PULSNARClassifier(
                classifier="xgboost",
                rseed=PUL_SEED + int(fold_number),
                **parameters,
            )
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                result = estimator.pulsnar(
                    discovery_x,
                    y_fit,
                    tru_label=None,
                    rec_list=np.arange(len(y_fit), dtype=np.int64),
                )
        finally:
            os.chdir(prior_cwd)

        prediction_source = temporary / Path(result["prediction_file"]).name
        alpha_source = temporary / Path(result["alpha_file"]).name
        if not prediction_source.is_file() or not alpha_source.is_file():
            raise FileNotFoundError(f"Official {method} did not produce its outputs.")
        shutil.copy2(prediction_source, fold_dir / "official_predictions.tsv")
        shutil.copy2(alpha_source, fold_dir / "official_alpha_estimates.tsv")
        for optional_key, destination_name in [
            ("important_features_file", "official_model_imp_features.pkl"),
            ("bic_vs_cluster_count_plot", "official_bic_vs_cluster_count.png"),
        ]:
            source_value = result.get(optional_key)
            if source_value:
                source = temporary / Path(source_value).name
                if source.is_file():
                    shutil.copy2(source, fold_dir / destination_name)

    predictions = pd.read_csv(fold_dir / "official_predictions.tsv", sep="\t")
    required_columns = {"rec_id", "ml_label", "calibrated_prob"}
    if not required_columns.issubset(predictions.columns) or len(predictions) != len(y_fit):
        raise ValueError(f"Official {method} prediction schema changed.")
    predictions["rec_id"] = pd.to_numeric(predictions["rec_id"], errors="raise").astype(int)
    predictions["ml_label"] = pd.to_numeric(predictions["ml_label"], errors="raise").astype(int)
    predictions["calibrated_prob"] = pd.to_numeric(
        predictions["calibrated_prob"], errors="raise"
    ).astype(float)
    predictions = predictions.sort_values("rec_id").reset_index(drop=True)
    if not np.array_equal(predictions["rec_id"].to_numpy(), np.arange(len(y_fit))):
        raise ValueError(f"Official {method} record positions changed.")
    if not np.array_equal(predictions["ml_label"].to_numpy(), y_fit):
        raise ValueError(f"Official {method} labels changed.")
    posterior = predictions["calibrated_prob"].to_numpy(float)
    if not np.isfinite(posterior).all() or np.any((posterior < 0) | (posterior > 1)):
        raise ValueError(f"Official {method} returned invalid posterior values.")

    estimated_alpha = float(result["estimated_alpha"])
    if not 0 <= estimated_alpha <= 1:
        raise ValueError(f"Official {method} alpha is outside [0,1].")
    unlabeled_local = np.flatnonzero(y_fit == 0)
    selection_count = int(math.floor(estimated_alpha * len(unlabeled_local) + 0.5))
    if not 0 <= selection_count <= len(unlabeled_local):
        raise ValueError(f"Invalid alpha-matched selection count for {method}.")

    ranking = pd.DataFrame(
        {"position": unlabeled_local, "posterior": posterior[unlabeled_local]}
    )
    ranking["tie_hash"] = ranking["position"].map(
        lambda value: hashlib.sha256(
            f"{method}|{PUL_SEED + int(fold_number)}|{int(value)}".encode("utf-8")
        ).hexdigest()
    )
    ranking = ranking.sort_values(
        ["posterior", "tie_hash"], ascending=[False, True], kind="stable"
    )
    selected_positions = ranking.iloc[:selection_count]["position"].to_numpy(int)
    selected = np.zeros(len(y_fit), dtype=bool)
    selected[selected_positions] = True

    pd.DataFrame(
        {ID_COL: np.asarray(fit_ids), "selected_probable_positive": selected}
    ).to_parquet(selected_path, index=False)
    audit = {
        "Method": method,
        "Fold": int(fold_number),
        "Training_rows": int(len(y_fit)),
        "Documented_positive": int(y_fit.sum()),
        "Unlabeled": int((y_fit == 0).sum()),
        "Estimated_alpha": estimated_alpha,
        "Selected_probable_positive": selection_count,
        "Selected_percent_of_unlabeled": 100.0 * selection_count / len(unlabeled_local),
        "Runtime_minutes": (time.perf_counter() - started) / 60.0,
        "Official_stdout": stdout_buffer.getvalue(),
        "Official_stderr": stderr_buffer.getvalue(),
    }
    atomic_json(audit_path, audit)
    return selected, audit


oof = {method: np.full(EXPECTED_PATIENTS, np.nan, dtype=float) for method in METHODS}
fit_rows = []
bag_rows = []
spy_rows = []
discovery_rows = []
preprocessing_rows = []

for fold_number, fold_value in enumerate(fold_values, start=1):
    fit_positions = np.flatnonzero(folds != fold_value)
    eval_positions = np.flatnonzero(folds == fold_value)
    fit_ids = training.iloc[fit_positions][ID_COL].to_numpy()
    eval_ids = training.iloc[eval_positions][ID_COL].to_numpy()
    y_fit = labels[fit_positions]
    w_fit = weights[fit_positions]
    positive_local = np.flatnonzero(y_fit == 1)
    unlabeled_local = np.flatnonzero(y_fit == 0)
    if not len(positive_local) or not len(unlabeled_local):
        raise ValueError(f"Fold {fold_number} lacks one PU class.")

    existing = {
        method: read_oof(CACHE_DIR / f"{method}__fold_{fold_number}.parquet", eval_ids)
        for method in METHODS
    }
    if all(value is not None for value in existing.values()):
        for method in METHODS:
            oof[method][eval_positions] = existing[method]
            fit_rows.append(
                {
                    "Method": method, "Fold": fold_number, "Status": "reused",
                    "Models_fitted": 0, "Runtime_minutes": 0.0,
                }
            )
        print(f"Reused all methods, fold {fold_number}.")
        continue

    fit_x, eval_x, discovery_x, transformed_names = transform_fold(
        fit_positions, eval_positions
    )
    preprocessing_rows.append(
        {
            "Fold": fold_number,
            "Fitting_patients": len(fit_positions),
            "Evaluation_patients": len(eval_positions),
            "Original_features": len(FEATURES),
            "Transformed_features": len(transformed_names),
            "NaN_in_fitting_matrix": int(np.isnan(fit_x.to_numpy()).sum()),
            "NaN_in_evaluation_matrix": int(np.isnan(eval_x.to_numpy()).sum()),
            "Missingness_indicators": 0,
        }
    )

    # 1) Naive documented-positive versus unlabeled reference.
    method = "naive_pn"
    if existing[method] is not None:
        probability = existing[method]
        status, models_fitted, runtime = "reused", 0, 0.0
    else:
        started = time.perf_counter()
        model = make_model(SEED + 10_000 + fold_number)
        model.fit(fit_x, y_fit, sample_weight=w_fit)
        probability = np.asarray(model.predict_proba(eval_x)[:, 1], float)
        save_oof(CACHE_DIR / f"{method}__fold_{fold_number}.parquet", eval_ids, probability)
        status, models_fitted = "fitted", 1
        runtime = (time.perf_counter() - started) / 60.0
        print(f"Completed {method}, fold {fold_number}, {runtime:.2f} minutes.")
        del model
    oof[method][eval_positions] = probability
    fit_rows.append(
        {"Method": method, "Fold": fold_number, "Status": status,
         "Models_fitted": models_fitted, "Runtime_minutes": runtime}
    )
    gc.collect()

    # 2) Standard bagging PU: every positive plus an equal random U sample.
    method = "standard_bagging_pu"
    if existing[method] is not None:
        probability = existing[method]
        status, models_fitted, runtime = "reused", 0, 0.0
    else:
        started = time.perf_counter()
        rng = np.random.default_rng(SEED + 20_000 + fold_number)
        sample_size = int(round(UNLABELED_TO_POSITIVE_RATIO * len(positive_local)))
        if sample_size > len(unlabeled_local):
            raise ValueError("Bagging requested more unlabeled records than available.")
        bag_predictions = np.empty((BAGS, len(eval_positions)), dtype=float)
        for bag in range(1, BAGS + 1):
            sampled_unlabeled = rng.choice(unlabeled_local, sample_size, replace=False)
            selected = np.concatenate([positive_local, sampled_unlabeled])
            target = np.concatenate(
                [np.ones(len(positive_local), int), np.zeros(sample_size, int)]
            )
            selected_weight = w_fit[selected]
            model = make_model(SEED + 20_000 + fold_number * 100 + bag)
            model.fit(
                fit_x.iloc[selected], target, sample_weight=selected_weight
            )
            bag_predictions[bag - 1] = model.predict_proba(eval_x)[:, 1]
            del model
        probability = bag_predictions.mean(axis=0)
        save_oof(CACHE_DIR / f"{method}__fold_{fold_number}.parquet", eval_ids, probability)
        runtime = (time.perf_counter() - started) / 60.0
        status, models_fitted = "fitted", BAGS
        bag_rows.append(
            {
                "Fold": fold_number,
                "Bags": BAGS,
                "Training_positives_per_bag": len(positive_local),
                "Sampled_unlabeled_per_bag": sample_size,
                "Mean_eval_probability_SD_across_bags": float(
                    bag_predictions.std(axis=0, ddof=1).mean()
                ),
            }
        )
        print(f"Completed {method}, fold {fold_number}, {runtime:.2f} minutes.")
        del bag_predictions
    oof[method][eval_positions] = probability
    fit_rows.append(
        {"Method": method, "Fold": fold_number, "Status": status,
         "Models_fitted": models_fitted, "Runtime_minutes": runtime}
    )
    gc.collect()

    # 3) Repeated-spy consensus reliable-negative method.
    method = "spy_consensus_rn"
    spy_audit_path = CACHE_DIR / f"{method}__fold_{fold_number}__audit.json"
    if existing[method] is not None and spy_audit_path.is_file():
        probability = existing[method]
        spy_rows.append(load_json(spy_audit_path))
        status, models_fitted, runtime = "reused", 0, 0.0
    else:
        started = time.perf_counter()
        rn_votes = np.zeros(len(unlabeled_local), dtype=np.int16)
        thresholds = []
        spy_count = max(1, int(round(SPY_FRACTION * len(positive_local))))
        for repetition in range(1, SPY_REPETITIONS + 1):
            rng = np.random.default_rng(
                SEED + 30_000 + fold_number * 100 + repetition
            )
            spy_local = rng.choice(positive_local, spy_count, replace=False)
            retained_positive = positive_local[~np.isin(positive_local, spy_local)]
            selected = np.concatenate([retained_positive, unlabeled_local, spy_local])
            target = np.concatenate(
                [
                    np.ones(len(retained_positive), int),
                    np.zeros(len(unlabeled_local) + len(spy_local), int),
                ]
            )
            detector = make_model(SEED + 30_000 + fold_number * 100 + repetition)
            detector.fit(
                fit_x.iloc[selected], target, sample_weight=w_fit[selected]
            )
            spy_scores = detector.predict_proba(fit_x.iloc[spy_local])[:, 1]
            unlabeled_scores = detector.predict_proba(fit_x.iloc[unlabeled_local])[:, 1]
            threshold = float(np.quantile(spy_scores, SPY_LOWER_QUANTILE))
            rn_votes += (unlabeled_scores <= threshold).astype(np.int16)
            thresholds.append(threshold)
            del detector
        reliable_negative_local = unlabeled_local[rn_votes >= RN_VOTES_REQUIRED]
        if len(reliable_negative_local) < len(positive_local):
            raise ValueError(
                f"Spy fold {fold_number} found only "
                f"{len(reliable_negative_local)} reliable negatives."
            )
        selected = np.concatenate([positive_local, reliable_negative_local])
        target = np.concatenate(
            [np.ones(len(positive_local), int), np.zeros(len(reliable_negative_local), int)]
        )
        model = make_model(SEED + 39_000 + fold_number)
        model.fit(
            fit_x.iloc[selected], target, sample_weight=w_fit[selected]
        )
        probability = np.asarray(model.predict_proba(eval_x)[:, 1], float)
        save_oof(CACHE_DIR / f"{method}__fold_{fold_number}.parquet", eval_ids, probability)
        runtime = (time.perf_counter() - started) / 60.0
        status, models_fitted = "fitted", SPY_REPETITIONS + 1
        spy_audit = {
            "Fold": fold_number,
            "Training_positives": len(positive_local),
            "Training_unlabeled": len(unlabeled_local),
            "Spies_per_repetition": spy_count,
            "Reliable_negatives": len(reliable_negative_local),
            "Reliable_negative_percent_of_unlabeled": (
                100.0 * len(reliable_negative_local) / len(unlabeled_local)
            ),
            "Median_threshold": float(np.median(thresholds)),
            "Minimum_threshold": float(np.min(thresholds)),
            "Maximum_threshold": float(np.max(thresholds)),
        }
        atomic_json(spy_audit_path, spy_audit)
        spy_rows.append(spy_audit)
        print(f"Completed {method}, fold {fold_number}, {runtime:.2f} minutes.")
        del model
    oof[method][eval_positions] = probability
    fit_rows.append(
        {"Method": method, "Fold": fold_number, "Status": status,
         "Models_fitted": models_fitted, "Runtime_minutes": runtime}
    )
    gc.collect()

    # 4–5) Official PU discovery and an otherwise identical Mean-XGBoost scorer.
    for method_index, method in enumerate(["official_pulscar", "official_pulsnar"]):
        if existing[method] is not None:
            probability = existing[method]
            status, models_fitted, runtime = "reused", 0, 0.0
        else:
            started = time.perf_counter()
            selected, audit = run_official_discovery(
                method, fold_number, discovery_x, y_fit, fit_ids
            )
            nested_target = y_fit.copy()
            nested_target[selected] = 1
            if int(nested_target.sum()) != int(y_fit.sum()) + int(selected.sum()):
                raise ValueError(f"Invalid nested target for {method}, fold {fold_number}.")
            model = make_model(
                SEED + 40_000 + method_index * 1_000 + fold_number
            )
            model.fit(fit_x, nested_target, sample_weight=w_fit)
            probability = np.asarray(model.predict_proba(eval_x)[:, 1], float)
            save_oof(
                CACHE_DIR / f"{method}__fold_{fold_number}.parquet",
                eval_ids,
                probability,
            )
            runtime = (time.perf_counter() - started) / 60.0
            status, models_fitted = "fitted", 1
            discovery_rows.append(audit)
            print(f"Completed {method}, fold {fold_number}, {runtime:.2f} minutes.")
            del model
        oof[method][eval_positions] = probability
        fit_rows.append(
            {"Method": method, "Fold": fold_number, "Status": status,
             "Models_fitted": models_fitted, "Runtime_minutes": runtime}
        )
        gc.collect()

    del fit_x, eval_x, discovery_x
    gc.collect()


for method in METHODS:
    if not np.isfinite(oof[method]).all():
        raise ValueError(f"Incomplete OOF predictions for {method}.")
    save_oof(
        PRIVATE_DIR / f"{method}__complete_oof.parquet",
        training[ID_COL].to_numpy(),
        oof[method],
    )

summary = pd.DataFrame(
    [metric_row(method, labels, oof[method]) for method in METHODS]
).sort_values(
    ["Average_precision", "Log_loss", "Brier_score"],
    ascending=[False, True, True],
    kind="stable",
).reset_index(drop=True)


# Paired patient-level, label-stratified bootstrap versus naive PN.
positive_positions = np.flatnonzero(labels == 1)
unlabeled_positions = np.flatnonzero(labels == 0)
rng = np.random.default_rng(SEED + 90_000)
bootstrap_values = {
    method: np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for method in METHODS if method != "naive_pn"
}
for replicate in range(BOOTSTRAP_REPLICATES):
    sampled = np.concatenate(
        [
            rng.choice(positive_positions, len(positive_positions), replace=True),
            rng.choice(unlabeled_positions, len(unlabeled_positions), replace=True),
        ]
    )
    rng.shuffle(sampled)
    reference_ap = average_precision_score(labels[sampled], oof["naive_pn"][sampled])
    for method in bootstrap_values:
        bootstrap_values[method][replicate] = (
            average_precision_score(labels[sampled], oof[method][sampled])
            - reference_ap
        )
    if (replicate + 1) % 100 == 0:
        print(f"Bootstrap {replicate + 1}/{BOOTSTRAP_REPLICATES}")

reference = metric_row("naive_pn", labels, oof["naive_pn"])
comparison_rows = []
for method, values in bootstrap_values.items():
    candidate = metric_row(method, labels, oof[method])
    comparison_rows.append(
        {
            "Method": method,
            "Candidate_AP": candidate["Average_precision"],
            "Naive_PN_AP": reference["Average_precision"],
            "Delta_AP_vs_naive_PN": (
                candidate["Average_precision"] - reference["Average_precision"]
            ),
            "Bootstrap_95CI_lower": float(np.quantile(values, 0.025)),
            "Bootstrap_95CI_upper": float(np.quantile(values, 0.975)),
            "Candidate_Brier": candidate["Brier_score"],
            "Naive_PN_Brier": reference["Brier_score"],
            "Candidate_Log_loss": candidate["Log_loss"],
            "Naive_PN_Log_loss": reference["Log_loss"],
            "Replicates": BOOTSTRAP_REPLICATES,
        }
    )
paired_comparison = pd.DataFrame(comparison_rows).sort_values(
    "Candidate_AP", ascending=False, kind="stable"
).reset_index(drop=True)

summary.to_csv(OUTPUT_DIR / "mean_imputed_pu_oof_summary.csv", index=False)
paired_comparison.to_csv(
    OUTPUT_DIR / "paired_pu_ap_difference_vs_naive_pn.csv", index=False
)
pd.DataFrame(fit_rows).to_csv(OUTPUT_DIR / "fit_manifest.csv", index=False)
pd.DataFrame(bag_rows).to_csv(OUTPUT_DIR / "bagging_fold_audit.csv", index=False)
pd.DataFrame(spy_rows).to_csv(OUTPUT_DIR / "spy_rn_fold_audit.csv", index=False)
pd.DataFrame(discovery_rows).to_csv(
    OUTPUT_DIR / "official_pu_discovery_fold_audit.csv", index=False
)
pd.DataFrame(preprocessing_rows).to_csv(
    OUTPUT_DIR / "fold_preprocessing_audit.csv", index=False
)

decision = {
    "step": "5.12N2",
    "status": "TRAINING_ONLY_MEAN_IMPUTED_FIVE_METHOD_PU_SCREEN_COMPLETE",
    "imputation_policy_locked": "mean",
    "retained_feature_count": len(FEATURES),
    "all_scorer_matrices_complete": True,
    "native_nan_routing_used": False,
    "missingness_indicators_used": False,
    "ranking": summary["Method"].tolist(),
    "no_method_promoted_yet": True,
    "selection_note": (
        "Observed-label AP is not sufficient alone to select a hidden-case "
        "method. Apply the prespecified utility/shortlist gates next."
    ),
    "next_step": (
        "Training-only hidden-case utility and shortlist gate for the eligible "
        "PU methods, then equal-budget tuning."
    ),
    "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
    "model_promoted_or_replaced": False,
}
atomic_json(OUTPUT_DIR / "screen_decision.json", decision)
atomic_json(
    OUTPUT_DIR / "completion.json",
    {
        **context,
        **decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
    },
)

print("=" * 112)
print("STEP 5.12N2 — TRAINING-ONLY MEAN-IMPUTED FIVE-METHOD PU SCREEN COMPLETE")
print("=" * 112)
print("\nPACKAGE VERSIONS")
print(json.dumps(VERSIONS, indent=2, ensure_ascii=False))
print("\nMEAN-IMPUTED PU OOF SUMMARY")
display(summary)
print("\nPAIRED AP COMPARISON VERSUS NAIVE PN")
display(paired_comparison)
print("\nFOLD PREPROCESSING AUDIT")
display(pd.DataFrame(preprocessing_rows))
print("\nOFFICIAL PU DISCOVERY AUDIT")
display(pd.DataFrame(discovery_rows))
print("\nSCREEN STATUS")
print(json.dumps(decision, indent=2, ensure_ascii=False))
print("\nSaved run:", OUTPUT_DIR)
print("Validation accessed in this step: No")
print("Test accessed in this step: No")
print("Native-NaN routing used: No")
print("Missingness indicators used: No")
print("Model promoted or replaced: No")
