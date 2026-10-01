"""STEP 5.12N4 — Equal-budget nested Training-only PULSCAR tuning.

Pipelines: official PULSCAR and naive PN reference. Five locked outer folds
produce nested OOF predictions. Within each outer fitting partition, its four
remaining locked folds are used for Optuna tuning. Mean imputation and one-hot
encoding are fitted inside every split. Official PULSCAR discovery is rerun
inside every inner fitting partition and the outer fitting partition.

No missingness indicators, native-NaN routing, Validation, or Test are used.
This step tunes but does not select or promote a final pipeline.
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
    "optuna": "optuna>=4,<5",
}.items():
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", requirement])

import numpy as np
import optuna
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
    "optuna": optuna.__version__,
}
if Version(VERSIONS["scikit_learn"]) < Version("1.9.0"):
    raise RuntimeError(
        "scikit-learn was upgraded on disk but an older version remains loaded. "
        "Restart the Colab runtime once, then rerun this complete cell."
    )


SEED = 71_000_044
PUL_SEED = 123
EXPECTED_PATIENTS = 69_701
EXPECTED_POSITIVES = 5_042
EXPECTED_UNLABELED = 64_659
EXPECTED_FOLDS = 5
EXPECTED_FEATURES = 20
EXPECTED_TRANSFORMED_FEATURES = 21
TRIALS_PER_OUTER_PIPELINE = 20
WEIGHT_COL = "TEMPORAL_BALANCING_WEIGHT"

METHODS = ["naive_pn", "official_pulscar"]
SEARCH_SPACE = {
    "n_estimators": [200, 700, 100],
    "learning_rate": [0.01, 0.15, "log"],
    "reg_lambda": [1e-4, 30.0, "log"],
    "colsample_bytree": [0.5, 1.0],
    "max_depth": [2, 6],
    "min_child_weight": [1.0, 40.0],
    "subsample": [0.6, 1.0],
}

PROJECT_ROOT = Path("/content/drive/MyDrive/CKD_Project")
ROOT = PROJECT_ROOT / "outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling"
N3_ROOT = ROOT / "05_12N3_training_only_hidden_case_utility_and_shortlist"
OUTPUT_ROOT = ROOT / "05_12N4_equal_budget_nested_training_only_pulscar_tuning"
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


def save_oof(path, ids, probability, id_col):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probability = np.asarray(probability, float)
    if len(ids) != len(probability) or not np.isfinite(probability).all():
        raise ValueError(f"Invalid probability vector: {path}")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"Probability outside [0,1]: {path}")
    temporary = path.with_suffix(".tmp.parquet")
    pd.DataFrame({id_col: np.asarray(ids), "probability": probability}).to_parquet(
        temporary, index=False
    )
    os.replace(temporary, path)


def read_oof(path, expected_ids, id_col):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        table = pd.read_parquet(path)
    except Exception:
        return None
    if list(table.columns) != [id_col, "probability"]:
        return None
    if not np.array_equal(table[id_col].to_numpy(), np.asarray(expected_ids)):
        return None
    probability = pd.to_numeric(table["probability"], errors="coerce").to_numpy(float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        return None
    return probability


def metrics(target, probability):
    clipped = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    return {
        "Average_precision": float(average_precision_score(target, clipped)),
        "ROC_AUC": float(roc_auc_score(target, clipped)),
        "Brier_score": float(brier_score_loss(target, clipped)),
        "Log_loss": float(log_loss(target, clipped, labels=[0, 1])),
    }


def method_seed(method):
    return {"naive_pn": 10_000, "official_pulscar": 20_000}[method]


# ---------------------------------------------------------------------------
# Restore the exact N3 shortlist and its locked upstream sources.
# ---------------------------------------------------------------------------
n3_candidates = []
if N3_ROOT.exists():
    for completion_path in N3_ROOT.rglob("completion.json"):
        payload = load_json(completion_path)
        if payload and payload.get("status") == "TRAINING_ONLY_HIDDEN_CASE_UTILITY_AND_SHORTLIST_COMPLETE":
            n3_candidates.append((completion_path, payload))
if not n3_candidates:
    raise FileNotFoundError("Run Step 5.12N3 before Step 5.12N4.")

N3_FILE, n3 = max(n3_candidates, key=lambda item: item[0].stat().st_mtime)
if n3.get("shortlisted_for_equal_budget_nested_training_only_tuning") != ["official_pulscar"]:
    raise ValueError("N3 shortlist is not exactly official_pulscar.")
if n3.get("validation_accessed_in_this_step") or n3.get("test_accessed_in_this_step"):
    raise ValueError("N3 reports held-out access.")
if not n3.get("no_pipeline_selected_in_this_step", False):
    raise ValueError("N3 unexpectedly selected a final pipeline.")

N2_FILE = Path(n3["n2_completion"])
n2 = load_json(N2_FILE)
if n2 is None or n2.get("status") != "TRAINING_ONLY_MEAN_IMPUTED_FIVE_METHOD_PU_SCREEN_COMPLETE":
    raise ValueError("N3 points to an invalid N2 completion.")
if sha256_file(N2_FILE) != n3["n2_completion_sha256"]:
    raise ValueError("N2 completion changed after N3.")

N0_FILE = Path(n2["n0_policy_lock"])
n0 = load_json(N0_FILE)
if n0 is None or n0.get("status") != "TRAINING_ONLY_HIGH_MISSINGNESS_POLICY_LOCKED":
    raise ValueError("N2 points to an invalid N0 policy lock.")
if sha256_file(N0_FILE) != n2["n0_policy_lock_sha256"]:
    raise ValueError("N0 policy lock changed after N2.")

SNAPSHOT_FILE = Path(n2["training_snapshot"])
if not SNAPSHOT_FILE.is_file() or sha256_file(SNAPSHOT_FILE) != n2["training_snapshot_sha256"]:
    raise ValueError("The frozen Training snapshot is missing or changed.")

d0 = load_json(n0["source_protocol"])
if d0 is None:
    raise ValueError("The source protocol recorded by N0 is unreadable.")
ID_COL = d0["cohort"]["patient_id_column"]
LABEL_COL = d0["cohort"]["observed_label_column"]
FOLD_COL = d0["cohort"]["locked_fold_column"]
FEATURES = list(n0["retained_features"])
EXCLUDED_FEATURES = list(n0["excluded_features"])
if len(FEATURES) != EXPECTED_FEATURES or FEATURES.count("GENDER") != 1:
    raise ValueError("The N0 retained feature panel is incompatible.")
if set(EXCLUDED_FEATURES) != {
    "UACR_LOG1P_FINAL",
    "EGFR_PREVIOUS_90D",
    "EGFR_CHANGE_LAST_MINUS_PREVIOUS_90D",
    "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS",
    "EGFR_SLOPE_PER_30_DAYS_90D",
}:
    raise ValueError("The N0 exclusion panel changed.")
NUMERIC_FEATURES = [feature for feature in FEATURES if feature != "GENDER"]

training = pd.read_parquet(
    SNAPSHOT_FILE,
    columns=[ID_COL, LABEL_COL, FOLD_COL, WEIGHT_COL, *FEATURES],
)
if len(training) != EXPECTED_PATIENTS:
    raise ValueError(f"Training row count changed: {len(training):,}")
if training[ID_COL].isna().any() or training[ID_COL].duplicated().any():
    raise ValueError("Training identifiers are missing or duplicated.")
labels = pd.to_numeric(training[LABEL_COL], errors="raise").astype(int).to_numpy()
weights = pd.to_numeric(training[WEIGHT_COL], errors="raise").to_numpy(float)
weights = weights / weights.mean()
folds = pd.to_numeric(training[FOLD_COL], errors="raise").astype(int).to_numpy()
fold_values = sorted(np.unique(folds).tolist())
if int(labels.sum()) != EXPECTED_POSITIVES or int((labels == 0).sum()) != EXPECTED_UNLABELED:
    raise ValueError("Training label counts changed.")
if fold_values != list(range(1, EXPECTED_FOLDS + 1)):
    raise ValueError(f"Unexpected folds: {fold_values}")
if not np.isfinite(weights).all() or np.any(weights <= 0):
    raise ValueError("Temporal balancing weights are invalid.")
if training["GENDER"].isna().any():
    raise ValueError("GENDER unexpectedly contains missing values.")

numeric_frame = training[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan)
gender_frame = training[["GENDER"]].astype("string")


software_lock = load_json(SOFTWARE_LOCK)
if software_lock is None or software_lock.get("status") != "PULSNAR_SOFTWARE_AND_PARAMETERS_LOCKED":
    raise ValueError("Official PULSNAR software lock is unavailable.")
official = software_lock["official_implementation"]
WHEEL = Path(official["archived_wheel"])
if not WHEEL.is_file() or sha256_file(WHEEL) != official["archived_wheel_sha256"]:
    raise ValueError("The archived official PULSNAR wheel is missing or changed.")
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


def transform_split(fit_positions, eval_positions):
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    fit_numeric = imputer.fit_transform(numeric_frame.iloc[fit_positions]).astype(np.float32)
    eval_numeric = imputer.transform(numeric_frame.iloc[eval_positions]).astype(np.float32)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    fit_gender = encoder.fit_transform(gender_frame.iloc[fit_positions])
    eval_gender = encoder.transform(gender_frame.iloc[eval_positions])
    names = list(NUMERIC_FEATURES) + [
        str(value) for value in encoder.get_feature_names_out(["GENDER"])
    ]
    fit_x = np.column_stack([fit_numeric, fit_gender]).astype(np.float32)
    eval_x = np.column_stack([eval_numeric, eval_gender]).astype(np.float32)
    if fit_x.shape[1] != EXPECTED_TRANSFORMED_FEATURES:
        raise ValueError(f"Unexpected transformed feature count: {fit_x.shape[1]}")
    if names[-2:] != ["GENDER_Female", "GENDER_Male"]:
        raise ValueError(f"Unexpected gender encoding: {names[-2:]}")
    if not np.isfinite(fit_x).all() or not np.isfinite(eval_x).all():
        raise ValueError("Mean-imputed scorer matrices contain nonfinite values.")
    return fit_x, eval_x, names


def discovery_matrix(fit_positions):
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    numeric = imputer.fit_transform(numeric_frame.iloc[fit_positions]).astype(np.float64)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    gender = encoder.fit_transform(gender_frame.iloc[fit_positions])
    matrix = np.column_stack([numeric, gender])
    matrix = StandardScaler().fit_transform(matrix).astype(np.float64)
    if matrix.shape[1] != EXPECTED_TRANSFORMED_FEATURES or not np.isfinite(matrix).all():
        raise ValueError("Invalid official PULSCAR discovery matrix.")
    return matrix


def suggest_parameters(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 30.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 40.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


def make_model(parameters, seed):
    return XGBClassifier(
        **dict(parameters),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=int(seed),
        verbosity=0,
    )


context = {
    "step": "5.12N4",
    "status": "EQUAL_BUDGET_NESTED_TRAINING_ONLY_PULSCAR_TUNING",
    "n3_completion": str(N3_FILE),
    "n3_completion_sha256": sha256_file(N3_FILE),
    "n0_policy_lock": str(N0_FILE),
    "n0_policy_lock_sha256": sha256_file(N0_FILE),
    "training_snapshot": str(SNAPSHOT_FILE),
    "training_snapshot_sha256": sha256_file(SNAPSHOT_FILE),
    "software_lock": str(SOFTWARE_LOCK),
    "software_lock_sha256": sha256_file(SOFTWARE_LOCK),
    "features": FEATURES,
    "excluded_features": EXCLUDED_FEATURES,
    "methods": METHODS,
    "outer_folds": fold_values,
    "inner_folds_per_outer": EXPECTED_FOLDS - 1,
    "trials_per_outer_pipeline": TRIALS_PER_OUTER_PIPELINE,
    "total_planned_trials_per_pipeline": TRIALS_PER_OUTER_PIPELINE * EXPECTED_FOLDS,
    "search_space": SEARCH_SPACE,
    "objective": "mean inner-fold observed-label average precision",
    "imputation_policy": "split-fitted arithmetic mean; no missingness indicators",
    "engine": "complete-matrix XGBoost; no native-NaN routing; no monotonic constraint",
    "training_weight_policy": "locked temporal balancing weights normalized to mean one",
    "official_pulscar_parameters": {**expected_shared, "scar": True},
    "automatic_final_winner_selection_prohibited": True,
    "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
    "validation_previously_accessed_for_superseded_pipeline": True,
    "test_previously_accessed_for_superseded_pipeline": True,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
DISCOVERY_DIR = PRIVATE_DIR / "official_pulscar_discovery"
STUDY_DIR = PRIVATE_DIR / "optuna_studies"
PREDICTION_DIR = PRIVATE_DIR / "outer_fold_predictions"
for directory in [OUTPUT_DIR, PRIVATE_DIR, DISCOVERY_DIR, STUDY_DIR, PREDICTION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
CONTEXT_FILE = OUTPUT_DIR / "context.json"
if CONTEXT_FILE.is_file():
    existing = load_json(CONTEXT_FILE)
    if existing != json_safe(context):
        raise ValueError("Existing N4 context is incompatible.")
else:
    atomic_json(CONTEXT_FILE, context)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
    "--force-reinstall", str(WHEEL),
])
from PULSNAR import PULSNAR  # noqa: E402


def run_pulscar_discovery(scope_name, fit_positions, seed):
    scope_dir = DISCOVERY_DIR / scope_name
    scope_dir.mkdir(parents=True, exist_ok=True)
    audit_path = scope_dir / "discovery_audit.json"
    selected_path = scope_dir / "selected_probable_positive.parquet"
    fit_ids = training.iloc[fit_positions][ID_COL].to_numpy()
    y_fit = labels[fit_positions]

    if audit_path.is_file() and selected_path.is_file():
        audit = load_json(audit_path)
        table = pd.read_parquet(selected_path)
        if list(table.columns) != [ID_COL, "selected_probable_positive"]:
            raise ValueError(f"Invalid discovery checkpoint: {selected_path}")
        if not np.array_equal(table[ID_COL].to_numpy(), fit_ids):
            raise ValueError(f"Discovery checkpoint IDs changed: {selected_path}")
        selected = table["selected_probable_positive"].astype(bool).to_numpy()
        if int(selected.sum()) != int(audit["Selected_probable_positive"]):
            raise ValueError(f"Discovery checkpoint count changed: {selected_path}")
        return selected, audit

    matrix = discovery_matrix(fit_positions)
    parameters = dict(expected_shared)
    parameters["scar"] = True
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"pulscar_{scope_name}_", dir=str(scope_dir)) as temp:
        temp = Path(temp)
        prior_cwd = Path.cwd()
        try:
            os.chdir(temp)
            estimator = PULSNAR.PULSNARClassifier(
                classifier="xgboost", rseed=int(seed), **parameters
            )
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                result = estimator.pulsnar(
                    matrix,
                    y_fit,
                    tru_label=None,
                    rec_list=np.arange(len(y_fit), dtype=np.int64),
                )
        finally:
            os.chdir(prior_cwd)
        prediction_source = temp / Path(result["prediction_file"]).name
        alpha_source = temp / Path(result["alpha_file"]).name
        if not prediction_source.is_file() or not alpha_source.is_file():
            raise FileNotFoundError("Official PULSCAR did not produce its locked outputs.")
        shutil.copy2(prediction_source, scope_dir / "official_predictions.tsv")
        shutil.copy2(alpha_source, scope_dir / "official_alpha_estimates.tsv")

    predictions = pd.read_csv(scope_dir / "official_predictions.tsv", sep="\t")
    required = {"rec_id", "ml_label", "calibrated_prob"}
    if not required.issubset(predictions.columns) or len(predictions) != len(y_fit):
        raise ValueError("Official PULSCAR prediction schema changed.")
    predictions["rec_id"] = pd.to_numeric(predictions["rec_id"], errors="raise").astype(int)
    predictions["ml_label"] = pd.to_numeric(predictions["ml_label"], errors="raise").astype(int)
    predictions["calibrated_prob"] = pd.to_numeric(
        predictions["calibrated_prob"], errors="raise"
    ).astype(float)
    predictions = predictions.sort_values("rec_id").reset_index(drop=True)
    if not np.array_equal(predictions["rec_id"].to_numpy(), np.arange(len(y_fit))):
        raise ValueError("Official discovery record positions changed.")
    if not np.array_equal(predictions["ml_label"].to_numpy(), y_fit):
        raise ValueError("Official discovery labels changed.")
    posterior = predictions["calibrated_prob"].to_numpy(float)
    if not np.isfinite(posterior).all() or np.any((posterior < 0) | (posterior > 1)):
        raise ValueError("Official discovery returned invalid probabilities.")

    alpha = float(result["estimated_alpha"])
    if not 0 <= alpha <= 1:
        raise ValueError("Official PULSCAR alpha is outside [0,1].")
    unlabeled_local = np.flatnonzero(y_fit == 0)
    selection_count = int(math.floor(alpha * len(unlabeled_local) + 0.5))
    ranking = pd.DataFrame({
        "position": unlabeled_local,
        "posterior": posterior[unlabeled_local],
        "patient_id": fit_ids[unlabeled_local].astype(str),
    })
    ranking["tie_hash"] = ranking["patient_id"].map(
        lambda value: hashlib.sha256(
            f"official_pulscar|{int(seed)}|{value}".encode("utf-8")
        ).hexdigest()
    )
    ranking = ranking.sort_values(
        ["posterior", "tie_hash"], ascending=[False, True], kind="stable"
    )
    selected = np.zeros(len(y_fit), dtype=bool)
    selected[ranking.iloc[:selection_count]["position"].to_numpy(int)] = True
    if np.any(selected & (y_fit == 1)):
        raise ValueError("PULSCAR selection unexpectedly includes documented positives.")
    pd.DataFrame({
        ID_COL: fit_ids,
        "selected_probable_positive": selected,
    }).to_parquet(selected_path, index=False)
    audit = {
        "Method": "official_pulscar",
        "Scope": scope_name,
        "Training_rows": int(len(y_fit)),
        "Documented_positive": int(y_fit.sum()),
        "Unlabeled": int((y_fit == 0).sum()),
        "Estimated_alpha": alpha,
        "Selected_probable_positive": int(selected.sum()),
        "Runtime_minutes": (time.perf_counter() - started) / 60.0,
        "Official_stdout": stdout_buffer.getvalue(),
        "Official_stderr": stderr_buffer.getvalue(),
    }
    atomic_json(audit_path, audit)
    del matrix
    return selected, audit


oof = {method: np.full(EXPECTED_PATIENTS, np.nan, dtype=float) for method in METHODS}
outer_rows = []
best_parameter_rows = []
discovery_rows = []
preprocessing_rows = []
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 112)
print("STEP 5.12N4 — EQUAL-BUDGET NESTED TRAINING-ONLY PULSCAR TUNING")
print("=" * 112)
print(f"Budget: {TRIALS_PER_OUTER_PIPELINE} trials per outer fold and pipeline")
print("This run is resumable; rerun the same cell after a runtime interruption.")

for outer_fold in fold_values:
    outer_fit_positions = np.flatnonzero(folds != outer_fold)
    outer_eval_positions = np.flatnonzero(folds == outer_fold)
    outer_eval_ids = training.iloc[outer_eval_positions][ID_COL].to_numpy()
    existing_outer = {
        method: read_oof(
            PREDICTION_DIR / f"{method}__outer_fold_{outer_fold}.parquet",
            outer_eval_ids, ID_COL,
        )
        for method in METHODS
    }

    inner_bundles = []
    for inner_eval_fold in [value for value in fold_values if value != outer_fold]:
        inner_fit_positions = np.flatnonzero(
            (folds != outer_fold) & (folds != inner_eval_fold)
        )
        inner_eval_positions = np.flatnonzero(folds == inner_eval_fold)
        fit_x, eval_x, names = transform_split(inner_fit_positions, inner_eval_positions)
        inner_bundles.append({
            "inner_eval_fold": inner_eval_fold,
            "fit_positions": inner_fit_positions,
            "eval_positions": inner_eval_positions,
            "fit_x": fit_x,
            "eval_x": eval_x,
            "base_target": labels[inner_fit_positions],
            "eval_target": labels[inner_eval_positions],
            "fit_weight": weights[inner_fit_positions],
        })
        preprocessing_rows.append({
            "Outer_fold": outer_fold,
            "Inner_eval_fold": inner_eval_fold,
            "Fit_patients": len(inner_fit_positions),
            "Evaluation_patients": len(inner_eval_positions),
            "Transformed_features": len(names),
            "Fit_nonfinite_after_transform": int((~np.isfinite(fit_x)).sum()),
            "Evaluation_nonfinite_after_transform": int((~np.isfinite(eval_x)).sum()),
        })

    pulscar_targets = []
    for bundle in inner_bundles:
        scope = f"outer_{outer_fold}__inner_eval_{bundle['inner_eval_fold']}"
        selected, audit = run_pulscar_discovery(
            scope,
            bundle["fit_positions"],
            PUL_SEED + method_seed("official_pulscar") + outer_fold * 100 + bundle["inner_eval_fold"],
        )
        target = bundle["base_target"].copy()
        target[selected] = 1
        pulscar_targets.append(target)
        discovery_rows.append(audit)
    method_targets = {
        "naive_pn": [bundle["base_target"] for bundle in inner_bundles],
        "official_pulscar": pulscar_targets,
    }

    for method in METHODS:
        study_name = f"{method}__outer_fold_{outer_fold}"
        database = STUDY_DIR / f"{study_name}.sqlite"
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{database.as_posix()}")
        sampler = optuna.samplers.TPESampler(
            seed=SEED + method_seed(method) + outer_fold,
            n_startup_trials=min(8, TRIALS_PER_OUTER_PIPELINE),
        )
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

        def objective(trial):
            parameters = suggest_parameters(trial)
            fold_scores = []
            for inner_index, bundle in enumerate(inner_bundles):
                model = make_model(
                    parameters,
                    SEED + method_seed(method) + outer_fold * 1000
                    + trial.number * 10 + inner_index,
                )
                model.fit(
                    bundle["fit_x"],
                    method_targets[method][inner_index],
                    sample_weight=bundle["fit_weight"],
                )
                probability = np.asarray(
                    model.predict_proba(bundle["eval_x"])[:, 1], float
                )
                fold_scores.append(
                    average_precision_score(bundle["eval_target"], probability)
                )
                del model
            trial.set_user_attr("inner_fold_AP", [float(value) for value in fold_scores])
            return float(np.mean(fold_scores))

        complete_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        remaining = TRIALS_PER_OUTER_PIPELINE - len(complete_trials)
        if remaining > 0:
            started = time.perf_counter()
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
            print(
                f"Completed {method}, outer fold {outer_fold}, {remaining} new trials, "
                f"{(time.perf_counter() - started) / 60:.2f} minutes."
            )
        study.trials_dataframe().to_csv(
            STUDY_DIR / f"{study_name}__trials.csv", index=False
        )
        complete_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(complete_trials) != TRIALS_PER_OUTER_PIPELINE:
            raise RuntimeError(
                f"{study_name} has {len(complete_trials)} completed trials; expected "
                f"{TRIALS_PER_OUTER_PIPELINE}. Rerun the same cell to resume."
            )
        best_parameters = dict(study.best_trial.params)
        best_parameter_rows.append({
            "Method": method,
            "Outer_fold": outer_fold,
            "Best_trial": int(study.best_trial.number),
            "Best_inner_mean_AP": float(study.best_value),
            **best_parameters,
        })

        selected_count = 0
        if existing_outer[method] is None:
            outer_fit_x, outer_eval_x, _ = transform_split(
                outer_fit_positions, outer_eval_positions
            )
            outer_target = labels[outer_fit_positions].copy()
            if method == "official_pulscar":
                outer_selected, outer_audit = run_pulscar_discovery(
                    f"outer_{outer_fold}__final_fit",
                    outer_fit_positions,
                    PUL_SEED + method_seed(method) + outer_fold * 10_000,
                )
                outer_target[outer_selected] = 1
                selected_count = int(outer_selected.sum())
                discovery_rows.append(outer_audit)
            model = make_model(
                best_parameters,
                SEED + method_seed(method) + outer_fold * 10_000,
            )
            model.fit(
                outer_fit_x,
                outer_target,
                sample_weight=weights[outer_fit_positions],
            )
            probability = np.asarray(model.predict_proba(outer_eval_x)[:, 1], float)
            save_oof(
                PREDICTION_DIR / f"{method}__outer_fold_{outer_fold}.parquet",
                outer_eval_ids, probability, ID_COL,
            )
            del model, outer_fit_x, outer_eval_x
        else:
            probability = existing_outer[method]
            if method == "official_pulscar":
                outer_selected, outer_audit = run_pulscar_discovery(
                    f"outer_{outer_fold}__final_fit",
                    outer_fit_positions,
                    PUL_SEED + method_seed(method) + outer_fold * 10_000,
                )
                selected_count = int(outer_selected.sum())
                discovery_rows.append(outer_audit)

        oof[method][outer_eval_positions] = probability
        outer_rows.append({
            "Method": method,
            "Outer_fold": outer_fold,
            "Patients": len(outer_eval_positions),
            "Selected_probable_positive_in_outer_fit": selected_count,
            **metrics(labels[outer_eval_positions], probability),
        })
        gc.collect()

    del inner_bundles, method_targets, pulscar_targets
    gc.collect()


for method in METHODS:
    if not np.isfinite(oof[method]).all():
        raise ValueError(f"Incomplete nested OOF predictions for {method}.")
    save_oof(
        PRIVATE_DIR / f"{method}__nested_tuned_complete_oof.parquet",
        training[ID_COL].to_numpy(), oof[method], ID_COL,
    )

summary = pd.DataFrame([
    {"Method": method, **metrics(labels, oof[method])} for method in METHODS
]).sort_values(
    ["Average_precision", "Log_loss", "Brier_score"],
    ascending=[False, True, True], kind="stable",
).reset_index(drop=True)
outer_results = pd.DataFrame(outer_rows).sort_values(
    ["Method", "Outer_fold"], kind="stable"
).reset_index(drop=True)
best_parameters = pd.DataFrame(best_parameter_rows).sort_values(
    ["Method", "Outer_fold"], kind="stable"
).reset_index(drop=True)
discovery_audit = pd.DataFrame(discovery_rows).drop_duplicates(
    ["Method", "Scope"], keep="last"
).sort_values(["Method", "Scope"], kind="stable").reset_index(drop=True)
preprocessing_audit = pd.DataFrame(preprocessing_rows).drop_duplicates(
    ["Outer_fold", "Inner_eval_fold"], keep="last"
).sort_values(["Outer_fold", "Inner_eval_fold"], kind="stable").reset_index(drop=True)

decision = {
    "step": "5.12N4",
    "status": "EQUAL_BUDGET_NESTED_TRAINING_ONLY_PULSCAR_TUNING_COMPLETE",
    "methods": METHODS,
    "trials_per_outer_pipeline": TRIALS_PER_OUTER_PIPELINE,
    "total_completed_trials_per_pipeline": TRIALS_PER_OUTER_PIPELINE * EXPECTED_FOLDS,
    "mean_imputation_used": True,
    "missingness_indicators_used": False,
    "native_nan_routing_used": False,
    "automatic_final_winner_selected": False,
    "next_step": (
        "Use these tuned nested OOF scores in a Training-only post-tuning gate "
        "and hidden-case utility audit before selecting any final pipeline."
    ),
    "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
    "model_promoted_or_replaced": False,
}

summary.to_csv(OUTPUT_DIR / "nested_tuned_oof_summary.csv", index=False)
outer_results.to_csv(OUTPUT_DIR / "outer_fold_metrics.csv", index=False)
best_parameters.to_csv(OUTPUT_DIR / "outer_fold_best_parameters.csv", index=False)
discovery_audit.to_csv(OUTPUT_DIR / "official_pulscar_discovery_audit.csv", index=False)
preprocessing_audit.to_csv(OUTPUT_DIR / "nested_preprocessing_audit.csv", index=False)
atomic_json(OUTPUT_DIR / "tuning_decision.json", decision)
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
print("STEP 5.12N4 — EQUAL-BUDGET NESTED TRAINING-ONLY PULSCAR TUNING COMPLETE")
print("=" * 112)
print("\nPACKAGE VERSIONS")
print(json.dumps(VERSIONS, indent=2, ensure_ascii=False))
print("\nNESTED TUNED OOF SUMMARY")
display(summary)
print("\nOUTER-FOLD METRICS")
display(outer_results)
print("\nOUTER-FOLD BEST PARAMETERS")
display(best_parameters)
print("\nOFFICIAL PULSCAR DISCOVERY AUDIT")
display(discovery_audit)
print("\nNESTED PREPROCESSING AUDIT")
display(preprocessing_audit)
print("\nTUNING STATUS")
print(json.dumps(decision, indent=2, ensure_ascii=False))
print(f"\nSaved run: {OUTPUT_DIR}")
print("Validation accessed in this step: No")
print("Test accessed in this step: No")
print("Missingness indicators used: No")
print("Native-NaN routing used: No")
print("Final winner selected or model replaced: No")
