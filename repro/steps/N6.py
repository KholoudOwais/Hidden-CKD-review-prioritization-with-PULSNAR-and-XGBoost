"""STEP 5.12N6 — Full-Training PULSCAR + mean-imputed XGBoost lock.

Uses only the 20 retained N0 features. uACR and the four sparse longitudinal
eGFR features are not read. Forty Optuna trials use the five locked Training
folds and the leakage-safe PULSCAR fold selections produced in N4. Official
PULSCAR is then run once on all Training patients and the final model is fit,
saved, hashed, and locked. Validation/Test are never loaded.
"""

from google.colab import drive
drive.mount("/content/drive")

from datetime import datetime, timezone
from pathlib import Path
import contextlib, gc, hashlib, importlib.metadata, importlib.util, io, json
import math, os, shutil, subprocess, sys, tempfile, time
from packaging.version import Version

try:
    sklearn_version_before = importlib.metadata.version("scikit-learn")
except importlib.metadata.PackageNotFoundError:
    sklearn_version_before = "0"
if Version(sklearn_version_before) < Version("1.9.0"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn>=1.9,<1.10"])
for package, requirement in {
    "xgboost": "xgboost", "catboost": "catboost==1.2.10",
    "optuna": "optuna>=4,<5", "joblib": "joblib",
}.items():
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", requirement])

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn, xgboost
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

VERSIONS = {
    "python": sys.version.split()[0], "numpy": np.__version__,
    "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    "xgboost": xgboost.__version__, "optuna": optuna.__version__,
    "joblib": importlib.metadata.version("joblib"),
}
if Version(VERSIONS["scikit_learn"]) < Version("1.9.0"):
    raise RuntimeError("Restart the Colab runtime once, then rerun this complete cell.")

SEED = 71_000_046
PUL_SEED = 123
FULL_TUNING_TRIALS = 40
EXPECTED_PATIENTS = 69_701
EXPECTED_POSITIVES = 5_042
EXPECTED_UNLABELED = 64_659
EXPECTED_FOLDS = 5
EXPECTED_FEATURES = 20
EXPECTED_TRANSFORMED_FEATURES = 21
WEIGHT_COL = "TEMPORAL_BALANCING_WEIGHT"

SEARCH_SPACE = {
    "n_estimators": [200, 700, 100], "learning_rate": [0.01, 0.15, "log"],
    "reg_lambda": [1e-4, 30.0, "log"], "colsample_bytree": [0.5, 1.0],
    "max_depth": [2, 6], "min_child_weight": [1.0, 40.0],
    "subsample": [0.6, 1.0],
}

PROJECT_ROOT = Path("/content/drive/MyDrive/CKD_Project")
ROOT = PROJECT_ROOT / "outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling"
N5_ROOT = ROOT / "05_12N5_training_only_post_tuning_pipeline_selection"
OUTPUT_ROOT = ROOT / "05_12N6_full_training_pulscar_mean_xgb_final_refit_and_lock"
SOFTWARE_LOCK = ROOT / (
    "05_6D1_pulsnar_software_and_parameter_lock/"
    "a7bf680d472f693279074683/pulsnar_software_and_parameter_lock.json"
)

def load_json(path):
    try: return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception: return None

def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""): digest.update(chunk)
    return digest.hexdigest()

def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value

def json_bytes(payload):
    return json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")

def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(json_safe(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def atomic_joblib(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary); os.replace(temporary, path)

def make_model(parameters, seed):
    return XGBClassifier(
        **dict(parameters), objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=int(seed), verbosity=0,
    )

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

# Resolve the completed N5 -> N4 -> N0 chain.
n5_candidates = []
if N5_ROOT.exists():
    for path in N5_ROOT.rglob("completion.json"):
        payload = load_json(path)
        if payload and payload.get("status") == "TRAINING_ONLY_PIPELINE_SELECTED_FOR_FULL_TRAINING_REFIT":
            n5_candidates.append((path, payload))
if not n5_candidates: raise FileNotFoundError("Run N5 before N6.")
N5_FILE, n5 = max(n5_candidates, key=lambda item: item[0].stat().st_mtime)
if n5.get("selected_method") != "official_pulscar" or not n5.get("all_locked_gates_passed_by_pulscar"):
    raise ValueError("N5 did not select official_pulscar through all locked gates.")
if n5.get("validation_accessed_in_this_step") or n5.get("test_accessed_in_this_step"):
    raise ValueError("N5 reports held-out access.")

N4_FILE = Path(n5["n4_completion"]); n4 = load_json(N4_FILE)
if n4 is None or n4.get("status") != "EQUAL_BUDGET_NESTED_TRAINING_ONLY_PULSCAR_TUNING_COMPLETE":
    raise ValueError("N5 points to an invalid N4 completion.")
if sha256_file(N4_FILE) != n5["n4_completion_sha256"]: raise ValueError("N4 changed after N5.")
N4_RUN = N4_FILE.parent
N4_DISCOVERY = N4_RUN / "PRIVATE_INTERNAL_DO_NOT_SHARE" / "official_pulscar_discovery"
N4_BEST = N4_RUN / "outer_fold_best_parameters.csv"
if not N4_BEST.is_file(): raise FileNotFoundError(N4_BEST)

N0_FILE = Path(n4["n0_policy_lock"]); n0 = load_json(N0_FILE)
if n0 is None or n0.get("status") != "TRAINING_ONLY_HIGH_MISSINGNESS_POLICY_LOCKED":
    raise ValueError("Invalid N0 lock.")
if sha256_file(N0_FILE) != n4["n0_policy_lock_sha256"]: raise ValueError("N0 changed after N4.")
SNAPSHOT_FILE = Path(n4["training_snapshot"])
if not SNAPSHOT_FILE.is_file() or sha256_file(SNAPSHOT_FILE) != n4["training_snapshot_sha256"]:
    raise ValueError("Training snapshot is missing or changed.")
d0 = load_json(n0["source_protocol"])
ID_COL = d0["cohort"]["patient_id_column"]
LABEL_COL = d0["cohort"]["observed_label_column"]
FOLD_COL = d0["cohort"]["locked_fold_column"]
FEATURES = list(n0["retained_features"]); EXCLUDED_FEATURES = list(n0["excluded_features"])
expected_excluded = {
    "UACR_LOG1P_FINAL", "EGFR_PREVIOUS_90D", "EGFR_CHANGE_LAST_MINUS_PREVIOUS_90D",
    "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS", "EGFR_SLOPE_PER_30_DAYS_90D",
}
if len(FEATURES) != EXPECTED_FEATURES or "GENDER" not in FEATURES: raise ValueError("Invalid retained panel.")
if set(EXCLUDED_FEATURES) != expected_excluded or any(x in FEATURES for x in expected_excluded):
    raise ValueError("Sparse-feature exclusion policy changed.")
NUMERIC_FEATURES = [x for x in FEATURES if x != "GENDER"]

training = pd.read_parquet(SNAPSHOT_FILE, columns=[ID_COL, LABEL_COL, FOLD_COL, WEIGHT_COL, *FEATURES])
if len(training) != EXPECTED_PATIENTS or training[ID_COL].isna().any() or training[ID_COL].duplicated().any():
    raise ValueError("Unexpected Training cohort.")
labels = pd.to_numeric(training[LABEL_COL], errors="raise").astype(int).to_numpy()
weights = pd.to_numeric(training[WEIGHT_COL], errors="raise").to_numpy(float); weights /= weights.mean()
folds = pd.to_numeric(training[FOLD_COL], errors="raise").astype(int).to_numpy()
fold_values = sorted(np.unique(folds).tolist())
if int(labels.sum()) != EXPECTED_POSITIVES or int((labels == 0).sum()) != EXPECTED_UNLABELED:
    raise ValueError("Training counts changed.")
if fold_values != list(range(1, EXPECTED_FOLDS + 1)): raise ValueError("Locked folds changed.")
if not np.isfinite(weights).all() or np.any(weights <= 0): raise ValueError("Invalid weights.")
if training["GENDER"].isna().any(): raise ValueError("GENDER contains missing values.")
numeric_frame = training[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
gender_frame = training[["GENDER"]].astype("string")

software_lock = load_json(SOFTWARE_LOCK)
if software_lock is None or software_lock.get("status") != "PULSNAR_SOFTWARE_AND_PARAMETERS_LOCKED":
    raise ValueError("Official software lock unavailable.")
official = software_lock["official_implementation"]; WHEEL = Path(official["archived_wheel"])
if not WHEEL.is_file() or sha256_file(WHEEL) != official["archived_wheel_sha256"]:
    raise ValueError("Official wheel missing or changed.")
shared = dict(software_lock["shared_PULSNAR_parameters"]); shared["scar"] = True

def fit_preprocessor(fit_positions):
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    numeric = imputer.fit_transform(numeric_frame.iloc[fit_positions]).astype(np.float32)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    gender = encoder.fit_transform(gender_frame.iloc[fit_positions])
    names = NUMERIC_FEATURES + [str(x) for x in encoder.get_feature_names_out(["GENDER"])]
    matrix = np.column_stack([numeric, gender]).astype(np.float32)
    if matrix.shape[1] != EXPECTED_TRANSFORMED_FEATURES or not np.isfinite(matrix).all():
        raise ValueError("Invalid mean-imputed matrix.")
    if names[-2:] != ["GENDER_Female", "GENDER_Male"]: raise ValueError("Gender encoding changed.")
    return matrix, {"numeric_features": NUMERIC_FEATURES, "categorical_feature": "GENDER",
                    "imputer": imputer, "gender_encoder": encoder, "transformed_features": names,
                    "missingness_indicators": False, "native_nan_routing": False}

def transform_positions(positions, preprocessor):
    numeric = preprocessor["imputer"].transform(numeric_frame.iloc[positions]).astype(np.float32)
    gender = preprocessor["gender_encoder"].transform(gender_frame.iloc[positions]).astype(np.float32)
    matrix = np.column_stack([numeric, gender]).astype(np.float32)
    if not np.isfinite(matrix).all(): raise ValueError("Nonfinite transformed values.")
    return matrix

def load_fold_selection(fold, fit_positions):
    path = N4_DISCOVERY / f"outer_{fold}__final_fit" / "selected_probable_positive.parquet"
    if not path.is_file(): raise FileNotFoundError(path)
    table = pd.read_parquet(path)
    expected_ids = training.iloc[fit_positions][ID_COL].to_numpy()
    if list(table.columns) != [ID_COL, "selected_probable_positive"] or not np.array_equal(table[ID_COL].to_numpy(), expected_ids):
        raise ValueError(f"N4 fold selection changed: {path}")
    return table["selected_probable_positive"].astype(bool).to_numpy(), path

fold_bundles = []
fold_selection_hashes = {}
for fold in fold_values:
    fit_pos = np.flatnonzero(folds != fold); eval_pos = np.flatnonzero(folds == fold)
    fit_x, preprocessor = fit_preprocessor(fit_pos); eval_x = transform_positions(eval_pos, preprocessor)
    selected, selection_path = load_fold_selection(fold, fit_pos)
    target = labels[fit_pos].copy(); target[selected] = 1
    fold_selection_hashes[str(fold)] = sha256_file(selection_path)
    fold_bundles.append({"fold": fold, "fit_x": fit_x, "eval_x": eval_x,
                         "target": target, "eval_target": labels[eval_pos], "weight": weights[fit_pos],
                         "selected": int(selected.sum())})

context = {
    "step": "5.12N6", "status": "FULL_TRAINING_PULSCAR_MEAN_XGB_REFIT_AND_LOCK",
    "n5_completion": str(N5_FILE), "n5_completion_sha256": sha256_file(N5_FILE),
    "n4_completion_sha256": sha256_file(N4_FILE), "n4_fold_selection_sha256": fold_selection_hashes,
    "n0_policy_lock_sha256": sha256_file(N0_FILE), "training_snapshot_sha256": sha256_file(SNAPSHOT_FILE),
    "software_lock_sha256": sha256_file(SOFTWARE_LOCK), "features": FEATURES,
    "excluded_features": EXCLUDED_FEATURES, "full_training_tuning_trials": FULL_TUNING_TRIALS,
    "search_space": SEARCH_SPACE, "objective": "mean five-fold observed-label average precision",
    "pipeline": "official PULSCAR discovery -> mean-imputed XGBoost",
    "uacr_used_anywhere_in_modeling": False, "missingness_indicators_used": False,
    "native_nan_routing_used": False, "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
    "validation_previously_accessed_for_superseded_pipeline": True,
    "test_previously_accessed_for_superseded_pipeline": True,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id; PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
DISCOVERY_DIR = PRIVATE_DIR / "full_training_official_pulscar_discovery"
for directory in [OUTPUT_DIR, PRIVATE_DIR, DISCOVERY_DIR]: directory.mkdir(parents=True, exist_ok=True)
CONTEXT_FILE = OUTPUT_DIR / "context.json"
if CONTEXT_FILE.is_file() and load_json(CONTEXT_FILE) != json_safe(context): raise ValueError("Incompatible N6 context.")
if not CONTEXT_FILE.is_file(): atomic_json(CONTEXT_FILE, context)

# Single 40-trial full-Training study, seeded with N4 outer-best parameter sets.
study_name = "official_pulscar_mean_xgb_full_training"
database = PRIVATE_DIR / "full_training_optuna.sqlite"
sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8)
study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler,
                            storage=f"sqlite:///{database.as_posix()}", load_if_exists=True)
if len(study.trials) == 0:
    best = pd.read_csv(N4_BEST)
    best = best[best["Method"].eq("official_pulscar")]
    parameter_columns = list(SEARCH_SPACE)
    seen = set()
    for _, row in best.iterrows():
        params = {key: (int(row[key]) if key in {"n_estimators", "max_depth"} else float(row[key])) for key in parameter_columns}
        signature = json.dumps(params, sort_keys=True)
        if signature not in seen: study.enqueue_trial(params); seen.add(signature)

def objective(trial):
    params = suggest_parameters(trial); scores = []
    for index, bundle in enumerate(fold_bundles):
        model = make_model(params, SEED + trial.number * 10 + index)
        model.fit(bundle["fit_x"], bundle["target"], sample_weight=bundle["weight"])
        probability = model.predict_proba(bundle["eval_x"])[:, 1]
        scores.append(average_precision_score(bundle["eval_target"], probability)); del model
    trial.set_user_attr("fold_AP", [float(x) for x in scores])
    return float(np.mean(scores))

complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
remaining = FULL_TUNING_TRIALS - len(complete)
print(f"Completed Optuna trials already present: {len(complete)}/{FULL_TUNING_TRIALS}")
if remaining > 0:
    started = time.perf_counter(); study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    print(f"Completed {remaining} new trials in {(time.perf_counter()-started)/60:.2f} minutes.")
study.trials_dataframe().to_csv(OUTPUT_DIR / "full_training_optuna_trials.csv", index=False)
complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if len(complete) != FULL_TUNING_TRIALS: raise RuntimeError("Incomplete Optuna budget; rerun this cell to resume.")
best_parameters = dict(study.best_trial.params)

# Install audited official implementation and run full-Training discovery once.
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--force-reinstall", str(WHEEL)])
from PULSNAR import PULSNAR  # noqa: E402

selection_path = DISCOVERY_DIR / "selected_probable_positive.parquet"
audit_path = DISCOVERY_DIR / "discovery_audit.json"
discovery_preprocessor_path = OUTPUT_DIR / "locked_pulscar_discovery_preprocessor.joblib"
if selection_path.is_file() and audit_path.is_file() and discovery_preprocessor_path.is_file():
    selection_table = pd.read_parquet(selection_path); discovery_audit = load_json(audit_path)
    if list(selection_table.columns) != [ID_COL, "selected_probable_positive"] or not np.array_equal(selection_table[ID_COL].to_numpy(), training[ID_COL].to_numpy()):
        raise ValueError("Saved full discovery selection changed.")
    full_selected = selection_table["selected_probable_positive"].astype(bool).to_numpy()
    if int(full_selected.sum()) != int(discovery_audit["selected_probable_positive"]):
        raise ValueError("Saved full discovery selection count changed.")
else:
    full_positions = np.arange(EXPECTED_PATIENTS)
    full_numeric_imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    d_numeric = full_numeric_imputer.fit_transform(numeric_frame).astype(np.float64)
    d_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    d_gender = d_encoder.fit_transform(gender_frame)
    scaler = StandardScaler(); discovery_x = scaler.fit_transform(np.column_stack([d_numeric, d_gender])).astype(np.float64)
    if discovery_x.shape[1] != EXPECTED_TRANSFORMED_FEATURES or not np.isfinite(discovery_x).all(): raise ValueError("Invalid full discovery matrix.")
    atomic_joblib(discovery_preprocessor_path, {"numeric_features": NUMERIC_FEATURES, "categorical_feature": "GENDER",
        "imputer": full_numeric_imputer, "gender_encoder": d_encoder, "scaler": scaler,
        "missingness_indicators": False, "uacr_used": False})
    stdout_buffer = io.StringIO(); stderr_buffer = io.StringIO(); started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="full_pulscar_", dir=str(DISCOVERY_DIR)) as temp:
        temp = Path(temp); prior = Path.cwd()
        try:
            os.chdir(temp); estimator = PULSNAR.PULSNARClassifier(classifier="xgboost", rseed=PUL_SEED + 60_000, **shared)
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                result = estimator.pulsnar(discovery_x, labels, tru_label=None, rec_list=np.arange(EXPECTED_PATIENTS, dtype=np.int64))
        finally: os.chdir(prior)
        prediction_source = temp / Path(result["prediction_file"]).name; alpha_source = temp / Path(result["alpha_file"]).name
        if not prediction_source.is_file() or not alpha_source.is_file(): raise FileNotFoundError("Official PULSCAR outputs missing.")
        shutil.copy2(prediction_source, DISCOVERY_DIR / "official_predictions.tsv")
        shutil.copy2(alpha_source, DISCOVERY_DIR / "official_alpha_estimates.tsv")
    predictions = pd.read_csv(DISCOVERY_DIR / "official_predictions.tsv", sep="\t").sort_values("rec_id").reset_index(drop=True)
    if len(predictions) != EXPECTED_PATIENTS: raise ValueError("Full discovery row count changed.")
    rec_id = pd.to_numeric(predictions["rec_id"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(rec_id, np.arange(EXPECTED_PATIENTS)): raise ValueError("Full discovery record order changed.")
    if not np.array_equal(pd.to_numeric(predictions["ml_label"]).astype(int), labels): raise ValueError("Full discovery labels changed.")
    posterior = pd.to_numeric(predictions["calibrated_prob"], errors="raise").to_numpy(float)
    if not np.isfinite(posterior).all() or np.any((posterior < 0) | (posterior > 1)): raise ValueError("Invalid full discovery probabilities.")
    alpha = float(result["estimated_alpha"])
    if not 0 <= alpha <= 1: raise ValueError("Full PULSCAR alpha is outside [0,1].")
    unlabeled = np.flatnonzero(labels == 0)
    count = int(math.floor(alpha * len(unlabeled) + 0.5))
    ranking = pd.DataFrame({"position": unlabeled, "posterior": posterior[unlabeled], "id": training.iloc[unlabeled][ID_COL].astype(str).to_numpy()})
    ranking["tie"] = ranking["id"].map(lambda x: hashlib.sha256(f"full_pulscar|{x}".encode()).hexdigest())
    ranking = ranking.sort_values(["posterior", "tie"], ascending=[False, True], kind="stable")
    full_selected = np.zeros(EXPECTED_PATIENTS, bool); full_selected[ranking.iloc[:count]["position"].to_numpy(int)] = True
    pd.DataFrame({ID_COL: training[ID_COL].to_numpy(), "selected_probable_positive": full_selected}).to_parquet(selection_path, index=False)
    discovery_audit = {"method": "official_pulscar", "training_patients": EXPECTED_PATIENTS,
        "documented_positive": EXPECTED_POSITIVES, "unlabeled": EXPECTED_UNLABELED,
        "estimated_alpha": alpha, "selected_probable_positive": int(full_selected.sum()),
        "runtime_minutes": (time.perf_counter()-started)/60, "official_stdout": stdout_buffer.getvalue(),
        "official_stderr": stderr_buffer.getvalue(), "uacr_used": False}
    atomic_json(audit_path, discovery_audit); del discovery_x
if np.any(full_selected & (labels == 1)): raise ValueError("Discovery selected documented positives.")

# Final full-Training refit and immutable artifacts.
full_positions = np.arange(EXPECTED_PATIENTS)
full_x, downstream_preprocessor = fit_preprocessor(full_positions)
final_target = labels.copy(); final_target[full_selected] = 1
model = make_model(best_parameters, SEED + 90_000)
model.fit(full_x, final_target, sample_weight=weights)
training_probability = np.asarray(model.predict_proba(full_x)[:, 1], float)
if not np.isfinite(training_probability).all(): raise ValueError("Invalid final Training scores.")

model_path = OUTPUT_DIR / "locked_official_pulscar_mean_xgb_model.joblib"
model_json_path = OUTPUT_DIR / "locked_official_pulscar_mean_xgb_model.json"
preprocessor_path = OUTPUT_DIR / "locked_mean_imputation_preprocessor.joblib"
manifest_path = OUTPUT_DIR / "locked_transformed_feature_manifest.csv"
atomic_joblib(model_path, model); atomic_joblib(preprocessor_path, downstream_preprocessor)
temp_json = model_json_path.with_suffix(".tmp.json"); model.save_model(temp_json); os.replace(temp_json, model_json_path)
manifest = pd.DataFrame({"transformed_position": np.arange(EXPECTED_TRANSFORMED_FEATURES),
                         "transformed_feature": downstream_preprocessor["transformed_features"]})
manifest.to_csv(manifest_path, index=False)
score_path = PRIVATE_DIR / "full_training_fitted_scores.parquet"
pd.DataFrame({ID_COL: training[ID_COL].to_numpy(), "probability": training_probability}).to_parquet(score_path, index=False)

loaded_model = joblib.load(model_path); reproduced = loaded_model.predict_proba(full_x)[:, 1]
reproduction_error = float(np.max(np.abs(reproduced - training_probability)))
if reproduction_error > 1e-12: raise ValueError(f"Model reload reproduction failed: {reproduction_error}")

artifact_hashes = {
    "model_joblib_sha256": sha256_file(model_path), "model_json_sha256": sha256_file(model_json_path),
    "downstream_preprocessor_sha256": sha256_file(preprocessor_path),
    "discovery_preprocessor_sha256": sha256_file(discovery_preprocessor_path),
    "feature_manifest_sha256": sha256_file(manifest_path),
    "private_discovery_selection_sha256": sha256_file(selection_path),
    "private_training_scores_sha256": sha256_file(score_path),
}
lock = {
    "step": "5.12N6", "status": "FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED",
    "pipeline": "official PULSCAR discovery -> mean-imputed XGBoost",
    "training_patients": EXPECTED_PATIENTS, "documented_positive": EXPECTED_POSITIVES,
    "unlabeled": EXPECTED_UNLABELED, "estimated_alpha": float(discovery_audit["estimated_alpha"]),
    "selected_probable_positive": int(full_selected.sum()),
    "final_training_positive": int(final_target.sum()), "final_training_provisional_negative": int((final_target == 0).sum()),
    "best_optuna_trial": int(study.best_trial.number), "best_mean_five_fold_AP": float(study.best_value),
    "locked_xgboost_parameters": best_parameters, "raw_feature_count": len(FEATURES),
    "transformed_feature_count": EXPECTED_TRANSFORMED_FEATURES,
    "excluded_high_missingness_features": EXCLUDED_FEATURES, "uacr_used_anywhere_in_modeling": False,
    "numeric_missing_policy": "fold/full-fit arithmetic mean; no missingness indicators; no NaN reaches XGBoost",
    "model_reload_max_absolute_difference": reproduction_error, "artifact_hashes": artifact_hashes,
    "threshold_selected": False, "priority_categories_selected": False,
    "validation_accessed_in_this_step": False, "test_accessed_in_this_step": False,
    "legacy_model_overwritten": False, "clinician_review_sample_replaced": False,
    "next_step": "Outcome-blind scoring of the locked Validation cohort; do not load Validation outcomes.",
}
atomic_json(OUTPUT_DIR / "final_model_lock.json", lock)
atomic_json(OUTPUT_DIR / "completion.json", {**context, **lock, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(OUTPUT_DIR)})

print("=" * 112)
print("STEP 5.12N6 — FULL-TRAINING PULSCAR + MEAN-IMPUTED XGBOOST FINAL REFIT AND LOCK COMPLETE")
print("=" * 112)
print("\nFULL-TRAINING OFFICIAL PULSCAR DISCOVERY")
print(json.dumps(discovery_audit, indent=2, ensure_ascii=False))
print("\nLOCKED XGBOOST HYPERPARAMETERS")
print(json.dumps(best_parameters, indent=2, ensure_ascii=False))
print("\nTRANSFORMED FEATURE MANIFEST")
display(manifest)
print("\nFINAL MODEL LOCK")
print(json.dumps(lock, indent=2, ensure_ascii=False))
print(f"\nSaved run: {OUTPUT_DIR}")
print("Validation accessed in this step: No")
print("Test accessed in this step: No")
print("uACR used in discovery, imputation, tuning, or final model: No")
print("Missingness indicators used: No")
print("Native-NaN routing used: No")
