"""STEP 5.12N6A — Supplementary Training-only imputation sensitivity.

Adds fold-fitted median imputation to the already completed, methodologically
matched N1 mean and five-draw MICE comparison. Uses the same 20 features,
locked folds, temporal weights, fixed XGBoost engine, and random seeds. This is
supplementary only and cannot replace the locked N6 model. Validation/Test are
not loaded. KNN is excluded a priori because full-cohort neighbour imputation
has quadratic distance cost and is not operationally scalable here.
"""

from google.colab import drive
drive.mount("/content/drive")

from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, tempfile, time
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

EXPECTED_PATIENTS = 69_701
EXPECTED_POSITIVES = 5_042
EXPECTED_UNLABELED = 64_659
EXPECTED_FEATURES = 20
EXPECTED_FOLDS = 5
BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 71_000_042
WEIGHT_COL = "TEMPORAL_BALANCING_WEIGHT"

LOCKED_XGB_PARAMETERS = {
    "n_estimators": 500, "learning_rate": 0.04570147939250174,
    "reg_lambda": 10.204527160227807, "colsample_bytree": 0.7400473792766912,
    "max_depth": 5, "min_child_weight": 7.323551441473847,
    "subsample": 0.8377920943702721,
}

ROOT = Path("/content/drive/MyDrive/CKD_Project/outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling")
N1_ROOT = ROOT / "05_12N1_training_only_mean_vs_multiple_mice_screen"
N6_ROOT = ROOT / "05_12N6_full_training_pulscar_mean_xgb_final_refit_and_lock"
OUTPUT_ROOT = ROOT / "05_12N6A_training_only_imputation_sensitivity"

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

def make_model(seed):
    return XGBClassifier(**LOCKED_XGB_PARAMETERS, objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", n_jobs=-1,
        random_state=int(seed), verbosity=0)

def metric_row(name, target, probability):
    clipped = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    return {"Candidate": name,
        "Average_precision": float(average_precision_score(target, clipped)),
        "ROC_AUC": float(roc_auc_score(target, clipped)),
        "Brier_score": float(brier_score_loss(target, clipped)),
        "Log_loss": float(log_loss(target, clipped, labels=[0, 1]))}

# Require the completed locked N6 model; this analysis cannot modify it.
n6_candidates = []
if N6_ROOT.exists():
    for path in N6_ROOT.rglob("completion.json"):
        payload = load_json(path)
        if payload and payload.get("status") == "FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED":
            n6_candidates.append((path, payload))
if not n6_candidates: raise FileNotFoundError("Run N6 before N6A.")
N6_FILE, n6 = max(n6_candidates, key=lambda item: item[0].stat().st_mtime)
if n6.get("validation_accessed_in_this_step") or n6.get("test_accessed_in_this_step"):
    raise ValueError("N6 reports held-out access.")
if n6.get("uacr_used_anywhere_in_modeling") or n6.get("raw_feature_count") != EXPECTED_FEATURES:
    raise ValueError("N6 feature policy is incompatible.")

# Restore the corrected N1 run (temporal weights applied).
n1_candidates = []
if N1_ROOT.exists():
    for path in N1_ROOT.rglob("completion.json"):
        payload = load_json(path)
        oof_path = path.parent / "PRIVATE_INTERNAL_DO_NOT_SHARE" / "training_oof_imputation_scores.parquet"
        if (payload and payload.get("status") == "TRAINING_ONLY_MEAN_VS_MULTIPLE_MICE_SCREEN_COMPLETE"
                and "temporal balancing weights" in str(payload.get("training_weight_policy", "")).lower()
                and oof_path.is_file()):
            n1_candidates.append((path, payload, oof_path))
if not n1_candidates: raise FileNotFoundError("Corrected N1 run was not found.")
N1_FILE, n1, N1_OOF = max(n1_candidates, key=lambda item: item[0].stat().st_mtime)
N0_FILE = Path(n1["source_policy_lock"]); n0 = load_json(N0_FILE)
if n0 is None or n0.get("status") != "TRAINING_ONLY_HIGH_MISSINGNESS_POLICY_LOCKED": raise ValueError("Invalid N0 lock.")
SNAPSHOT_FILE = Path(n0["frozen_training_snapshot"])
if not SNAPSHOT_FILE.is_file() or sha256_file(SNAPSHOT_FILE) != n0["frozen_training_snapshot_sha256"]:
    raise ValueError("Training snapshot changed.")
FEATURES = list(n0["retained_features"]); EXCLUDED = list(n0["excluded_features"])
if FEATURES != list(n6["features"]) or len(FEATURES) != EXPECTED_FEATURES: raise ValueError("Feature panel changed.")
d0 = load_json(n0["source_protocol"])
ID_COL = d0["cohort"]["patient_id_column"]; LABEL_COL = d0["cohort"]["observed_label_column"]
FOLD_COL = d0["cohort"]["locked_fold_column"]
training = pd.read_parquet(SNAPSHOT_FILE, columns=[ID_COL, LABEL_COL, FOLD_COL, WEIGHT_COL, *FEATURES])
if len(training) != EXPECTED_PATIENTS or not training[ID_COL].is_unique: raise ValueError("Unexpected Training cohort.")
target = pd.to_numeric(training[LABEL_COL], errors="raise").astype(int).to_numpy()
weights = pd.to_numeric(training[WEIGHT_COL], errors="raise").to_numpy(float); weights /= weights.mean()
if int(target.sum()) != EXPECTED_POSITIVES or int((target == 0).sum()) != EXPECTED_UNLABELED: raise ValueError("Class counts changed.")
fold_values = sorted(training[FOLD_COL].dropna().unique().tolist())
if len(fold_values) != EXPECTED_FOLDS: raise ValueError("Fold count changed.")
NUMERIC = [x for x in FEATURES if x != "GENDER"]
numeric = training[NUMERIC].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
gender = training[["GENDER"]].astype("string")
if gender["GENDER"].isna().any(): raise ValueError("GENDER contains missing values.")

prior = pd.read_parquet(N1_OOF)
if not np.array_equal(prior[ID_COL].to_numpy(), training[ID_COL].to_numpy()): raise ValueError("N1 OOF order changed.")
mean_oof = pd.to_numeric(prior["mean_imputation_oof_score"], errors="raise").to_numpy(float)
mice_oof = pd.to_numeric(prior["multiple_mice_5_oof_score"], errors="raise").to_numpy(float)
if not np.isfinite(mean_oof).all() or not np.isfinite(mice_oof).all(): raise ValueError("Invalid N1 OOF scores.")

context = {"step": "5.12N6A", "status": "SUPPLEMENTARY_TRAINING_ONLY_IMPUTATION_SENSITIVITY",
    "n6_completion_sha256": sha256_file(N6_FILE), "n1_completion_sha256": sha256_file(N1_FILE),
    "n1_oof_sha256": sha256_file(N1_OOF), "training_snapshot_sha256": sha256_file(SNAPSHOT_FILE),
    "features": FEATURES, "excluded_features": EXCLUDED,
    "candidates": ["mean_imputation", "median_imputation", "multiple_mice_5"],
    "excluded_candidate": {"knn_imputation": "excluded before results because full-cohort neighbour-distance computation is quadratic and not operationally scalable for 69,701 patients"},
    "engine": "fixed-parameter XGBoost", "engine_parameters": LOCKED_XGB_PARAMETERS,
    "role": "supplementary sensitivity only; cannot alter the locked N6 model",
    "validation_accessed_in_this_step": False, "test_accessed_in_this_step": False}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id; PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
CACHE_DIR = PRIVATE_DIR / "median_fold_cache"
for directory in [OUTPUT_DIR, PRIVATE_DIR, CACHE_DIR]: directory.mkdir(parents=True, exist_ok=True)
CONTEXT_FILE = OUTPUT_DIR / "context.json"
if CONTEXT_FILE.is_file() and load_json(CONTEXT_FILE) != json_safe(context): raise ValueError("Incompatible N6A context.")
if not CONTEXT_FILE.is_file(): atomic_json(CONTEXT_FILE, context)

median_oof = np.full(EXPECTED_PATIENTS, np.nan)
fold_rows = []
for fold_number, fold_value in enumerate(fold_values, start=1):
    eval_mask = training[FOLD_COL].eq(fold_value).to_numpy(); fit_mask = ~eval_mask
    cache = CACHE_DIR / f"median_fold_{fold_number}.npz"
    if cache.is_file():
        prediction = np.load(cache)["prediction"]
        if len(prediction) != int(eval_mask.sum()): raise ValueError("Invalid median cache.")
        print(f"Reused median_imputation, fold {fold_number}.")
    else:
        started = time.perf_counter()
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        fit_gender = encoder.fit_transform(gender.loc[fit_mask]); eval_gender = encoder.transform(gender.loc[eval_mask])
        imputer = SimpleImputer(strategy="median", add_indicator=False)
        fit_num = imputer.fit_transform(numeric.loc[fit_mask]).astype(np.float32)
        eval_num = imputer.transform(numeric.loc[eval_mask]).astype(np.float32)
        fit_x = np.column_stack([fit_num, fit_gender]).astype(np.float32)
        eval_x = np.column_stack([eval_num, eval_gender]).astype(np.float32)
        if not np.isfinite(fit_x).all() or not np.isfinite(eval_x).all(): raise ValueError("Median passed nonfinite values.")
        model = make_model(RANDOM_SEED + fold_number)
        model.fit(fit_x, target[fit_mask], sample_weight=weights[fit_mask])
        prediction = model.predict_proba(eval_x)[:, 1]
        np.savez_compressed(cache, prediction=prediction)
        print(f"Completed median_imputation, fold {fold_number}, {(time.perf_counter()-started)/60:.2f} minutes.")
    median_oof[eval_mask] = prediction
    fold_rows.append({"Fold": fold_number, "Evaluation_patients": int(eval_mask.sum()),
        "All_scores_finite": bool(np.isfinite(prediction).all())})
if not np.isfinite(median_oof).all(): raise ValueError("Incomplete median OOF scores.")

predictions = {"mean_imputation": mean_oof, "median_imputation": median_oof, "multiple_mice_5": mice_oof}
summary = pd.DataFrame([metric_row(name, target, pred) for name, pred in predictions.items()])
mean_ap = float(summary.loc[summary["Candidate"].eq("mean_imputation"), "Average_precision"].iloc[0])
summary["Delta_AP_vs_mean"] = summary["Average_precision"] - mean_ap

positive = np.flatnonzero(target == 1); unlabeled = np.flatnonzero(target == 0)
rng = np.random.default_rng(RANDOM_SEED + 60_000)
deltas = {name: np.empty(BOOTSTRAP_REPLICATES) for name in ["median_imputation", "multiple_mice_5"]}
for replicate in range(BOOTSTRAP_REPLICATES):
    sampled = np.concatenate([rng.choice(positive, len(positive), replace=True), rng.choice(unlabeled, len(unlabeled), replace=True)])
    reference_ap = average_precision_score(target[sampled], mean_oof[sampled])
    for name in deltas: deltas[name][replicate] = average_precision_score(target[sampled], predictions[name][sampled]) - reference_ap
    if (replicate + 1) % 100 == 0: print(f"Bootstrap {replicate+1}/{BOOTSTRAP_REPLICATES}")
bootstrap = pd.DataFrame([{"Candidate": name, "Reference": "mean_imputation",
    "Delta_AP": float(summary.loc[summary["Candidate"].eq(name), "Delta_AP_vs_mean"].iloc[0]),
    "Bootstrap_95CI_lower": float(np.quantile(values, .025)), "Bootstrap_95CI_upper": float(np.quantile(values, .975)),
    "Replicates": BOOTSTRAP_REPLICATES} for name, values in deltas.items()])

pd.DataFrame({ID_COL: training[ID_COL], "median_imputation_oof_score": median_oof}).to_parquet(PRIVATE_DIR / "median_imputation_oof.parquet", index=False)
summary.to_csv(OUTPUT_DIR / "supplementary_imputation_summary.csv", index=False)
bootstrap.to_csv(OUTPUT_DIR / "paired_bootstrap_vs_mean.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "median_fold_audit.csv", index=False)
decision = {"step": "5.12N6A", "status": "SUPPLEMENTARY_TRAINING_ONLY_IMPUTATION_SENSITIVITY_COMPLETE",
    "primary_locked_policy_remains": "mean_imputation", "locked_n6_model_changed": False,
    "interpretation": "Algorithm-development sensitivity analysis on Training OOF predictions only.",
    "next_step": "Run the prespecified Training-only baseline-model benchmark before any new Validation scoring.",
    "validation_accessed_in_this_step": False, "test_accessed_in_this_step": False}
atomic_json(OUTPUT_DIR / "imputation_sensitivity_decision.json", decision)
atomic_json(OUTPUT_DIR / "completion.json", {**context, **decision, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(OUTPUT_DIR)})

print("=" * 112)
print("STEP 5.12N6A — SUPPLEMENTARY TRAINING-ONLY IMPUTATION SENSITIVITY COMPLETE")
print("=" * 112)
print("\nIMPUTATION SUMMARY"); display(summary)
print("\nPAIRED BOOTSTRAP VERSUS MEAN"); display(bootstrap)
print("\nMEDIAN FOLD AUDIT"); display(pd.DataFrame(fold_rows))
print("\nDECISION"); print(json.dumps(decision, indent=2, ensure_ascii=False))
print(f"\nSaved run: {OUTPUT_DIR}")
print("Validation accessed: No"); print("Test accessed: No")
print("Locked N6 model changed: No")
