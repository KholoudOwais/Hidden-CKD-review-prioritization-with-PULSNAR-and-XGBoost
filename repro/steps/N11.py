"""STEP 5.12N11 — Outcome-blind locked Internal Test scoring.

Applies the immutable N6 official-PULSCAR/mean-imputed-XGBoost pipeline to the
authoritative predictor-only Internal Test table. The step freezes continuous
model outputs, percentages, percentile ranks, and original feature availability.

It does not load Test outcomes, calculate Test metrics, create review-priority
categories, apply an alpha-matched unlabeled review set, or change the model.
"""

from google.colab import drive

drive.mount("/content/drive")

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd


EXPECTED_TEST_PATIENTS = 14_927
ID_COL = "patient_id"
SCORE_COL = "locked_pulscar_mean_xgb_internal_test_score"
PERCENT_COL = "model_output_percentage"
PERCENTILE_COL = "within_cohort_percentile"

ROOT = Path(
    "/content/drive/MyDrive/CKD_Project/outputs/"
    "section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
N6_ROOT = ROOT / "05_12N6_full_training_pulscar_mean_xgb_final_refit_and_lock"
N10_ROOT = ROOT / (
    "05_12N10_reference_grounded_reporting_and_internal_test_protocol_freeze"
)
OUTPUT_ROOT = ROOT / (
    "05_12N11_outcome_blind_locked_pulscar_mean_xgb_internal_test_scoring"
)
TEST_PREDICTORS = ROOT / (
    "05_8B1_authoritative_outcome_blind_test_predictors/"
    "fd404fbc084e9ff467c1f767/PRIVATE_INTERNAL_DO_NOT_SHARE/"
    "renal_eligible_test_predictors_no_outcome.parquet"
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
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def latest_completed(root, expected_status):
    candidates = []
    for path in root.rglob("completion.json"):
        payload = load_json(path)
        if payload and payload.get("status") == expected_status:
            candidates.append((path, payload))
    if not candidates:
        raise FileNotFoundError(
            f"No completed artifact with status {expected_status!r} under {root}."
        )
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def patient_key(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        nonmissing = numeric.dropna()
        if (np.abs(nonmissing - np.round(nonmissing)) < 1e-9).all():
            return numeric.round().astype("Int64").astype("string")
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


N6_COMPLETION, n6 = latest_completed(
    N6_ROOT, "FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED"
)
N10_COMPLETION, n10 = latest_completed(
    N10_ROOT,
    "REFERENCE_GROUNDED_REPORTING_AND_INTERNAL_TEST_PROTOCOL_FREEZE_COMPLETE",
)
N6_RUN = N6_COMPLETION.parent

MODEL = N6_RUN / "locked_official_pulscar_mean_xgb_model.joblib"
MODEL_JSON = N6_RUN / "locked_official_pulscar_mean_xgb_model.json"
PREPROCESSOR = N6_RUN / "locked_mean_imputation_preprocessor.joblib"
MANIFEST = N6_RUN / "locked_transformed_feature_manifest.csv"

for required in [
    N6_COMPLETION,
    N10_COMPLETION,
    MODEL,
    MODEL_JSON,
    PREPROCESSOR,
    MANIFEST,
    TEST_PREDICTORS,
]:
    if not required.is_file():
        raise FileNotFoundError(required)

if n6.get("pipeline") != "official PULSCAR discovery -> mean-imputed XGBoost":
    raise ValueError("The N6 pipeline identity changed.")
if n6.get("uacr_used_anywhere_in_modeling") is not False:
    raise ValueError("N6 reports uACR use.")
if n6.get("raw_feature_count") != 20 or n6.get("transformed_feature_count") != 21:
    raise ValueError("The locked N6 feature dimensions changed.")
if n6.get("threshold_selected") or n6.get("priority_categories_selected"):
    raise ValueError("N6 unexpectedly selected a threshold or categories.")
if n10.get("model_changed") or n10.get("preprocessing_changed"):
    raise ValueError("N10 reports a forbidden pipeline change.")
if n10.get("numeric_threshold_selected"):
    raise ValueError("N10 unexpectedly selected a numeric threshold.")
if n10.get("clinical_priority_categories_selected"):
    raise ValueError("N10 unexpectedly selected clinical priority categories.")
if n10.get("test_accessed"):
    raise ValueError("N10 reports Internal Test access.")

artifact_hashes = n6["artifact_hashes"]
for path, hash_key in [
    (MODEL, "model_joblib_sha256"),
    (MODEL_JSON, "model_json_sha256"),
    (PREPROCESSOR, "downstream_preprocessor_sha256"),
    (MANIFEST, "feature_manifest_sha256"),
]:
    actual = sha256_file(path)
    expected = artifact_hashes[hash_key]
    if actual != expected:
        raise ValueError(f"Locked artifact changed: {path}")

model = joblib.load(MODEL)
preprocessor = joblib.load(PREPROCESSOR)
manifest = pd.read_csv(MANIFEST)
test = pd.read_parquet(TEST_PREDICTORS)

if len(test) != EXPECTED_TEST_PATIENTS:
    raise ValueError(
        f"Expected {EXPECTED_TEST_PATIENTS:,} Internal Test rows; found {len(test):,}."
    )
if ID_COL not in test.columns:
    raise KeyError(f"Internal Test table lacks {ID_COL!r}.")
test["_PATIENT_KEY"] = patient_key(test[ID_COL])
if test["_PATIENT_KEY"].isna().any() or test["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Internal Test patient identifiers are invalid.")

outcome_tokens = (
    "ckd_class",
    "documented_ckd",
    "observed_label",
    "true_label",
    "outcome",
    "persistent_evidence",
    "final_label",
)
for column in test.columns:
    if any(token in str(column).lower() for token in outcome_tokens):
        raise ValueError(f"Outcome-like Internal Test column present: {column}")

numeric_features = list(preprocessor["numeric_features"])
categorical_feature = preprocessor["categorical_feature"]
features = numeric_features + [categorical_feature]
if len(features) != 20:
    raise ValueError(f"Expected 20 locked raw features; found {len(features)}.")
if any("uacr" in feature.lower() for feature in features):
    raise ValueError("uACR reached Internal Test scoring.")
missing = sorted(set([ID_COL, *features]) - set(test.columns))
if missing:
    raise KeyError(f"Internal Test predictors lack: {missing}")

raw_numeric = (
    test[numeric_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)
availability = pd.DataFrame({ID_COL: test[ID_COL].to_numpy()})
for feature in features:
    availability[feature] = test[feature].notna().to_numpy(bool)

numeric_matrix = preprocessor["imputer"].transform(raw_numeric).astype("float32")
gender_matrix = preprocessor["gender_encoder"].transform(
    test[[categorical_feature]].astype("string")
).astype("float32")
matrix = np.column_stack([numeric_matrix, gender_matrix]).astype("float32")

transformed_names = list(preprocessor["transformed_features"])
manifest_names = (
    manifest.sort_values("transformed_position")["transformed_feature"]
    .astype(str)
    .tolist()
)
if matrix.shape != (EXPECTED_TEST_PATIENTS, 21):
    raise ValueError(f"Unexpected transformed Internal Test shape: {matrix.shape}")
if not np.isfinite(matrix).all():
    raise ValueError("Nonfinite value reached locked XGBoost.")
if transformed_names != manifest_names:
    raise ValueError("Preprocessor and manifest transformed names differ.")

scores = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
    raise ValueError("Locked Internal Test scores are invalid.")

# Outcome-independent deterministic rank. A high percentile means a high score
# relative to this cohort. It is not a clinical probability or KDIGO category.
rank_frame = pd.DataFrame(
    {
        ID_COL: test[ID_COL].to_numpy(),
        "_PATIENT_KEY": test["_PATIENT_KEY"].to_numpy(),
        SCORE_COL: scores,
    }
)
rank_frame["_TIE_HASH"] = rank_frame["_PATIENT_KEY"].map(
    lambda value: hashlib.sha256(
        f"N11|internal-test-rank|{value}".encode("utf-8")
    ).hexdigest()
)
ordered = rank_frame.sort_values(
    [SCORE_COL, "_TIE_HASH"], ascending=[False, True], kind="stable"
).reset_index(drop=True)
ordered["_RANK"] = np.arange(1, len(ordered) + 1, dtype=int)
ordered[PERCENTILE_COL] = 100.0 * (
    len(ordered) - ordered["_RANK"] + 1
) / len(ordered)
percentile_map = ordered.set_index("_PATIENT_KEY")[PERCENTILE_COL]
rank_frame[PERCENTILE_COL] = rank_frame["_PATIENT_KEY"].map(percentile_map)
rank_frame[PERCENT_COL] = 100.0 * rank_frame[SCORE_COL]

frozen_scores = rank_frame[
    [ID_COL, SCORE_COL, PERCENT_COL, PERCENTILE_COL]
].copy()
if frozen_scores[PERCENTILE_COL].isna().any():
    raise ValueError("Internal Test percentile calculation failed.")

availability_summary = pd.DataFrame(
    {
        "feature": features,
        "observed_patients": [int(availability[x].sum()) for x in features],
        "observed_percent": [100.0 * float(availability[x].mean()) for x in features],
    }
)
availability_summary["missing_patients"] = (
    EXPECTED_TEST_PATIENTS - availability_summary["observed_patients"]
)
availability_summary["missing_percent"] = (
    100.0 - availability_summary["observed_percent"]
)

distribution = {
    "patients_scored": EXPECTED_TEST_PATIENTS,
    "minimum": float(scores.min()),
    "p01": float(np.quantile(scores, 0.01)),
    "p05": float(np.quantile(scores, 0.05)),
    "median": float(np.median(scores)),
    "mean": float(scores.mean()),
    "p95": float(np.quantile(scores, 0.95)),
    "p99": float(np.quantile(scores, 0.99)),
    "maximum": float(scores.max()),
}

context = {
    "step": "5.12N11",
    "status": "OUTCOME_BLIND_LOCKED_INTERNAL_TEST_SCORING",
    "n6_completion_sha256": sha256_file(N6_COMPLETION),
    "n10_completion_sha256": sha256_file(N10_COMPLETION),
    "model_sha256": sha256_file(MODEL),
    "preprocessor_sha256": sha256_file(PREPROCESSOR),
    "manifest_sha256": sha256_file(MANIFEST),
    "internal_test_predictors": str(TEST_PREDICTORS),
    "internal_test_predictors_sha256": sha256_file(TEST_PREDICTORS),
    "features": features,
    "numeric_missing_policy": (
        "Locked Training-mean imputation; no indicators; no NaN reaches XGBoost."
    ),
    "score_interpretation": (
        "Uncalibrated model-estimated positive-class output; not a clinical "
        "probability of CKD."
    ),
    "percentile_interpretation": (
        "Within-cohort relative rank only; not a probability or clinical category."
    ),
    "internal_test_previously_accessed_for_superseded_pipeline": True,
    "internal_test_outcomes_loaded": False,
    "internal_test_metrics_calculated": False,
    "alpha_matched_review_set_applied": False,
    "numeric_threshold_selected": False,
    "priority_categories_created": False,
    "model_or_preprocessing_changed": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
atomic_json(OUTPUT_DIR / "context.json", context)

SCORE_FILE = PRIVATE_DIR / "locked_internal_test_continuous_scores.parquet"
AVAILABILITY_FILE = PRIVATE_DIR / "internal_test_original_feature_availability.parquet"
AVAILABILITY_SUMMARY = OUTPUT_DIR / "internal_test_predictor_availability_summary.csv"

temporary = SCORE_FILE.with_suffix(".tmp.parquet")
frozen_scores.to_parquet(temporary, index=False)
os.replace(temporary, SCORE_FILE)
temporary = AVAILABILITY_FILE.with_suffix(".tmp.parquet")
availability.to_parquet(temporary, index=False)
os.replace(temporary, AVAILABILITY_FILE)
availability_summary.to_csv(AVAILABILITY_SUMMARY, index=False)

completion = {
    **context,
    "status": "OUTCOME_BLIND_LOCKED_INTERNAL_TEST_SCORING_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(OUTPUT_DIR),
    "score_file": str(SCORE_FILE),
    "score_file_sha256": sha256_file(SCORE_FILE),
    "availability_file": str(AVAILABILITY_FILE),
    "availability_file_sha256": sha256_file(AVAILABILITY_FILE),
    "availability_summary_file": str(AVAILABILITY_SUMMARY),
    "availability_summary_sha256": sha256_file(AVAILABILITY_SUMMARY),
    "score_distribution": distribution,
    "model_changed": False,
    "next_step": (
        "Open Internal Test outcomes once and evaluate the immutable continuous "
        "score vector under the frozen N10 protocol."
    ),
}
atomic_json(OUTPUT_DIR / "completion.json", completion)

print("=" * 116)
print("STEP 5.12N11 — OUTCOME-BLIND LOCKED INTERNAL TEST SCORING COMPLETE")
print("=" * 116)
print("\nSCORE DISTRIBUTION")
print(json.dumps(distribution, indent=2, ensure_ascii=False))
print(f"\nScore file: {SCORE_FILE}")
print(f"SHA256: {completion['score_file_sha256']}")
print(f"Observed-feature availability: {AVAILABILITY_FILE}")
print("Internal Test predictors accessed: Yes — predictor-only authoritative table")
print("Internal Test outcomes loaded: No")
print("Internal Test metrics calculated: No")
print("PULSCAR-alpha review set applied: No — labels remain unopened")
print("High/Moderate/Low categories created: No")
print("Model or preprocessing changed: No")
