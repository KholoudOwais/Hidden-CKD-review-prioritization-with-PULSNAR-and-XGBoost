"""STEP 5.12N0 — Training-only availability audit and >70% policy lock.

Colab-ready. This step reads only the previously frozen Training snapshot.
It does not fit a model and does not access Validation or Test.
"""

from google.colab import drive
drive.mount("/content/drive")

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd
from IPython.display import display


EXPECTED_PATIENTS = 69_701
EXPECTED_POSITIVES = 5_042
EXPECTED_UNLABELED = 64_659
HIGH_MISSINGNESS_THRESHOLD_PERCENT = 70.0

ROOT = Path(
    "/content/drive/MyDrive/CKD_Project/outputs/"
    "section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
D0_ROOT = ROOT / "05_6D0_pulsnar_dual_engine_protocol_amendment"
OUTPUT_ROOT = ROOT / "05_12N0_training_only_feature_availability_and_policy_lock"


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Restore the exact frozen Training snapshot and Long25 feature registry.
valid_d0 = []
for path in D0_ROOT.rglob("*.json"):
    payload = load_json(path)
    if payload is None:
        continue
    snapshot_text = payload.get("source_files", {}).get("frozen_training_snapshot", "")
    expected_hash = payload.get("source_files", {}).get(
        "frozen_training_snapshot_sha256", ""
    )
    features = payload.get("feature_design", {}).get("features")
    cohort = payload.get("cohort", {})
    if snapshot_text and expected_hash and isinstance(features, list) and cohort:
        valid_d0.append((path, payload))

if not valid_d0:
    raise FileNotFoundError("No valid Step 5.6D0 Training protocol lock was found.")

D0_FILE, d0 = max(valid_d0, key=lambda item: item[0].stat().st_mtime)
SNAPSHOT_FILE = Path(d0["source_files"]["frozen_training_snapshot"])
EXPECTED_SNAPSHOT_HASH = d0["source_files"]["frozen_training_snapshot_sha256"]
FEATURES = list(d0["feature_design"]["features"])
ID_COL = d0["cohort"]["patient_id_column"]
LABEL_COL = d0["cohort"]["observed_label_column"]
FOLD_COL = d0["cohort"]["locked_fold_column"]

if len(FEATURES) != 25:
    raise ValueError(f"Expected Long25, found {len(FEATURES)} features.")
if not SNAPSHOT_FILE.is_file():
    raise FileNotFoundError(SNAPSHOT_FILE)
if sha256_file(SNAPSHOT_FILE) != EXPECTED_SNAPSHOT_HASH:
    raise ValueError("Frozen Training snapshot hash changed.")

required_columns = [ID_COL, LABEL_COL, FOLD_COL, *FEATURES]
training = pd.read_parquet(SNAPSHOT_FILE, columns=required_columns)

if len(training) != EXPECTED_PATIENTS:
    raise ValueError(f"Training row count changed: {len(training):,}")
if training[ID_COL].isna().any() or training[ID_COL].duplicated().any():
    raise ValueError("Training patient identifiers are missing or duplicated.")

labels = pd.to_numeric(training[LABEL_COL], errors="raise").astype(int)
if set(labels.unique()) - {0, 1}:
    raise ValueError("Observed Training label is not binary.")
if int(labels.sum()) != EXPECTED_POSITIVES:
    raise ValueError(f"Training positives changed: {int(labels.sum()):,}")
if int((labels == 0).sum()) != EXPECTED_UNLABELED:
    raise ValueError(f"Training unlabeled count changed: {int((labels == 0).sum()):,}")


audit_rows = []
for feature in FEATURES:
    observed_n = int(training[feature].notna().sum())
    missing_n = int(training[feature].isna().sum())
    observed_percent = 100.0 * observed_n / len(training)
    missing_percent = 100.0 * missing_n / len(training)
    audit_rows.append(
        {
            "Feature": feature,
            "Observed_n": observed_n,
            "Observed_percent": observed_percent,
            "Missing_n": missing_n,
            "Missing_percent": missing_percent,
            "Exceeds_70pct_missing": bool(
                missing_percent > HIGH_MISSINGNESS_THRESHOLD_PERCENT
            ),
        }
    )

availability = pd.DataFrame(audit_rows).sort_values(
    ["Observed_percent", "Feature"], ascending=[True, True]
).reset_index(drop=True)

excluded_features = availability.loc[
    availability["Exceeds_70pct_missing"], "Feature"
].tolist()
retained_features = [feature for feature in FEATURES if feature not in excluded_features]

expected_excluded = {
    "UACR_LOG1P_FINAL",
    "EGFR_PREVIOUS_90D",
    "EGFR_CHANGE_LAST_MINUS_PREVIOUS_90D",
    "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS",
    "EGFR_SLOPE_PER_30_DAYS_90D",
}
if set(excluded_features) != expected_excluded:
    raise ValueError(
        "The >70% rule did not produce the expected five-feature exclusion set: "
        f"{excluded_features}"
    )
if len(retained_features) != 20:
    raise ValueError(f"Expected 20 retained features, found {len(retained_features)}.")


run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_dir = OUTPUT_ROOT / run_id
output_dir.mkdir(parents=True, exist_ok=False)

availability_file = output_dir / "training_feature_availability_and_policy.csv"
availability.to_csv(availability_file, index=False)

policy = {
    "step": "5.12N0",
    "status": "TRAINING_ONLY_HIGH_MISSINGNESS_POLICY_LOCKED",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_protocol": str(D0_FILE),
    "frozen_training_snapshot": str(SNAPSHOT_FILE),
    "frozen_training_snapshot_sha256": EXPECTED_SNAPSHOT_HASH,
    "training_patients": int(len(training)),
    "documented_ckd": int(labels.sum()),
    "unlabeled": int((labels == 0).sum()),
    "original_feature_count": int(len(FEATURES)),
    "high_missingness_threshold_percent": HIGH_MISSINGNESS_THRESHOLD_PERCENT,
    "threshold_rule": "Exclude when Training missingness is strictly greater than 70%.",
    "excluded_features": excluded_features,
    "retained_feature_count": int(len(retained_features)),
    "retained_features": retained_features,
    "primary_model_policy": {
        "uacr": "excluded",
        "sparse_90d_egfr_features": "excluded",
        "native_nan_permitted": False,
        "missingness_indicators_permitted": False,
        "remaining_missing_values": (
            "must be imputed inside each Training fold before PULSNAR and model fitting"
        ),
    },
    "planned_training_only_comparators": [
        "fold-fitted mean imputation",
        "proper multiple MICE with pooled predictions",
    ],
    "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
    "model_fitted_or_changed": False,
}
atomic_json(output_dir / "training_only_feature_policy_lock.json", policy)


print("=" * 104)
print("STEP 5.12N0 — TRAINING-ONLY FEATURE AVAILABILITY AND >70% POLICY LOCK COMPLETE")
print("=" * 104)
print("\nTRAINING FEATURE AVAILABILITY")
display(
    availability.style.format(
        {"Observed_percent": "{:.2f}", "Missing_percent": "{:.2f}"}
    )
)
print("\nEXCLUDED FROM THE NEW PRIMARY MODEL")
display(availability.loc[availability["Exceeds_70pct_missing"]])
print("\nRETAINED FOR IMPUTATION COMPARISON")
display(availability.loc[~availability["Exceeds_70pct_missing"]])
print("\nSaved run:", output_dir)
print("Training accessed: Yes — frozen Training snapshot only")
print("Validation accessed: No")
print("Test accessed: No")
print("Model fitted or changed: No")
print("Next: Training-only mean versus proper multiple-MICE pipeline screen.")
