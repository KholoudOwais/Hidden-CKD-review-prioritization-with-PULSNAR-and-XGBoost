"""STEP 5.12N9 — Validation Low-Priority Documented-CKD Safety Audit.

Colab-ready post-model audit of the 16 documented-CKD Validation patients who
were assigned Low review priority by the immutable N6 model and N8 rank policy.

The audit deliberately separates:
  1. variables actually available to the locked model;
  2. excluded renal evidence used only after prediction for safety review; and
  3. questions that require clinical-record adjudication.

It does not train, tune, recalibrate, rescore, change priority categories,
replace the clinician-review sample, or access the internal Test cohort.
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
import pyarrow.parquet as pq
from IPython.display import display


EXPECTED_CASES = 16
EXPECTED_VALIDATION_PATIENTS = 14_923
ID_COL = "patient_id"
SCORE_COL = "locked_pulscar_mean_xgb_validation_score"
LABEL_COL = "OBSERVED_CKD_LABEL"

PROJECT_ROOT = Path("/content/drive/MyDrive/CKD_Project")
ROOT = PROJECT_ROOT / (
    "outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
N8_ROOT = ROOT / "05_12N8_validation_characterization_and_workload_lock"
N7_ROOT = ROOT / (
    "05_12N7_outcome_blind_locked_pulscar_mean_xgb_validation_scoring"
)
CHECKPOINT = PROJECT_ROOT / "checkpoints/step_3_15_6_20260903_230014"
PATIENT_FILE = CHECKPOINT / "patient_level_data.parquet"
ENCOUNTER_FILE = CHECKPOINT / "df.parquet"
OUTPUT_ROOT = ROOT / (
    "05_12N9_validation_low_priority_documented_ckd_safety_audit"
)

MODEL_FEATURES = [
    "AGE_AT_INDEX",
    "BMI_MODEL",
    "SBP",
    "DBP",
    "PULSE_FINAL",
    "DERIVED_EGFR_CKD_EPI_2021",
    "BUN_FINAL_MMOL_L",
    "POTASSIUM_FINAL_MMOL_L",
    "SODIUM_FINAL_MMOL_L",
    "HEMOGLOBIN_MODEL_G_L",
    "MCV_MODEL_FL",
    "RDW_CV_MODEL_PERCENT",
    "WBC_MODEL_10E9_L",
    "PLATELETS_MODEL_10E9_L",
    "MPV_MODEL_FL",
    "LYMPHOCYTES_ABSOLUTE_MODEL_10E9_L",
    "EOSINOPHILS_ABSOLUTE_MODEL_10E9_L",
    "EGFR_MIN_24M",
    "EGFR_MEAN_24M",
    "GENDER",
]
EXCLUDED_RENAL_FEATURES = [
    "UACR_LOG1P_FINAL",
    "EGFR_PREVIOUS_90D",
    "EGFR_CHANGE_LAST_MINUS_PREVIOUS_90D",
    "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS",
    "EGFR_SLOPE_PER_30_DAYS_90D",
]


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


def patient_key(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        nonmissing = numeric.dropna()
        if (np.abs(nonmissing - np.round(nonmissing)) < 1e-9).all():
            return numeric.round().astype("Int64").astype("string")
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


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


def read_selected_rows(path, columns, raw_ids):
    """Read only the 16 selected Validation patient records from a checkpoint."""
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[(ID_COL, "in", list(raw_ids))],
    )
    requested = set(patient_key(pd.Series(list(raw_ids))).dropna())
    returned = set(patient_key(frame[ID_COL]).dropna())
    unexpected = returned - requested
    if unexpected:
        raise ValueError(f"Filtered checkpoint read returned unexpected IDs: {unexpected}")
    return frame


def egfr_band(value):
    if pd.isna(value):
        return "Not available"
    value = float(value)
    if value < 15:
        return "G5 (<15)"
    if value < 30:
        return "G4 (15-29)"
    if value < 45:
        return "G3b (30-44)"
    if value < 60:
        return "G3a (45-59)"
    if value < 90:
        return "G2 (60-89)"
    return "G1 (>=90)"


def uacr_band(value):
    if pd.isna(value):
        return "Not available"
    value = float(value)
    if value < 3:
        return "A1 (<3 mg/mmol)"
    if value <= 30:
        return "A2 (3-30 mg/mmol)"
    return "A3 (>30 mg/mmol)"


def renal_pattern(row):
    persistent_egfr = bool(row["PERSISTENT_LOW_EGFR_90D"])
    repeated_uacr = bool(row["REPEATED_HIGH_UACR_90D"])
    if persistent_egfr and repeated_uacr:
        return "Repeated low eGFR and repeated A2/A3 uACR evidence"
    if persistent_egfr:
        return "Repeated eGFR <60 at least 90 days apart"
    if repeated_uacr:
        return "Repeated uACR >=3 mg/mmol at least 90 days apart"
    if bool(row["ANY_EGFR_BELOW_60_IN_MODEL_WINDOW"]):
        return "Low eGFR observed, but chronicity not confirmed by the available 90-day pair"
    if pd.notna(row["CURRENT_UACR_MG_MMOL"]) and row["CURRENT_UACR_MG_MMOL"] >= 3:
        return "A2/A3 uACR observed once, but persistence not confirmed"
    return "No computable KDIGO-aligned chronic laboratory evidence in available data"


for required in [ROOT, PATIENT_FILE, ENCOUNTER_FILE]:
    if not required.exists():
        raise FileNotFoundError(required)

N8_COMPLETION, n8 = latest_completed(
    N8_ROOT, "LOCKED_N8_VALIDATION_CHARACTERIZATION_AND_WORKLOAD_LOCK_COMPLETE"
)
N8_RUN = N8_COMPLETION.parent
N8_PATIENT_LEVEL = (
    N8_RUN
    / "PRIVATE_INTERNAL_DO_NOT_SHARE"
    / "patient_level_validation_characterization.parquet"
)
N7_COMPLETION, n7 = latest_completed(
    N7_ROOT, "OUTCOME_BLIND_LOCKED_PULSCAR_MEAN_XGB_VALIDATION_SCORING_COMPLETE"
)
VALIDATION_PREDICTORS = Path(n7["validation_predictors"])
AVAILABILITY_FILE = Path(n7["availability_file"])
SCORE_FILE = Path(n7["score_file"])

for required in [
    N8_PATIENT_LEVEL,
    VALIDATION_PREDICTORS,
    AVAILABILITY_FILE,
    SCORE_FILE,
]:
    if not required.is_file():
        raise FileNotFoundError(required)

if n8.get("model_trained_tuned_or_changed") is not False:
    raise ValueError("N8 reports a model change.")
if n8.get("rank_based_workload_rule_locked") is not True:
    raise ValueError("N8 rank policy is not locked.")
if n8.get("test_accessed") is not False:
    raise ValueError("N8 reports Test access.")
if n7.get("model_changed") is not False:
    raise ValueError("N7 reports a model change.")
if sha256_file(SCORE_FILE) != n7["score_file_sha256"]:
    raise ValueError("The frozen Validation score artifact changed.")
if sha256_file(AVAILABILITY_FILE) != n7["availability_file_sha256"]:
    raise ValueError("The frozen Validation availability artifact changed.")

characterization = pd.read_parquet(N8_PATIENT_LEVEL)
required_characterization = {
    ID_COL,
    "_PATIENT_KEY",
    SCORE_COL,
    LABEL_COL,
    "REVIEW_RANK",
    "REVIEW_PERCENTILE",
    "REVIEW_PRIORITY_CATEGORY",
}
missing = sorted(required_characterization - set(characterization.columns))
if missing:
    raise KeyError(f"N8 patient-level artifact lacks: {missing}")
if len(characterization) != EXPECTED_VALIDATION_PATIENTS:
    raise ValueError("Unexpected N8 Validation patient count.")

cases = characterization.loc[
    characterization[LABEL_COL].eq(1)
    & characterization["REVIEW_PRIORITY_CATEGORY"].eq("Low")
].copy()
if len(cases) != EXPECTED_CASES:
    raise ValueError(
        f"Expected {EXPECTED_CASES} documented-CKD Low-priority cases; "
        f"found {len(cases)}."
    )
cases["_PATIENT_KEY"] = patient_key(cases[ID_COL])
if cases["_PATIENT_KEY"].isna().any() or cases["_PATIENT_KEY"].duplicated().any():
    raise ValueError("The 16 selected Validation IDs are invalid.")
cases = cases.sort_values(
    [SCORE_COL, "_PATIENT_KEY"], ascending=[False, True], kind="stable"
).reset_index(drop=True)
cases["audit_case_code"] = [
    f"VAL-LOW-CKD-{index:02d}" for index in range(1, len(cases) + 1)
]

# Confirm the frozen original-observation mask for the exact 16 cases.
availability = pd.read_parquet(AVAILABILITY_FILE)
availability["_PATIENT_KEY"] = patient_key(availability[ID_COL])
if availability["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Validation availability identifiers are duplicated.")
available_feature_columns = [
    feature for feature in MODEL_FEATURES if feature in availability.columns
]
if available_feature_columns != MODEL_FEATURES:
    raise KeyError(
        "Frozen availability artifact does not contain the exact 20 model features."
    )
availability["MODEL_FEATURES_OBSERVED_N"] = availability[
    MODEL_FEATURES
].astype(bool).sum(axis=1)
availability["MODEL_FEATURES_MEAN_IMPUTED_N"] = (
    len(MODEL_FEATURES) - availability["MODEL_FEATURES_OBSERVED_N"]
)
cases = cases.merge(
    availability[
        [
            "_PATIENT_KEY",
            "MODEL_FEATURES_OBSERVED_N",
            "MODEL_FEATURES_MEAN_IMPUTED_N",
        ]
    ],
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
if cases["MODEL_FEATURES_OBSERVED_N"].isna().any():
    raise ValueError("One or more audited cases lack the frozen availability record.")

# The predictor table is Validation-only and contains no outcome. Read model
# variables plus the five features excluded before final development.
predictor_schema = pq.ParquetFile(VALIDATION_PREDICTORS).schema_arrow.names
required_predictor_columns = [ID_COL, *MODEL_FEATURES]
missing = sorted(set(required_predictor_columns) - set(predictor_schema))
if missing:
    raise KeyError(f"Validation predictor table lacks model features: {missing}")
supporting_columns = [
    feature for feature in EXCLUDED_RENAL_FEATURES if feature in predictor_schema
]
predictors = pd.read_parquet(
    VALIDATION_PREDICTORS,
    columns=required_predictor_columns + supporting_columns,
)
predictors["_PATIENT_KEY"] = patient_key(predictors[ID_COL])
if len(predictors) != EXPECTED_VALIDATION_PATIENTS:
    raise ValueError("Unexpected Validation predictor count.")
if predictors["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Validation predictor identifiers are duplicated.")
predictors = predictors.loc[
    predictors["_PATIENT_KEY"].isin(cases["_PATIENT_KEY"])
].copy()
if len(predictors) != EXPECTED_CASES:
    raise ValueError("Not all 16 audited cases linked to Validation predictors.")

for feature in MODEL_FEATURES:
    if feature != "GENDER":
        predictors[feature] = pd.to_numeric(predictors[feature], errors="coerce")
for feature in supporting_columns:
    predictors[feature] = pd.to_numeric(predictors[feature], errors="coerce")

audit = cases.merge(
    predictors.drop(columns=[ID_COL]),
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)

for feature in EXCLUDED_RENAL_FEATURES:
    if feature not in audit.columns:
        audit[feature] = np.nan
audit["CURRENT_UACR_MG_MMOL"] = np.expm1(audit["UACR_LOG1P_FINAL"])
audit.loc[audit["UACR_LOG1P_FINAL"].isna(), "CURRENT_UACR_MG_MMOL"] = np.nan

# Retrieve dates and uACR history only for these 16 Validation IDs. These data
# are supporting post-prediction evidence and never enter the locked model.
raw_case_ids = cases[ID_COL].tolist()
patient_schema = pq.ParquetFile(PATIENT_FILE).schema_arrow.names
patient_columns = [ID_COL, "PATIENT_INDEX_DATE"]
if "UACR_FINAL_MG_MMOL" in patient_schema:
    patient_columns.append("UACR_FINAL_MG_MMOL")
patient = read_selected_rows(PATIENT_FILE, patient_columns, raw_case_ids)
patient["_PATIENT_KEY"] = patient_key(patient[ID_COL])
patient["PATIENT_INDEX_DATE"] = pd.to_datetime(
    patient["PATIENT_INDEX_DATE"], errors="coerce"
)
if patient["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Patient checkpoint contains duplicate audited IDs.")
audit = audit.merge(
    patient.drop(columns=[ID_COL]),
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
if audit["PATIENT_INDEX_DATE"].isna().any():
    raise ValueError("One or more audited cases lack an index date.")

encounter_columns = [ID_COL, "ENCOUNTER_DATE", "UACR_FINAL_MG_MMOL"]
encounters = read_selected_rows(ENCOUNTER_FILE, encounter_columns, raw_case_ids)
encounters["_PATIENT_KEY"] = patient_key(encounters[ID_COL])
encounters["ENCOUNTER_DATE"] = pd.to_datetime(
    encounters["ENCOUNTER_DATE"], errors="coerce"
)
encounters["UACR_FINAL_MG_MMOL"] = pd.to_numeric(
    encounters["UACR_FINAL_MG_MMOL"], errors="coerce"
)
encounters = encounters.loc[
    encounters["ENCOUNTER_DATE"].notna()
    & encounters["UACR_FINAL_MG_MMOL"].ge(0)
].copy()

uacr_history_rows = []
for _, row in audit[["_PATIENT_KEY", "PATIENT_INDEX_DATE"]].iterrows():
    key = row["_PATIENT_KEY"]
    index_date = row["PATIENT_INDEX_DATE"]
    history = encounters.loc[
        encounters["_PATIENT_KEY"].eq(key)
        & encounters["ENCOUNTER_DATE"].le(index_date)
        & encounters["ENCOUNTER_DATE"].ge(index_date - pd.Timedelta(days=730))
    ].sort_values("ENCOUNTER_DATE")
    history = history.drop_duplicates("ENCOUNTER_DATE", keep="last")

    latest_value = np.nan
    latest_date = pd.NaT
    previous_value = np.nan
    previous_date = pd.NaT
    days_between = np.nan
    if len(history):
        latest = history.iloc[-1]
        latest_value = float(latest["UACR_FINAL_MG_MMOL"])
        latest_date = latest["ENCOUNTER_DATE"]
        qualifying = history.loc[
            history["ENCOUNTER_DATE"].le(latest_date - pd.Timedelta(days=90))
        ]
        if len(qualifying):
            previous = qualifying.iloc[-1]
            previous_value = float(previous["UACR_FINAL_MG_MMOL"])
            previous_date = previous["ENCOUNTER_DATE"]
            days_between = int((latest_date - previous_date).days)

    uacr_history_rows.append(
        {
            "_PATIENT_KEY": key,
            "LATEST_OBSERVED_UACR_MG_MMOL": latest_value,
            "LATEST_OBSERVED_UACR_DATE": latest_date,
            "PREVIOUS_UACR_90D_MG_MMOL": previous_value,
            "PREVIOUS_UACR_DATE": previous_date,
            "UACR_DAYS_BETWEEN": days_between,
        }
    )

audit = audit.merge(
    pd.DataFrame(uacr_history_rows),
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)

# If both sources contain the current uACR, require agreement rather than
# silently choosing one. Use the longitudinal latest value for chronicity.
both_current = (
    audit["CURRENT_UACR_MG_MMOL"].notna()
    & audit["LATEST_OBSERVED_UACR_MG_MMOL"].notna()
)
if both_current.any() and not np.allclose(
    audit.loc[both_current, "CURRENT_UACR_MG_MMOL"],
    audit.loc[both_current, "LATEST_OBSERVED_UACR_MG_MMOL"],
    rtol=1e-5,
    atol=1e-5,
):
    raise ValueError("Current uACR differs between frozen predictors and history.")

audit["CURRENT_EGFR_BAND"] = audit["DERIVED_EGFR_CKD_EPI_2021"].map(egfr_band)
audit["MINIMUM_24M_EGFR_BAND"] = audit["EGFR_MIN_24M"].map(egfr_band)
audit["CURRENT_UACR_BAND"] = audit["CURRENT_UACR_MG_MMOL"].map(uacr_band)
audit["PERSISTENT_LOW_EGFR_90D"] = (
    audit["DERIVED_EGFR_CKD_EPI_2021"].lt(60)
    & audit["EGFR_PREVIOUS_90D"].lt(60)
    & audit["EGFR_DAYS_BETWEEN_90D_MEASUREMENTS"].ge(90)
)
audit["REPEATED_HIGH_UACR_90D"] = (
    audit["LATEST_OBSERVED_UACR_MG_MMOL"].ge(3)
    & audit["PREVIOUS_UACR_90D_MG_MMOL"].ge(3)
    & audit["UACR_DAYS_BETWEEN"].ge(90)
)
audit["ANY_EGFR_BELOW_60_IN_MODEL_WINDOW"] = (
    audit["DERIVED_EGFR_CKD_EPI_2021"].lt(60)
    | audit["EGFR_MIN_24M"].lt(60)
)
audit["COMPUTABLE_KDIGO_ALIGNED_CHRONIC_LAB_EVIDENCE"] = (
    audit["PERSISTENT_LOW_EGFR_90D"] | audit["REPEATED_HIGH_UACR_90D"]
)
audit["RENAL_LAB_PATTERN"] = audit.apply(renal_pattern, axis=1)
audit["CLINICAL_RECORD_ADJUDICATION_REQUIRED"] = ~audit[
    "COMPUTABLE_KDIGO_ALIGNED_CHRONIC_LAB_EVIDENCE"
]

display_columns = [
    "audit_case_code",
    SCORE_COL,
    "REVIEW_PERCENTILE",
    "MODEL_FEATURES_OBSERVED_N",
    "MODEL_FEATURES_MEAN_IMPUTED_N",
    "DERIVED_EGFR_CKD_EPI_2021",
    "CURRENT_EGFR_BAND",
    "EGFR_MIN_24M",
    "MINIMUM_24M_EGFR_BAND",
    "EGFR_MEAN_24M",
    "EGFR_PREVIOUS_90D",
    "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS",
    "PERSISTENT_LOW_EGFR_90D",
    "CURRENT_UACR_MG_MMOL",
    "CURRENT_UACR_BAND",
    "PREVIOUS_UACR_90D_MG_MMOL",
    "UACR_DAYS_BETWEEN",
    "REPEATED_HIGH_UACR_90D",
    "COMPUTABLE_KDIGO_ALIGNED_CHRONIC_LAB_EVIDENCE",
    "RENAL_LAB_PATTERN",
    "CLINICAL_RECORD_ADJUDICATION_REQUIRED",
]
deidentified = audit[display_columns].copy()

summary = {
    "documented_ckd_low_priority_cases": int(len(audit)),
    "current_egfr_below_60": int(
        audit["DERIVED_EGFR_CKD_EPI_2021"].lt(60).sum()
    ),
    "minimum_egfr_24m_below_60": int(audit["EGFR_MIN_24M"].lt(60).sum()),
    "persistent_low_egfr_90d": int(audit["PERSISTENT_LOW_EGFR_90D"].sum()),
    "current_uacr_observed": int(audit["CURRENT_UACR_MG_MMOL"].notna().sum()),
    "current_uacr_A2_or_A3": int(audit["CURRENT_UACR_MG_MMOL"].ge(3).sum()),
    "two_uacr_values_at_least_90d_apart": int(
        audit["PREVIOUS_UACR_90D_MG_MMOL"].notna().sum()
    ),
    "repeated_high_uacr_90d": int(audit["REPEATED_HIGH_UACR_90D"].sum()),
    "computable_kdigo_aligned_chronic_lab_evidence": int(
        audit["COMPUTABLE_KDIGO_ALIGNED_CHRONIC_LAB_EVIDENCE"].sum()
    ),
    "clinical_record_adjudication_required": int(
        audit["CLINICAL_RECORD_ADJUDICATION_REQUIRED"].sum()
    ),
    "median_model_features_observed": float(
        audit["MODEL_FEATURES_OBSERVED_N"].median()
    ),
    "range_model_features_observed": [
        int(audit["MODEL_FEATURES_OBSERVED_N"].min()),
        int(audit["MODEL_FEATURES_OBSERVED_N"].max()),
    ],
}

context = {
    "step": "5.12N9",
    "status": "VALIDATION_LOW_PRIORITY_DOCUMENTED_CKD_SAFETY_AUDIT",
    "n8_completion_sha256": sha256_file(N8_COMPLETION),
    "n8_patient_level_sha256": sha256_file(N8_PATIENT_LEVEL),
    "n7_completion_sha256": sha256_file(N7_COMPLETION),
    "validation_predictors_sha256": sha256_file(VALIDATION_PREDICTORS),
    "availability_sha256": sha256_file(AVAILABILITY_FILE),
    "audit_cohort": "Validation only",
    "audited_cases": EXPECTED_CASES,
    "uacr_role": (
        "Excluded from model development and prediction; post-prediction "
        "supporting safety evidence only."
    ),
    "excluded_90d_egfr_role": (
        "Excluded from model development and prediction; post-prediction "
        "supporting safety evidence only."
    ),
    "model_or_preprocessing_changed": False,
    "scores_or_priority_categories_changed": False,
    "test_accessed": False,
    "patient_identifiers_publicly_written": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
atomic_json(OUTPUT_DIR / "context.json", context)

PRIVATE_AUDIT = PRIVATE_DIR / (
    "validation_low_priority_documented_ckd_private_safety_audit.parquet"
)
PRIVATE_LINKAGE = PRIVATE_DIR / (
    "validation_low_priority_documented_ckd_private_linkage.csv"
)
DEIDENTIFIED_AUDIT = OUTPUT_DIR / (
    "validation_low_priority_documented_ckd_deidentified_safety_audit.csv"
)
SUMMARY_FILE = OUTPUT_DIR / "validation_low_priority_safety_audit_summary.json"

audit.to_parquet(PRIVATE_AUDIT, index=False)
audit[["audit_case_code", ID_COL, "_PATIENT_KEY"]].to_csv(
    PRIVATE_LINKAGE, index=False
)
deidentified.to_csv(DEIDENTIFIED_AUDIT, index=False)
atomic_json(SUMMARY_FILE, summary)

completion = {
    **context,
    "status": "VALIDATION_LOW_PRIORITY_DOCUMENTED_CKD_SAFETY_AUDIT_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "summary": summary,
    "output_dir": str(OUTPUT_DIR),
    "private_audit_file": str(PRIVATE_AUDIT),
    "private_audit_sha256": sha256_file(PRIVATE_AUDIT),
    "private_linkage_file": str(PRIVATE_LINKAGE),
    "private_linkage_sha256": sha256_file(PRIVATE_LINKAGE),
    "deidentified_audit_file": str(DEIDENTIFIED_AUDIT),
    "deidentified_audit_sha256": sha256_file(DEIDENTIFIED_AUDIT),
    "model_changed": False,
    "preprocessing_changed": False,
    "scores_changed": False,
    "priority_categories_changed": False,
    "clinician_review_sample_replaced": False,
    "interpretation": (
        "Post-model descriptive safety audit. Absence of computable chronic "
        "laboratory evidence does not invalidate a documented CKD diagnosis: "
        "the diagnosis may predate the window, be based on imaging/pathology/"
        "etiology, reflect treatment or recovery, or require chart review. "
        "The audit cannot be used to retune the locked model."
    ),
    "next_step": (
        "Freeze the post-redevelopment internal Test protocol without accessing "
        "Test predictors or outcomes."
    ),
}
atomic_json(OUTPUT_DIR / "completion.json", completion)

print("=" * 116)
print(
    "STEP 5.12N9 — VALIDATION LOW-PRIORITY DOCUMENTED-CKD "
    "SAFETY AUDIT COMPLETE"
)
print("=" * 116)
print("\nAUDIT SUMMARY")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nDEIDENTIFIED CASE-LEVEL SAFETY AUDIT")
display(deidentified)
print(f"\nSaved run: {OUTPUT_DIR}")
print(f"Private audit: {PRIVATE_AUDIT}")
print(f"Private linkage: {PRIVATE_LINKAGE}")
print(f"Deidentified audit: {DEIDENTIFIED_AUDIT}")
print("uACR used by model: No — supporting post-prediction evidence only")
print("Excluded 90-day eGFR variables used by model: No")
print("Model, preprocessing, scores, or priority categories changed: No")
print("Internal Test accessed: No")
print("Patient identifiers publicly written: No")
