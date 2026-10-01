"""STEP 5.12N14 — Locked blinded 15-case clinician-review sample.

Creates a deterministic, information-oriented sample from the locked Internal
Test cohort:

  * all five alpha-selected patients with repeated low eGFR >=90 days apart;
  * five systematically spaced alpha-selected patients without computable
    chronicity; and
  * the five unlabeled patients immediately below the alpha selection boundary.

The Phase-A casebook contains only observed values from the locked 20-feature
model panel. It contains no patient identifier, model output, percentile, rank,
SHAP value, sampling role, observed label, uACR, or imputed value.
"""

from google.colab import drive

drive.mount("/content/drive")

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display


EXPECTED_CASES = 15
EXPECTED_PERSISTENT_CASES = 5
ID_COL = "patient_id"
LABEL_COL = "OBSERVED_CKD_LABEL"
SCORE_COL = "locked_pulscar_mean_xgb_internal_test_score"
PERCENT_COL = "model_output_percentage"
PERCENTILE_COL = "within_cohort_percentile"

ROOT = Path(
    "/content/drive/MyDrive/CKD_Project/outputs/"
    "section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
N6_ROOT = ROOT / "05_12N6_full_training_pulscar_mean_xgb_final_refit_and_lock"
N11_ROOT = ROOT / (
    "05_12N11_outcome_blind_locked_pulscar_mean_xgb_internal_test_scoring"
)
N12_ROOT = ROOT / "05_12N12_locked_internal_test_evaluation"
N13_ROOT = ROOT / "05_12N13_alpha_review_integrity_and_egfr_benchmark"
N13A_ROOT = ROOT / "05_12N13A_alpha_review_temporal_egfr_replay"
OUTPUT_ROOT = ROOT / "05_12N14_locked_blinded_clinician_review_sample"
OUTCOME_SOURCE = ROOT / (
    "05_8E_one_time_final_test_outcome_evaluation/"
    "cc3336e8918ba542f405061c/PRIVATE_INTERNAL_DO_NOT_SHARE/"
    "final_test_labels_evidence_scores_and_percentiles.parquet"
)

DISPLAY_SALT = "N14|blinded-clinician-display|fixed-v1"

CLINICAL_METADATA = {
    "AGE_AT_INDEX": ("Demographics", "Age at index", "years", 0),
    "GENDER": ("Demographics", "Gender", "", None),
    "BMI_MODEL": ("Vitals and anthropometrics", "Body mass index", "kg/m²", 1),
    "SBP": ("Vitals and anthropometrics", "Systolic blood pressure", "mmHg", 0),
    "DBP": ("Vitals and anthropometrics", "Diastolic blood pressure", "mmHg", 0),
    "PULSE_FINAL": ("Vitals and anthropometrics", "Pulse", "beats/min", 0),
    "DERIVED_EGFR_CKD_EPI_2021": (
        "Kidney function",
        "Current derived eGFR",
        "mL/min/1.73 m²",
        1,
    ),
    "EGFR_MIN_24M": (
        "Kidney function",
        "Minimum eGFR within the 24-month observation window",
        "mL/min/1.73 m²",
        1,
    ),
    "EGFR_MEAN_24M": (
        "Kidney function",
        "Mean eGFR within the 24-month observation window",
        "mL/min/1.73 m²",
        1,
    ),
    "BUN_FINAL_MMOL_L": ("Chemistry", "Blood urea nitrogen", "mmol/L", 1),
    "POTASSIUM_FINAL_MMOL_L": ("Chemistry", "Potassium", "mmol/L", 1),
    "SODIUM_FINAL_MMOL_L": ("Chemistry", "Sodium", "mmol/L", 0),
    "HEMOGLOBIN_MODEL_G_L": ("Hematology", "Hemoglobin", "g/L", 0),
    "MCV_MODEL_FL": ("Hematology", "Mean corpuscular volume", "fL", 1),
    "RDW_CV_MODEL_PERCENT": ("Hematology", "Red cell distribution width", "%", 1),
    "WBC_MODEL_10E9_L": ("Hematology", "White blood cell count", "10⁹/L", 1),
    "PLATELETS_MODEL_10E9_L": ("Hematology", "Platelet count", "10⁹/L", 0),
    "MPV_MODEL_FL": ("Hematology", "Mean platelet volume", "fL", 1),
    "LYMPHOCYTES_ABSOLUTE_MODEL_10E9_L": (
        "Hematology",
        "Absolute lymphocyte count",
        "10⁹/L",
        2,
    ),
    "EOSINOPHILS_ABSOLUTE_MODEL_10E9_L": (
        "Hematology",
        "Absolute eosinophil count",
        "10⁹/L",
        2,
    ),
}


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


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


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


def normalize(value):
    return "".join(character.lower() for character in str(value) if character.isalnum())


def resolve_name(names, candidates, token_groups, description):
    lookup = {normalize(name): name for name in names}
    for candidate in candidates:
        if normalize(candidate) in lookup:
            return lookup[normalize(candidate)]
    matches = []
    for name in names:
        normalized = normalize(name)
        if all(
            any(normalize(token) in normalized for token in group)
            for group in token_groups
        ):
            matches.append(name)
    if len(matches) != 1:
        raise KeyError(f"Could not uniquely resolve {description}: {matches}")
    return matches[0]


def patient_key(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        nonmissing = numeric.dropna()
        if (np.abs(nonmissing - np.round(nonmissing)) < 1e-9).all():
            return numeric.round().astype("Int64").astype("string")
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def systematic_positions(pool_size, sample_size):
    if pool_size < sample_size:
        raise ValueError(
            f"Cannot select {sample_size} systematic cases from pool of {pool_size}."
        )
    positions = np.rint(np.linspace(0, pool_size - 1, sample_size)).astype(int)
    if len(np.unique(positions)) != sample_size:
        raise ValueError("Systematic sampling positions are not unique.")
    return positions


def display_value(value, decimals):
    if pd.isna(value):
        return "Not available"
    if decimals is None:
        return str(value).strip()
    numeric = float(value)
    if decimals == 0:
        return str(int(round(numeric)))
    return f"{numeric:.{decimals}f}"


N6_COMPLETION, n6 = latest_completed(
    N6_ROOT, "FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED"
)
N11_COMPLETION, n11 = latest_completed(
    N11_ROOT, "OUTCOME_BLIND_LOCKED_INTERNAL_TEST_SCORING_COMPLETE"
)
N12_COMPLETION, n12 = latest_completed(
    N12_ROOT, "LOCKED_INTERNAL_TEST_EVALUATION_COMPLETE"
)
N13_COMPLETION, n13 = latest_completed(
    N13_ROOT, "ALPHA_REVIEW_INTEGRITY_AND_EGFR_BENCHMARK_COMPLETE"
)
N13A_COMPLETION, n13a = latest_completed(
    N13A_ROOT, "ALPHA_REVIEW_TEMPORAL_EGFR_REPLAY_COMPLETE"
)

N6_RUN = N6_COMPLETION.parent
PREPROCESSOR = N6_RUN / "locked_mean_imputation_preprocessor.joblib"
SCORE_FILE = Path(n11["score_file"])
TEST_PREDICTORS = Path(n11["internal_test_predictors"])
N13A_PRIVATE = N13A_COMPLETION.parent / (
    "PRIVATE_INTERNAL_DO_NOT_SHARE/private_alpha_temporal_egfr_replay.parquet"
)

for required in [
    N6_COMPLETION,
    N11_COMPLETION,
    N12_COMPLETION,
    N13_COMPLETION,
    N13A_COMPLETION,
    PREPROCESSOR,
    SCORE_FILE,
    TEST_PREDICTORS,
    OUTCOME_SOURCE,
    N13A_PRIVATE,
]:
    if not required.is_file():
        raise FileNotFoundError(required)

if n13a.get("hard_integrity_failures") != 0:
    raise ValueError("N13A temporal replay did not pass.")
if n13a.get("uacr_loaded_or_reported") is not False:
    raise ValueError("N13A reports uACR use.")
if n13a.get("model_changed") is not False:
    raise ValueError("N13A reports a model change.")
if sha256_file(SCORE_FILE) != n11["score_file_sha256"]:
    raise ValueError("The frozen N11 score file changed.")
if sha256_file(TEST_PREDICTORS) != n11["internal_test_predictors_sha256"]:
    raise ValueError("The authoritative Test predictor table changed.")
if sha256_file(OUTCOME_SOURCE) != n12["outcome_source_sha256"]:
    raise ValueError("The N12 outcome source changed.")

preprocessor = joblib.load(PREPROCESSOR)
numeric_features = list(preprocessor["numeric_features"])
categorical_feature = str(preprocessor["categorical_feature"])
model_features = numeric_features + [categorical_feature]
if len(model_features) != 20 or set(model_features) != set(CLINICAL_METADATA):
    raise ValueError(f"Unexpected locked raw feature panel: {model_features}")
if any("uacr" in feature.lower() for feature in model_features):
    raise ValueError("uACR is present in the locked feature panel.")

# Rebuild all unlabeled ranks with the exact N12 tie rule.
scores = pd.read_parquet(SCORE_FILE)
scores["_PATIENT_KEY"] = patient_key(scores[ID_COL])
if scores["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Frozen scores contain duplicate patient keys.")
outcome_schema = pq.ParquetFile(OUTCOME_SOURCE).schema_arrow.names
outcome_id = resolve_name(
    outcome_schema,
    [ID_COL, "PATIENT_ID"],
    [("patient",), ("id",)],
    "outcome patient ID",
)
outcome_label = resolve_name(
    outcome_schema,
    ["observed_label", "CKD_CLASS", "documented_ckd", "OBSERVED_CKD_LABEL"],
    [("ckd",), ("class", "label")],
    "observed CKD label",
)
outcomes = pd.read_parquet(OUTCOME_SOURCE, columns=[outcome_id, outcome_label])
outcomes["_PATIENT_KEY"] = patient_key(outcomes[outcome_id])
outcomes[LABEL_COL] = pd.to_numeric(outcomes[outcome_label], errors="raise").astype(int)
ranked = scores.merge(
    outcomes[["_PATIENT_KEY", LABEL_COL]],
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
if ranked[LABEL_COL].isna().any():
    raise ValueError("Some frozen scores lack Test labels.")
unlabeled = ranked.loc[ranked[LABEL_COL].eq(0)].copy()
unlabeled["_TIE_HASH"] = unlabeled["_PATIENT_KEY"].map(
    lambda value: hashlib.sha256(
        f"N12|alpha-review|{value}".encode("utf-8")
    ).hexdigest()
)
unlabeled = unlabeled.sort_values(
    [SCORE_COL, "_TIE_HASH"], ascending=[False, True], kind="stable"
).reset_index(drop=True)
unlabeled["unlabeled_score_rank"] = np.arange(1, len(unlabeled) + 1, dtype=int)
alpha_n = int(math.floor(float(n6["estimated_alpha"]) * len(unlabeled)))
if alpha_n != EXPECTED_ALPHA_CASES:
    raise ValueError(f"Expected alpha boundary at 56; found {alpha_n}.")
unlabeled["inside_alpha_review_set"] = unlabeled["unlabeled_score_rank"].le(alpha_n)

temporal = pd.read_parquet(N13A_PRIVATE)
temporal["_PATIENT_KEY"] = patient_key(temporal[ID_COL])
if len(temporal) != EXPECTED_ALPHA_CASES:
    raise ValueError("N13A private temporal audit does not contain 56 cases.")
if temporal["_PATIENT_KEY"].duplicated().any():
    raise ValueError("N13A private temporal audit contains duplicate cases.")
if int(temporal["REPLAY_PERSISTENT_LOW_EGFR_90D"].sum()) != EXPECTED_PERSISTENT_CASES:
    raise ValueError("Expected five persistent-low-eGFR alpha cases.")

alpha_pool = unlabeled.loc[unlabeled["inside_alpha_review_set"]].merge(
    temporal[
        [
            "_PATIENT_KEY",
            "REPLAY_PERSISTENT_LOW_EGFR_90D",
            "REPLAY_DISTINCT_EGFR_DATES",
            "REPLAY_PREVIOUS_EGFR_90D",
            "REPLAY_EGFR_GAP_DAYS",
        ]
    ],
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
if alpha_pool["REPLAY_PERSISTENT_LOW_EGFR_90D"].isna().any():
    raise ValueError("N13A temporal evidence did not link to every alpha case.")

persistent = alpha_pool.loc[
    alpha_pool["REPLAY_PERSISTENT_LOW_EGFR_90D"].astype(bool)
].copy()
persistent["sampling_role"] = (
    "Alpha-selected; repeated low eGFR at least 90 days apart"
)

nonpersistent_pool = alpha_pool.loc[
    ~alpha_pool["REPLAY_PERSISTENT_LOW_EGFR_90D"].astype(bool)
].sort_values("unlabeled_score_rank").reset_index(drop=True)
selected_positions = systematic_positions(len(nonpersistent_pool), 5)
alpha_nonpersistent = nonpersistent_pool.iloc[selected_positions].copy()
alpha_nonpersistent["sampling_role"] = (
    "Alpha-selected; no computable repeated-low-eGFR pair"
)

boundary = unlabeled.loc[
    unlabeled["unlabeled_score_rank"].between(alpha_n + 1, alpha_n + 5)
].copy()
if len(boundary) != 5:
    raise ValueError("Could not select exactly five immediate boundary comparators.")
boundary["sampling_role"] = "Immediately below the PULSCAR-alpha boundary"
boundary["REPLAY_PERSISTENT_LOW_EGFR_90D"] = pd.NA
boundary["REPLAY_DISTINCT_EGFR_DATES"] = pd.NA
boundary["REPLAY_PREVIOUS_EGFR_90D"] = np.nan
boundary["REPLAY_EGFR_GAP_DAYS"] = np.nan

selected = pd.concat(
    [persistent, alpha_nonpersistent, boundary],
    ignore_index=True,
    sort=False,
)
if len(selected) != EXPECTED_CASES:
    raise ValueError(f"Expected 15 selected cases; found {len(selected)}.")
if selected["_PATIENT_KEY"].duplicated().any():
    raise ValueError("The selected clinician-review cases are not unique.")
if not selected[LABEL_COL].eq(0).all():
    raise ValueError("The clinician-review sample contains documented CKD.")

# Blind role and ranking through a fixed patient-key hash, then assign case codes.
selected["_DISPLAY_HASH"] = selected["_PATIENT_KEY"].map(
    lambda value: hashlib.sha256(
        f"{DISPLAY_SALT}|{value}".encode("utf-8")
    ).hexdigest()
)
selected = selected.sort_values("_DISPLAY_HASH", kind="stable").reset_index(drop=True)
selected["study_case_code"] = [
    f"CKD-CR-{index:03d}" for index in range(1, EXPECTED_CASES + 1)
]
selected["presentation_order"] = np.arange(1, EXPECTED_CASES + 1, dtype=int)

predictors = pd.read_parquet(
    TEST_PREDICTORS,
    columns=[ID_COL, *model_features],
    filters=[(ID_COL, "in", selected[ID_COL].tolist())],
)
predictors["_PATIENT_KEY"] = patient_key(predictors[ID_COL])
if predictors["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Selected predictor records are duplicated.")
case_data = selected.merge(
    predictors.drop(columns=[ID_COL]),
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
if len(case_data) != EXPECTED_CASES:
    raise ValueError("Case-data linkage changed the selected sample size.")

# Build the clinician-facing Phase-A casebook with clinical names and units.
casebook_rows = []
json_cases = []
for _, row in case_data.sort_values("presentation_order").iterrows():
    json_fields = []
    wide = {"Study case code": row["study_case_code"]}
    for feature in model_features:
        section, clinical_name, unit, decimals = CLINICAL_METADATA[feature]
        shown = display_value(row[feature], decimals)
        heading = f"{clinical_name} ({unit})" if unit else clinical_name
        wide[heading] = shown
        json_fields.append(
            {
                "section": section,
                "clinical_measure": clinical_name,
                "value": shown,
                "unit": unit,
                "availability": "Observed" if pd.notna(row[feature]) else "Not available",
            }
        )
    casebook_rows.append(wide)
    json_cases.append(
        {
            "study_case_code": row["study_case_code"],
            "clinical_information": json_fields,
            "missing_information_notice": (
                "Not available means that the measurement was absent from the "
                "study dataset; it must not be interpreted as a normal result."
            ),
        }
    )
casebook = pd.DataFrame(casebook_rows)

phase_a_questions = [
    {
        "question_id": "A1",
        "question": (
            "Do the presented data show evidence of current or previous kidney "
            "function impairment?"
        ),
        "response_options": ["Yes", "No", "Uncertain"],
    },
    {
        "question_id": "A2",
        "question": (
            "Do the presented data establish chronic kidney disease, including "
            "evidence of persistence for at least three months?"
        ),
        "response_options": ["Yes", "No", "Insufficient information"],
    },
    {
        "question_id": "A3",
        "question": "Would you recommend additional clinical-record review?",
        "response_options": ["Yes", "No", "Uncertain"],
    },
    {
        "question_id": "A4",
        "question": "Would you recommend repeat kidney-function testing?",
        "response_options": ["Yes", "No", "Uncertain"],
    },
    {
        "question_id": "A5",
        "question": "Optional clinical comments",
        "response_options": ["Free text"],
    },
]

observed_counts = case_data[model_features].notna().sum(axis=1)
role_table = (
    selected.groupby("sampling_role", dropna=False)
    .size()
    .rename("Selected_cases")
    .reset_index()
)
availability_audit = pd.DataFrame(
    {
        "study_case_code": case_data["study_case_code"],
        "observed_model_features": observed_counts.astype(int),
        "missing_model_features": (len(model_features) - observed_counts).astype(int),
    }
)

for forbidden in [
    ID_COL,
    SCORE_COL,
    PERCENT_COL,
    PERCENTILE_COL,
    "rank",
    "priority",
    "shap",
    "uacr",
    "sampling_role",
    "label",
]:
    if any(forbidden.lower() in str(column).lower() for column in casebook.columns):
        raise ValueError(f"Forbidden Phase-A casebook field leaked: {forbidden}")
serialized_casebook = json.dumps(
    {"cases": json_cases, "questions": phase_a_questions},
    ensure_ascii=False,
).lower()
for forbidden_text in ["patient_id", "score", "percentile", "shap", "uacr", "sampling_role"]:
    if forbidden_text in serialized_casebook:
        raise ValueError(f"Forbidden Phase-A JSON content leaked: {forbidden_text}")

context = {
    "step": "5.12N14",
    "status": "LOCKED_BLINDED_15_CASE_CLINICIAN_REVIEW_SAMPLE",
    "n6_completion_sha256": sha256_file(N6_COMPLETION),
    "n11_completion_sha256": sha256_file(N11_COMPLETION),
    "n12_completion_sha256": sha256_file(N12_COMPLETION),
    "n13_completion_sha256": sha256_file(N13_COMPLETION),
    "n13a_completion_sha256": sha256_file(N13A_COMPLETION),
    "n13a_private_temporal_audit_sha256": sha256_file(N13A_PRIVATE),
    "test_predictors_sha256": sha256_file(TEST_PREDICTORS),
    "selection_design": {
        "persistent_alpha_cases": 5,
        "systematically_spaced_nonpersistent_alpha_cases": 5,
        "immediate_below_alpha_boundary_cases": 5,
        "nonpersistent_alpha_sampling_positions_zero_based": selected_positions.tolist(),
        "display_order": "Fixed salted SHA256 of private patient key",
    },
    "clinician_review_purpose": (
        "Information-oriented blinded face-validity assessment; not prevalence "
        "estimation, diagnostic-accuracy estimation, or gold-standard adjudication."
    ),
    "clinician_terminology_version": "gender-v2",
    "gender_display_label": "Gender",
    "uacr_loaded_or_displayed": False,
    "model_score_percentile_rank_or_shap_in_phase_a": False,
    "imputed_values_displayed_as_observed": False,
    "model_or_scores_changed": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

PHASE_A_CSV = PRIVATE_DIR / "blinded_phase_a_clinician_casebook_15_cases.csv"
PHASE_A_JSON = PRIVATE_DIR / "blinded_phase_a_clinician_casebook_15_cases.json"
PRIVATE_LINKAGE = PRIVATE_DIR / "clinician_review_private_linkage.parquet"

atomic_json(OUTPUT_DIR / "context.json", context)
atomic_csv(role_table, OUTPUT_DIR / "selected_sampling_design_summary.csv")
atomic_csv(availability_audit, OUTPUT_DIR / "phase_a_feature_availability_audit.csv")
atomic_csv(casebook, PHASE_A_CSV)
atomic_json(
    PHASE_A_JSON,
    {
        "phase": "A — Independent clinical assessment",
        "instructions": (
            "Review each case using only the displayed clinical information. "
            "Model results will be shown only after Phase A responses are locked."
        ),
        "cases": json_cases,
        "questions": phase_a_questions,
    },
)
atomic_parquet(
    selected[
        [
            ID_COL,
            "_PATIENT_KEY",
            "study_case_code",
            "presentation_order",
            "sampling_role",
            "unlabeled_score_rank",
            "inside_alpha_review_set",
            SCORE_COL,
            PERCENT_COL,
            PERCENTILE_COL,
            "REPLAY_PERSISTENT_LOW_EGFR_90D",
            "REPLAY_DISTINCT_EGFR_DATES",
            "REPLAY_PREVIOUS_EGFR_90D",
            "REPLAY_EGFR_GAP_DAYS",
        ]
    ],
    PRIVATE_LINKAGE,
)

completion = {
    **context,
    "status": "LOCKED_BLINDED_15_CASE_CLINICIAN_REVIEW_SAMPLE_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(OUTPUT_DIR),
    "phase_a_csv": str(PHASE_A_CSV),
    "phase_a_csv_sha256": sha256_file(PHASE_A_CSV),
    "phase_a_json": str(PHASE_A_JSON),
    "phase_a_json_sha256": sha256_file(PHASE_A_JSON),
    "private_linkage": str(PRIVATE_LINKAGE),
    "private_linkage_sha256": sha256_file(PRIVATE_LINKAGE),
    "selected_cases": EXPECTED_CASES,
    "selected_documented_ckd": 0,
    "selected_unlabeled": EXPECTED_CASES,
    "patient_identifiers_in_phase_a_casebook": False,
    "model_results_in_phase_a_casebook": False,
    "model_changed": False,
    "clinician_review_responses_collected": False,
    "next_step": (
        "Compute observed-model-feature-only local SHAP for the locked 15 cases "
        "and build the separate Phase-B reveal package without modifying Phase A."
    ),
}
atomic_json(OUTPUT_DIR / "completion.json", completion)

print("=" * 116)
print("STEP 5.12N14 — LOCKED BLINDED 15-CASE CLINICIAN-REVIEW SAMPLE COMPLETE")
print("=" * 116)
print("\nSELECTED SAMPLING DESIGN")
display(role_table)
print("\nPHASE-A FEATURE AVAILABILITY AUDIT")
display(availability_audit)
print("\nSAFETY CHECKS")
print(
    json.dumps(
        {
            "selected_cases": EXPECTED_CASES,
            "selected_documented_ckd": 0,
            "selected_unlabeled": EXPECTED_CASES,
            "patient_identifiers_in_phase_a": False,
            "scores_percentiles_ranks_or_shap_in_phase_a": False,
            "uacr_loaded_or_displayed": False,
            "imputed_values_displayed_as_observed": False,
        },
        indent=2,
    )
)
print(f"\nSaved run: {OUTPUT_DIR}")
print(f"Private blinded Phase-A CSV: {PHASE_A_CSV}")
print(f"Private blinded Phase-A JSON: {PHASE_A_JSON}")
print(f"Private linkage — do not send to reviewers: {PRIVATE_LINKAGE}")
print("Model or frozen scores changed: No")
print("Clinician responses collected: No")
