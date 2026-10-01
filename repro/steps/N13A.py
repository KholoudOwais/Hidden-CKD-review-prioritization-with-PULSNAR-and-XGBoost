"""STEP 5.12N13A — Alpha-review temporal eGFR replay.

Reconstructs the eGFR summaries for the 56 locked alpha-review patients from
dated encounter records restricted to the 730 days ending at each patient's
index date.  It verifies that no post-index measurement enters the replay and
requires exact agreement (within floating-point tolerance) with the frozen
Internal Test predictors.

This is a private integrity audit.  It loads no uACR, changes no model output,
and performs no model or threshold selection.
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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display


EXPECTED_ALPHA_CASES = 56
WINDOW_DAYS = 730
MINIMUM_GAP_DAYS = 90
ID_COL = "patient_id"
LABEL_COL = "OBSERVED_CKD_LABEL"
SCORE_COL = "locked_pulscar_mean_xgb_internal_test_score"
CURRENT_EGFR = "DERIVED_EGFR_CKD_EPI_2021"
MIN_EGFR = "EGFR_MIN_24M"
MEAN_EGFR = "EGFR_MEAN_24M"

PROJECT_ROOT = Path("/content/drive/MyDrive/CKD_Project")
ROOT = PROJECT_ROOT / (
    "outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
CHECKPOINT = PROJECT_ROOT / "checkpoints/step_3_15_6_20260903_230014"
PATIENT_FILE = CHECKPOINT / "patient_level_data.parquet"
ENCOUNTER_FILE = CHECKPOINT / "df.parquet"
N6_ROOT = ROOT / "05_12N6_full_training_pulscar_mean_xgb_final_refit_and_lock"
N11_ROOT = ROOT / (
    "05_12N11_outcome_blind_locked_pulscar_mean_xgb_internal_test_scoring"
)
N12_ROOT = ROOT / "05_12N12_locked_internal_test_evaluation"
N13_ROOT = ROOT / "05_12N13_alpha_review_integrity_and_egfr_benchmark"
OUTPUT_ROOT = ROOT / "05_12N13A_alpha_review_temporal_egfr_replay"
OUTCOME_SOURCE = ROOT / (
    "05_8E_one_time_final_test_outcome_evaluation/"
    "cc3336e8918ba542f405061c/PRIVATE_INTERNAL_DO_NOT_SHARE/"
    "final_test_labels_evidence_scores_and_percentiles.parquet"
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


def boolean_mask(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).ne(0)
    return series.astype("string").str.strip().str.lower().isin(
        ["true", "t", "yes", "y", "1"]
    )


def read_selected_rows(path, columns, raw_ids, id_column):
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[(id_column, "in", list(raw_ids))],
    )
    requested = set(patient_key(pd.Series(list(raw_ids))).dropna())
    returned = set(patient_key(frame[id_column]).dropna())
    unexpected = returned - requested
    if unexpected:
        raise ValueError(f"Filtered read returned unexpected IDs: {unexpected}")
    return frame


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

SCORE_FILE = Path(n11["score_file"])
TEST_PREDICTORS = Path(n11["internal_test_predictors"])
for required in [
    N6_COMPLETION,
    N11_COMPLETION,
    N12_COMPLETION,
    N13_COMPLETION,
    SCORE_FILE,
    TEST_PREDICTORS,
    OUTCOME_SOURCE,
    PATIENT_FILE,
    ENCOUNTER_FILE,
]:
    if not required.is_file():
        raise FileNotFoundError(required)

if n13.get("model_changed") is not False:
    raise ValueError("N13 reports a model change.")
if n13.get("uacr_loaded_or_reported") is not False:
    raise ValueError("N13 reports uACR use.")
if sha256_file(SCORE_FILE) != n11["score_file_sha256"]:
    raise ValueError("The frozen N11 score file changed.")
if sha256_file(TEST_PREDICTORS) != n11["internal_test_predictors_sha256"]:
    raise ValueError("The authoritative Test predictor table changed.")
if sha256_file(OUTCOME_SOURCE) != n12["outcome_source_sha256"]:
    raise ValueError("The N12 outcome source changed.")

# Reconstruct the exact 56-case alpha set.
scores = pd.read_parquet(SCORE_FILE)
scores["_PATIENT_KEY"] = patient_key(scores[ID_COL])
outcome_schema = pq.ParquetFile(OUTCOME_SOURCE).schema_arrow.names
outcome_id = resolve_name(
    outcome_schema,
    ["patient_id", "PATIENT_ID"],
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
evaluation = scores.merge(
    outcomes[["_PATIENT_KEY", LABEL_COL]],
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
if evaluation[LABEL_COL].isna().any():
    raise ValueError("Some frozen scores lack observed labels.")
unlabeled = evaluation.loc[evaluation[LABEL_COL].eq(0)].copy()
alpha = float(n6["estimated_alpha"])
alpha_n = int(math.floor(alpha * len(unlabeled)))
unlabeled["_TIE_HASH"] = unlabeled["_PATIENT_KEY"].map(
    lambda value: hashlib.sha256(
        f"N12|alpha-review|{value}".encode("utf-8")
    ).hexdigest()
)
alpha_cases = unlabeled.sort_values(
    [SCORE_COL, "_TIE_HASH"], ascending=[False, True], kind="stable"
).head(alpha_n).reset_index(drop=True)
alpha_cases["alpha_review_rank"] = np.arange(1, len(alpha_cases) + 1)
if len(alpha_cases) != EXPECTED_ALPHA_CASES:
    raise ValueError(f"Expected 56 alpha-review cases; found {len(alpha_cases)}.")

raw_ids = alpha_cases[ID_COL].tolist()

# Load patient-specific index dates only for the selected IDs.
patient_schema = pq.ParquetFile(PATIENT_FILE).schema_arrow.names
patient_id = resolve_name(
    patient_schema,
    [ID_COL, "PATIENT_ID"],
    [("patient",), ("id",)],
    "checkpoint patient ID",
)
index_date_col = resolve_name(
    patient_schema,
    ["PATIENT_INDEX_DATE", "INDEX_DATE"],
    [("index",), ("date",)],
    "patient index date",
)
patients = read_selected_rows(
    PATIENT_FILE,
    [patient_id, index_date_col],
    raw_ids,
    patient_id,
)
patients["_PATIENT_KEY"] = patient_key(patients[patient_id])
patients["PATIENT_INDEX_DATE"] = pd.to_datetime(
    patients[index_date_col], errors="coerce"
).dt.normalize()
if patients["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Checkpoint patient records are duplicated.")
if patients["PATIENT_INDEX_DATE"].isna().any():
    raise ValueError("One or more alpha-review patients lack an index date.")

# Load dated eGFR encounters only; uACR is deliberately absent.
encounter_schema = pq.ParquetFile(ENCOUNTER_FILE).schema_arrow.names
encounter_id = resolve_name(
    encounter_schema,
    [patient_id, ID_COL, "PATIENT_ID"],
    [("patient",), ("id",)],
    "encounter patient ID",
)
encounter_date = resolve_name(
    encounter_schema,
    ["ENCOUNTER_DATE", "VISIT_DATE", "RESULT_DATE"],
    [("date",), ("encounter", "visit", "result")],
    "encounter date",
)
encounter_egfr = resolve_name(
    encounter_schema,
    [CURRENT_EGFR],
    [("egfr",), ("2021", "ckdepi")],
    "encounter eGFR",
)
primary_eligible = resolve_name(
    encounter_schema,
    ["PRIMARY_COHORT_ELIGIBLE"],
    [("primary",), ("cohort",), ("eligible",)],
    "primary-cohort eligibility",
)
window_flag = resolve_name(
    encounter_schema,
    ["IN_24_MONTH_OBSERVATION_WINDOW"],
    [("24",), ("month",), ("window",)],
    "24-month window flag",
)
encounters = read_selected_rows(
    ENCOUNTER_FILE,
    [
        encounter_id,
        encounter_date,
        encounter_egfr,
        primary_eligible,
        window_flag,
    ],
    raw_ids,
    encounter_id,
)
encounters["_PATIENT_KEY"] = patient_key(encounters[encounter_id])
encounters["ENCOUNTER_DATE"] = pd.to_datetime(
    encounters[encounter_date], errors="coerce"
).dt.normalize()
encounters["ENCOUNTER_EGFR"] = pd.to_numeric(
    encounters[encounter_egfr], errors="coerce"
)
encounters = encounters.merge(
    patients[["_PATIENT_KEY", "PATIENT_INDEX_DATE"]],
    on="_PATIENT_KEY",
    how="left",
    validate="many_to_one",
)
if encounters["PATIENT_INDEX_DATE"].isna().any():
    raise ValueError("Encounter-to-index-date linkage failed.")

encounters["DAYS_FROM_INDEX"] = (
    encounters["ENCOUNTER_DATE"] - encounters["PATIENT_INDEX_DATE"]
).dt.total_seconds() / 86400.0
encounters["DATE_IN_RECOMPUTED_WINDOW"] = encounters["DAYS_FROM_INDEX"].between(
    -WINDOW_DAYS, 0, inclusive="both"
)
encounters["SOURCE_ELIGIBLE_AND_WINDOW"] = (
    boolean_mask(encounters[primary_eligible])
    & boolean_mask(encounters[window_flag])
)

usable = encounters.loc[
    encounters["SOURCE_ELIGIBLE_AND_WINDOW"]
    & encounters["DATE_IN_RECOMPUTED_WINDOW"]
    & encounters["ENCOUNTER_DATE"].notna()
    & encounters["ENCOUNTER_EGFR"].notna()
].copy()
source_flagged_outside_date_window = int(
    (
        encounters["SOURCE_ELIGIBLE_AND_WINDOW"]
        & encounters["ENCOUNTER_EGFR"].notna()
        & ~encounters["DATE_IN_RECOMPUTED_WINDOW"]
    ).sum()
)
source_flagged_post_index = int(
    (
        encounters["SOURCE_ELIGIBLE_AND_WINDOW"]
        & encounters["ENCOUNTER_EGFR"].notna()
        & encounters["DAYS_FROM_INDEX"].gt(0)
    ).sum()
)
if source_flagged_outside_date_window or source_flagged_post_index:
    raise ValueError(
        "Source eligibility/window flags include dated eGFR outside the "
        "patient-specific pre-index window."
    )

# Same-day duplicates are collapsed to patient-day medians, matching the
# original longitudinal feature construction.
daily = (
    usable.groupby(["_PATIENT_KEY", "ENCOUNTER_DATE"], as_index=False)[
        "ENCOUNTER_EGFR"
    ]
    .median()
    .sort_values(["_PATIENT_KEY", "ENCOUNTER_DATE"], kind="stable")
    .reset_index(drop=True)
)
latest = (
    daily.groupby("_PATIENT_KEY", as_index=False)
    .tail(1)[["_PATIENT_KEY", "ENCOUNTER_DATE", "ENCOUNTER_EGFR"]]
    .rename(
        columns={
            "ENCOUNTER_DATE": "REPLAY_LATEST_EGFR_DATE",
            "ENCOUNTER_EGFR": "REPLAY_CURRENT_EGFR",
        }
    )
)
summary = (
    daily.groupby("_PATIENT_KEY")["ENCOUNTER_EGFR"]
    .agg(
        REPLAY_MIN_EGFR_24M="min",
        REPLAY_MEAN_EGFR_24M="mean",
        REPLAY_DISTINCT_EGFR_DATES="size",
    )
    .reset_index()
    .merge(latest, on="_PATIENT_KEY", how="left", validate="one_to_one")
)

# Previous qualifying value: most recent distinct-date value >=90 days before
# the latest replayed eGFR.
daily_previous = daily.merge(
    latest[["_PATIENT_KEY", "REPLAY_LATEST_EGFR_DATE"]],
    on="_PATIENT_KEY",
    how="left",
    validate="many_to_one",
)
daily_previous["GAP_TO_LATEST_DAYS"] = (
    daily_previous["REPLAY_LATEST_EGFR_DATE"]
    - daily_previous["ENCOUNTER_DATE"]
).dt.total_seconds() / 86400.0
previous = (
    daily_previous.loc[daily_previous["GAP_TO_LATEST_DAYS"].ge(MINIMUM_GAP_DAYS)]
    .groupby("_PATIENT_KEY", as_index=False)
    .tail(1)[
        [
            "_PATIENT_KEY",
            "ENCOUNTER_DATE",
            "ENCOUNTER_EGFR",
            "GAP_TO_LATEST_DAYS",
        ]
    ]
    .rename(
        columns={
            "ENCOUNTER_DATE": "REPLAY_PREVIOUS_EGFR_DATE",
            "ENCOUNTER_EGFR": "REPLAY_PREVIOUS_EGFR_90D",
            "GAP_TO_LATEST_DAYS": "REPLAY_EGFR_GAP_DAYS",
        }
    )
)
summary = summary.merge(previous, on="_PATIENT_KEY", how="left", validate="one_to_one")
summary["REPLAY_PERSISTENT_LOW_EGFR_90D"] = (
    summary["REPLAY_CURRENT_EGFR"].lt(60)
    & summary["REPLAY_PREVIOUS_EGFR_90D"].lt(60)
    & summary["REPLAY_EGFR_GAP_DAYS"].ge(MINIMUM_GAP_DAYS)
)

frozen = pd.read_parquet(
    TEST_PREDICTORS,
    columns=[ID_COL, CURRENT_EGFR, MIN_EGFR, MEAN_EGFR],
    filters=[(ID_COL, "in", list(raw_ids))],
)
frozen["_PATIENT_KEY"] = patient_key(frozen[ID_COL])
for column in [CURRENT_EGFR, MIN_EGFR, MEAN_EGFR]:
    frozen[column] = pd.to_numeric(frozen[column], errors="coerce")

audit = (
    alpha_cases[[ID_COL, "_PATIENT_KEY", "alpha_review_rank", SCORE_COL]]
    .merge(
        patients[["_PATIENT_KEY", "PATIENT_INDEX_DATE"]],
        on="_PATIENT_KEY",
        how="left",
        validate="one_to_one",
    )
    .merge(
        frozen[["_PATIENT_KEY", CURRENT_EGFR, MIN_EGFR, MEAN_EGFR]],
        on="_PATIENT_KEY",
        how="left",
        validate="one_to_one",
    )
    .merge(summary, on="_PATIENT_KEY", how="left", validate="one_to_one")
)
if len(audit) != EXPECTED_ALPHA_CASES:
    raise ValueError("Alpha-review audit row count changed.")

comparisons = [
    (CURRENT_EGFR, "REPLAY_CURRENT_EGFR", "Current eGFR"),
    (MIN_EGFR, "REPLAY_MIN_EGFR_24M", "Minimum eGFR in 24 months"),
    (MEAN_EGFR, "REPLAY_MEAN_EGFR_24M", "Mean eGFR in 24 months"),
]
replay_rows = []
total_mismatches = 0
for frozen_col, replay_col, label in comparisons:
    frozen_values = audit[frozen_col]
    replay_values = audit[replay_col]
    missingness_mismatch = frozen_values.isna() ^ replay_values.isna()
    value_mismatch = (
        frozen_values.notna()
        & replay_values.notna()
        & ~np.isclose(frozen_values, replay_values, rtol=1e-8, atol=1e-8)
    )
    mismatch = missingness_mismatch | value_mismatch
    count = int(mismatch.sum())
    total_mismatches += count
    replay_rows.append(
        {
            "Feature": label,
            "Frozen_observed": int(frozen_values.notna().sum()),
            "Replayed_observed": int(replay_values.notna().sum()),
            "Missingness_mismatches": int(missingness_mismatch.sum()),
            "Value_mismatches": int(value_mismatch.sum()),
            "Total_mismatches": count,
            "Maximum_absolute_difference": (
                float(np.nanmax(np.abs(frozen_values - replay_values)))
                if (frozen_values.notna() & replay_values.notna()).any()
                else np.nan
            ),
        }
    )
replay_table = pd.DataFrame(replay_rows)

audit["audit_case_code"] = [
    f"N13A-ALPHA-{index:03d}" for index in range(1, len(audit) + 1)
]
audit["LATEST_EGFR_ON_OR_BEFORE_INDEX"] = (
    audit["REPLAY_LATEST_EGFR_DATE"] <= audit["PATIENT_INDEX_DATE"]
)
audit["LATEST_EGFR_WITHIN_730_DAYS"] = (
    audit["REPLAY_LATEST_EGFR_DATE"]
    >= audit["PATIENT_INDEX_DATE"] - pd.Timedelta(days=WINDOW_DAYS)
)

summary_payload = {
    "alpha_review_patients": len(audit),
    "patients_with_replayed_egfr": int(audit["REPLAY_CURRENT_EGFR"].notna().sum()),
    "source_flagged_egfr_rows_outside_730d_preindex_window": (
        source_flagged_outside_date_window
    ),
    "source_flagged_postindex_egfr_rows": source_flagged_post_index,
    "latest_egfr_on_or_before_index": int(
        audit["LATEST_EGFR_ON_OR_BEFORE_INDEX"].sum()
    ),
    "latest_egfr_within_730_days": int(audit["LATEST_EGFR_WITHIN_730_DAYS"].sum()),
    "frozen_vs_replayed_feature_mismatches": total_mismatches,
    "patients_with_previous_egfr_at_least_90d_earlier": int(
        audit["REPLAY_PREVIOUS_EGFR_90D"].notna().sum()
    ),
    "persistent_low_egfr_at_least_90d_apart": int(
        audit["REPLAY_PERSISTENT_LOW_EGFR_90D"].sum()
    ),
    "uacr_loaded": False,
}

hard_failures = 0
hard_failures += source_flagged_outside_date_window
hard_failures += source_flagged_post_index
hard_failures += total_mismatches
hard_failures += int(audit["REPLAY_CURRENT_EGFR"].isna().any())
hard_failures += int((~audit["LATEST_EGFR_ON_OR_BEFORE_INDEX"]).any())
hard_failures += int((~audit["LATEST_EGFR_WITHIN_730_DAYS"]).any())
summary_payload["hard_integrity_failures"] = hard_failures

context = {
    "step": "5.12N13A",
    "status": "ALPHA_REVIEW_TEMPORAL_EGFR_REPLAY",
    "n11_completion_sha256": sha256_file(N11_COMPLETION),
    "n12_completion_sha256": sha256_file(N12_COMPLETION),
    "n13_completion_sha256": sha256_file(N13_COMPLETION),
    "patient_checkpoint_sha256": sha256_file(PATIENT_FILE),
    "encounter_checkpoint_sha256": sha256_file(ENCOUNTER_FILE),
    "test_predictors_sha256": sha256_file(TEST_PREDICTORS),
    "window_days": WINDOW_DAYS,
    "minimum_chronicity_gap_days": MINIMUM_GAP_DAYS,
    "same_day_aggregation": "median",
    "uacr_loaded_or_reported": False,
    "model_or_scores_changed": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

atomic_json(OUTPUT_DIR / "context.json", context)
atomic_json(OUTPUT_DIR / "temporal_egfr_replay_summary.json", summary_payload)
atomic_csv(replay_table, OUTPUT_DIR / "frozen_vs_replayed_egfr_audit.csv")
atomic_csv(
    audit[
        [
            "audit_case_code",
            "alpha_review_rank",
            SCORE_COL,
            "PATIENT_INDEX_DATE",
            "REPLAY_LATEST_EGFR_DATE",
            CURRENT_EGFR,
            "REPLAY_CURRENT_EGFR",
            MIN_EGFR,
            "REPLAY_MIN_EGFR_24M",
            MEAN_EGFR,
            "REPLAY_MEAN_EGFR_24M",
            "REPLAY_DISTINCT_EGFR_DATES",
            "REPLAY_PREVIOUS_EGFR_DATE",
            "REPLAY_PREVIOUS_EGFR_90D",
            "REPLAY_EGFR_GAP_DAYS",
            "REPLAY_PERSISTENT_LOW_EGFR_90D",
        ]
    ],
    PRIVATE_DIR / "deidentified_alpha_temporal_egfr_replay.csv",
)
atomic_parquet(
    audit,
    PRIVATE_DIR / "private_alpha_temporal_egfr_replay.parquet",
)

completion = {
    **context,
    "status": "ALPHA_REVIEW_TEMPORAL_EGFR_REPLAY_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(OUTPUT_DIR),
    "summary": summary_payload,
    "hard_integrity_failures": hard_failures,
    "model_changed": False,
    "next_step": (
        "If hard_integrity_failures equals zero, freeze the two-phase blinded "
        "clinician-review sampling design without changing the model."
    ),
}
atomic_json(OUTPUT_DIR / "completion.json", completion)

print("=" * 116)
print("STEP 5.12N13A — ALPHA-REVIEW TEMPORAL eGFR REPLAY COMPLETE")
print("=" * 116)
print("\nTEMPORAL REPLAY SUMMARY")
print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
print("\nFROZEN VERSUS REPLAYED eGFR FEATURES")
display(replay_table)
print(f"\nSaved run: {OUTPUT_DIR}")
print("uACR loaded or reported: No")
print("Model or frozen scores changed: No")
if hard_failures:
    raise RuntimeError(
        f"Temporal eGFR replay found {hard_failures} hard integrity failure(s). "
        "Do not proceed to clinician sampling; send the complete output."
    )
print("Hard integrity failures: 0")
print("Temporal eGFR replay passed: Yes")
