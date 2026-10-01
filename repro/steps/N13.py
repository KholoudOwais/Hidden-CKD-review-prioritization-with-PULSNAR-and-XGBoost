"""STEP 5.12N13 — Alpha-review integrity audit and eGFR benchmark.

Post-Test descriptive audit only.  Reconstructs the locked N12 PULSCAR-alpha
review set from the frozen N11 scores and observed labels, audits renal values
without loading uACR, and compares the locked model with reference-defined
simple eGFR-only clinical benchmarks.

Nothing in this step may select, retrain, recalibrate, replace, or modify the
locked model.  The clinical benchmarks are explicitly post hoc and descriptive.
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
from sklearn.metrics import average_precision_score, roc_auc_score


EXPECTED_TEST_PATIENTS = 14_927
EXPECTED_TEST_DOCUMENTED_CKD = 1_071
EXPECTED_TEST_UNLABELED = 13_856
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 51_213

ID_COL = "patient_id"
LABEL_COL = "OBSERVED_CKD_LABEL"
SCORE_COL = "locked_pulscar_mean_xgb_internal_test_score"
PERCENT_COL = "model_output_percentage"
PERCENTILE_COL = "within_cohort_percentile"

CURRENT_EGFR = "DERIVED_EGFR_CKD_EPI_2021"
MIN_EGFR = "EGFR_MIN_24M"
MEAN_EGFR = "EGFR_MEAN_24M"
PREVIOUS_EGFR = "EGFR_PREVIOUS_90D"
EGFR_GAP_DAYS = "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS"

ROOT = Path(
    "/content/drive/MyDrive/CKD_Project/outputs/"
    "section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
N6_ROOT = ROOT / "05_12N6_full_training_pulscar_mean_xgb_final_refit_and_lock"
N11_ROOT = ROOT / (
    "05_12N11_outcome_blind_locked_pulscar_mean_xgb_internal_test_scoring"
)
N12_ROOT = ROOT / "05_12N12_locked_internal_test_evaluation"
OUTPUT_ROOT = ROOT / (
    "05_12N13_alpha_review_integrity_and_egfr_benchmark"
)
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


def egfr_band(series):
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [
                values.lt(15),
                values.ge(15) & values.lt(30),
                values.ge(30) & values.lt(45),
                values.ge(45) & values.lt(60),
                values.ge(60) & values.lt(90),
                values.ge(90),
            ],
            [
                "G5 (<15)",
                "G4 (15-29)",
                "G3b (30-44)",
                "G3a (45-59)",
                "G2 (60-89)",
                "G1 (>=90)",
            ],
            default="Not available",
        ),
        index=series.index,
    )


def lowest_priority_missing(values):
    """Return -eGFR with missing records ranked below every observed record."""
    values = pd.to_numeric(values, errors="coerce").astype(float)
    score = -values
    observed = score[np.isfinite(score)]
    if observed.empty:
        raise ValueError("An eGFR comparator has no observed values.")
    score = score.fillna(float(observed.min()) - 1.0)
    return score.to_numpy(float)


def continuous_metric_row(name, labels, comparator_score, observed_n):
    return {
        "Method": name,
        "Average_precision": float(average_precision_score(labels, comparator_score)),
        "ROC_AUC": float(roc_auc_score(labels, comparator_score)),
        "Observed_eGFR_n": int(observed_n),
        "Missing_handling": (
            "Missing eGFR assigned lowest review priority; no imputation used "
            "for this simple descriptive comparator."
        ),
    }


def threshold_row(name, flags, labels):
    flags = np.asarray(flags, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    tp = int(np.sum(flags & (labels == 1)))
    fp = int(np.sum(flags & (labels == 0)))
    tn = int(np.sum((~flags) & (labels == 0)))
    fn = int(np.sum((~flags) & (labels == 1)))
    return {
        "Clinical_rule": name,
        "Patients_flagged": int(flags.sum()),
        "Workload_percent": 100.0 * float(flags.mean()),
        "Documented_CKD_captured": tp,
        "Sensitivity_percent": 100.0 * tp / (tp + fn),
        "Specificity_percent": 100.0 * tn / (tn + fp),
        "Positive_predictive_value_percent": (
            100.0 * tp / (tp + fp) if (tp + fp) else np.nan
        ),
        "Negative_predictive_value_percent": (
            100.0 * tn / (tn + fn) if (tn + fn) else np.nan
        ),
        "Unlabeled_flagged": fp,
    }


N6_COMPLETION, n6 = latest_completed(
    N6_ROOT, "FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED"
)
N11_COMPLETION, n11 = latest_completed(
    N11_ROOT, "OUTCOME_BLIND_LOCKED_INTERNAL_TEST_SCORING_COMPLETE"
)
N12_COMPLETION, n12 = latest_completed(
    N12_ROOT, "LOCKED_INTERNAL_TEST_EVALUATION_COMPLETE"
)

SCORE_FILE = Path(n11["score_file"])
TEST_PREDICTORS = Path(n11["internal_test_predictors"])

for required in [
    N6_COMPLETION,
    N11_COMPLETION,
    N12_COMPLETION,
    SCORE_FILE,
    TEST_PREDICTORS,
    OUTCOME_SOURCE,
]:
    if not required.is_file():
        raise FileNotFoundError(required)

if n6.get("uacr_used_anywhere_in_modeling") is not False:
    raise ValueError("N6 reports uACR use.")
if n11.get("priority_categories_created") is not False:
    raise ValueError("N11 created forbidden clinical categories.")
if n12.get("model_changed") is not False:
    raise ValueError("N12 reports a model change.")
if n12.get("high_moderate_low_categories_created") is not False:
    raise ValueError("N12 created forbidden clinical categories.")
if n12.get("uacr_loaded_or_reported") is not False:
    raise ValueError("N12 reports uACR use.")
if sha256_file(SCORE_FILE) != n11["score_file_sha256"]:
    raise ValueError("The frozen N11 score file changed.")
if sha256_file(TEST_PREDICTORS) != n11["internal_test_predictors_sha256"]:
    raise ValueError("The authoritative Internal Test predictor table changed.")
if sha256_file(OUTCOME_SOURCE) != n12["outcome_source_sha256"]:
    raise ValueError("The N12 outcome source changed.")

scores = pd.read_parquet(SCORE_FILE)
if list(scores.columns) != [ID_COL, SCORE_COL, PERCENT_COL, PERCENTILE_COL]:
    raise ValueError(f"Unexpected frozen score schema: {list(scores.columns)}")
if len(scores) != EXPECTED_TEST_PATIENTS:
    raise ValueError("Unexpected frozen score count.")
scores["_PATIENT_KEY"] = patient_key(scores[ID_COL])
if scores["_PATIENT_KEY"].isna().any() or scores["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Frozen-score patient identifiers are invalid.")

outcome_schema = pq.ParquetFile(OUTCOME_SOURCE).schema_arrow.names
outcome_id_col = resolve_name(
    outcome_schema,
    ["patient_id", "PATIENT_ID"],
    [("patient",), ("id",)],
    "patient ID",
)
outcome_label_col = resolve_name(
    outcome_schema,
    ["observed_label", "CKD_CLASS", "documented_ckd", "OBSERVED_CKD_LABEL"],
    [("ckd",), ("class", "label")],
    "documented CKD label",
)

optional_outcome_fields = [PREVIOUS_EGFR, EGFR_GAP_DAYS]
available_outcome_fields = [
    field for field in optional_outcome_fields if field in outcome_schema
]
outcomes = pd.read_parquet(
    OUTCOME_SOURCE,
    columns=[outcome_id_col, outcome_label_col, *available_outcome_fields],
)
outcomes["_PATIENT_KEY"] = patient_key(outcomes[outcome_id_col])
if outcomes["_PATIENT_KEY"].isna().any() or outcomes["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Outcome-source patient identifiers are invalid.")
outcomes[LABEL_COL] = pd.to_numeric(
    outcomes[outcome_label_col], errors="raise"
).astype(int)
if not outcomes[LABEL_COL].isin([0, 1]).all():
    raise ValueError("Observed Test labels are not binary.")

predictor_schema = pq.ParquetFile(TEST_PREDICTORS).schema_arrow.names
required_predictors = [ID_COL, CURRENT_EGFR, MIN_EGFR, MEAN_EGFR]
missing_predictors = sorted(set(required_predictors) - set(predictor_schema))
if missing_predictors:
    raise KeyError(f"Authoritative Test predictors lack: {missing_predictors}")
predictors = pd.read_parquet(TEST_PREDICTORS, columns=required_predictors)
predictors["_PATIENT_KEY"] = patient_key(predictors[ID_COL])
if predictors["_PATIENT_KEY"].isna().any() or predictors["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Predictor-source patient identifiers are invalid.")

evaluation = scores.merge(
    outcomes[["_PATIENT_KEY", LABEL_COL, *available_outcome_fields]],
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
).merge(
    predictors[["_PATIENT_KEY", CURRENT_EGFR, MIN_EGFR, MEAN_EGFR]],
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
if len(evaluation) != EXPECTED_TEST_PATIENTS:
    raise ValueError("Evaluation row count changed during linkage.")
if evaluation[LABEL_COL].isna().any():
    raise ValueError("Some frozen-score patients lack an observed Test label.")

for column in [CURRENT_EGFR, MIN_EGFR, MEAN_EGFR, *available_outcome_fields]:
    evaluation[column] = pd.to_numeric(evaluation[column], errors="coerce")

labels = evaluation[LABEL_COL].to_numpy(int)
documented_ckd = int(labels.sum())
unlabeled_n = int((labels == 0).sum())
if documented_ckd != EXPECTED_TEST_DOCUMENTED_CKD:
    raise ValueError(f"Unexpected documented-CKD count: {documented_ckd}")
if unlabeled_n != EXPECTED_TEST_UNLABELED:
    raise ValueError(f"Unexpected unlabeled count: {unlabeled_n}")

# Reconstruct the exact N12 alpha-matched set from frozen scores.
alpha = float(n6["estimated_alpha"])
alpha_review_n = int(math.floor(alpha * unlabeled_n))
unlabeled = evaluation.loc[evaluation[LABEL_COL].eq(0)].copy()
unlabeled["_TIE_HASH"] = unlabeled["_PATIENT_KEY"].map(
    lambda value: hashlib.sha256(
        f"N12|alpha-review|{value}".encode("utf-8")
    ).hexdigest()
)
unlabeled = unlabeled.sort_values(
    [SCORE_COL, "_TIE_HASH"], ascending=[False, True], kind="stable"
).reset_index(drop=True)
alpha_review = unlabeled.head(alpha_review_n).copy()
alpha_review["alpha_review_rank"] = np.arange(1, alpha_review_n + 1, dtype=int)

n12_alpha = n12["alpha_review_summary"]
if alpha_review_n != int(n12_alpha["selected_for_secondary_review"]):
    raise ValueError("Reconstructed alpha-review count differs from N12.")
if not np.isclose(
    float(alpha_review[SCORE_COL].min()),
    float(n12_alpha["minimum_selected_score"]),
    rtol=0,
    atol=1e-12,
):
    raise ValueError("Reconstructed alpha-review score boundary differs from N12.")

alpha_review["CURRENT_EGFR_BAND"] = egfr_band(alpha_review[CURRENT_EGFR])
alpha_review["CURRENT_EGFR_VALUE_PLAUSIBLE"] = (
    alpha_review[CURRENT_EGFR].notna()
    & alpha_review[CURRENT_EGFR].between(1, 200, inclusive="both")
)
alpha_review["CURRENT_EGFR_LT60"] = alpha_review[CURRENT_EGFR].lt(60)
alpha_review["CURRENT_EGFR_LT30"] = alpha_review[CURRENT_EGFR].lt(30)
alpha_review["CURRENT_EGFR_LT15"] = alpha_review[CURRENT_EGFR].lt(15)

if PREVIOUS_EGFR in available_outcome_fields and EGFR_GAP_DAYS in available_outcome_fields:
    alpha_review["PERSISTENT_LOW_EGFR_90D"] = (
        alpha_review[CURRENT_EGFR].lt(60)
        & alpha_review[PREVIOUS_EGFR].lt(60)
        & alpha_review[EGFR_GAP_DAYS].ge(90)
    )
    persistence_available = True
else:
    alpha_review["PERSISTENT_LOW_EGFR_90D"] = False
    persistence_available = False

band_order = [
    "G1 (>=90)",
    "G2 (60-89)",
    "G3a (45-59)",
    "G3b (30-44)",
    "G4 (15-29)",
    "G5 (<15)",
    "Not available",
]
band_table = (
    alpha_review["CURRENT_EGFR_BAND"]
    .value_counts()
    .reindex(band_order, fill_value=0)
    .rename_axis("Current_eGFR_band")
    .reset_index(name="Patients")
)
band_table["Percent_of_alpha_review_set"] = (
    100.0 * band_table["Patients"] / alpha_review_n
)

quantile_rows = []
for column, label in [
    (CURRENT_EGFR, "Current derived eGFR"),
    (MIN_EGFR, "Minimum eGFR within 24 months"),
    (MEAN_EGFR, "Mean eGFR within 24 months"),
]:
    values = alpha_review[column].dropna()
    quantile_rows.append(
        {
            "Measure": label,
            "Observed_n": int(values.size),
            "Minimum": float(values.min()) if len(values) else np.nan,
            "P25": float(values.quantile(0.25)) if len(values) else np.nan,
            "Median": float(values.median()) if len(values) else np.nan,
            "P75": float(values.quantile(0.75)) if len(values) else np.nan,
            "Maximum": float(values.max()) if len(values) else np.nan,
        }
    )
alpha_quantiles = pd.DataFrame(quantile_rows)

date_like_columns = [
    name
    for name in predictor_schema
    if "date" in str(name).lower() or "timestamp" in str(name).lower()
]
temporal_audit = {
    "date_like_columns_in_frozen_predictor_table": date_like_columns,
    "feature_timestamp_audit_computable": bool(date_like_columns),
    "interpretation": (
        "Feature-date ordering requires source timestamps; absence of date-like "
        "columns in the frozen predictor snapshot means temporal ordering must "
        "be verified from the upstream cohort-construction audit, not inferred here."
    ),
}

integrity_summary = {
    "alpha_review_patients": alpha_review_n,
    "unique_patient_keys": int(alpha_review["_PATIENT_KEY"].nunique()),
    "current_egfr_observed": int(alpha_review[CURRENT_EGFR].notna().sum()),
    "current_egfr_values_plausible_1_to_200": int(
        alpha_review["CURRENT_EGFR_VALUE_PLAUSIBLE"].sum()
    ),
    "current_egfr_below_60": int(alpha_review["CURRENT_EGFR_LT60"].sum()),
    "current_egfr_below_30": int(alpha_review["CURRENT_EGFR_LT30"].sum()),
    "current_egfr_below_15": int(alpha_review["CURRENT_EGFR_LT15"].sum()),
    "previous_egfr_and_gap_fields_available": persistence_available,
    "persistent_low_egfr_90d": int(
        alpha_review["PERSISTENT_LOW_EGFR_90D"].sum()
    ),
    "uacr_loaded": False,
    "clinical_interpretation": (
        "A single low eGFR is renal evidence but does not by itself establish "
        "chronicity or adjudicate CKD; G categories are descriptive only."
    ),
}

# Continuous, eGFR-only post-hoc clinical comparators.
model_scores = evaluation[SCORE_COL].to_numpy(float)
current_score = lowest_priority_missing(evaluation[CURRENT_EGFR])
minimum_score = lowest_priority_missing(evaluation[MIN_EGFR])
mean_score = lowest_priority_missing(evaluation[MEAN_EGFR])

continuous_table = pd.DataFrame(
    [
        {
            "Method": "Locked PULSCAR + mean-imputed XGBoost",
            "Average_precision": float(average_precision_score(labels, model_scores)),
            "ROC_AUC": float(roc_auc_score(labels, model_scores)),
            "Observed_eGFR_n": np.nan,
            "Missing_handling": "Locked N6 mean-imputation pipeline.",
        },
        continuous_metric_row(
            "Current eGFR alone",
            labels,
            current_score,
            evaluation[CURRENT_EGFR].notna().sum(),
        ),
        continuous_metric_row(
            "Minimum eGFR within 24 months alone",
            labels,
            minimum_score,
            evaluation[MIN_EGFR].notna().sum(),
        ),
        continuous_metric_row(
            "Mean eGFR within 24 months alone",
            labels,
            mean_score,
            evaluation[MEAN_EGFR].notna().sum(),
        ),
    ]
)

rule_table = pd.DataFrame(
    [
        threshold_row(
            "Observed current eGFR <60",
            evaluation[CURRENT_EGFR].notna() & evaluation[CURRENT_EGFR].lt(60),
            labels,
        ),
        threshold_row(
            "Observed current eGFR <30",
            evaluation[CURRENT_EGFR].notna() & evaluation[CURRENT_EGFR].lt(30),
            labels,
        ),
        threshold_row(
            "Observed current eGFR <15",
            evaluation[CURRENT_EGFR].notna() & evaluation[CURRENT_EGFR].lt(15),
            labels,
        ),
    ]
)

# Paired stratified bootstrap: comparator AP minus locked-model AP.
positive_positions = np.flatnonzero(labels == 1)
unlabeled_positions = np.flatnonzero(labels == 0)
rng = np.random.default_rng(BOOTSTRAP_SEED)
comparator_scores = {
    "Current eGFR alone": current_score,
    "Minimum eGFR within 24 months alone": minimum_score,
    "Mean eGFR within 24 months alone": mean_score,
}
bootstrap_rows = []
for replicate in range(1, BOOTSTRAP_REPLICATES + 1):
    sampled = np.concatenate(
        [
            rng.choice(positive_positions, len(positive_positions), replace=True),
            rng.choice(unlabeled_positions, len(unlabeled_positions), replace=True),
        ]
    )
    model_ap = average_precision_score(labels[sampled], model_scores[sampled])
    for name, values in comparator_scores.items():
        comparator_ap = average_precision_score(labels[sampled], values[sampled])
        bootstrap_rows.append(
            {
                "Replicate": replicate,
                "Comparator": name,
                "Comparator_AP_minus_locked_model_AP": float(
                    comparator_ap - model_ap
                ),
            }
        )
    if replicate % 100 == 0:
        print(f"Bootstrap {replicate}/{BOOTSTRAP_REPLICATES}")
bootstrap = pd.DataFrame(bootstrap_rows)

delta_rows = []
locked_ap = float(continuous_table.iloc[0]["Average_precision"])
for name in comparator_scores:
    comparator_ap = float(
        continuous_table.loc[
            continuous_table["Method"].eq(name), "Average_precision"
        ].iloc[0]
    )
    values = bootstrap.loc[
        bootstrap["Comparator"].eq(name),
        "Comparator_AP_minus_locked_model_AP",
    ]
    delta_rows.append(
        {
            "Comparator": name,
            "Reference": "Locked PULSCAR + mean-imputed XGBoost",
            "Comparator_AP_minus_locked_model_AP": comparator_ap - locked_ap,
            "Bootstrap_95CI_lower": float(values.quantile(0.025)),
            "Bootstrap_95CI_upper": float(values.quantile(0.975)),
            "Replicates": BOOTSTRAP_REPLICATES,
        }
    )
delta_table = pd.DataFrame(delta_rows)

# Deidentify the private audit display while preserving the separate patient-key
# linkage only inside the private artifact.
alpha_review = alpha_review.reset_index(drop=True)
alpha_review["audit_case_code"] = [
    f"N13-ALPHA-{index:03d}" for index in range(1, len(alpha_review) + 1)
]
public_case_columns = [
    "audit_case_code",
    "alpha_review_rank",
    SCORE_COL,
    PERCENT_COL,
    PERCENTILE_COL,
    CURRENT_EGFR,
    "CURRENT_EGFR_BAND",
    MIN_EGFR,
    MEAN_EGFR,
    "CURRENT_EGFR_VALUE_PLAUSIBLE",
    "PERSISTENT_LOW_EGFR_90D",
]
if PREVIOUS_EGFR in available_outcome_fields:
    public_case_columns.append(PREVIOUS_EGFR)
if EGFR_GAP_DAYS in available_outcome_fields:
    public_case_columns.append(EGFR_GAP_DAYS)
deidentified_cases = alpha_review[public_case_columns].copy()

context = {
    "step": "5.12N13",
    "status": "ALPHA_REVIEW_INTEGRITY_AND_EGFR_BENCHMARK",
    "n6_completion_sha256": sha256_file(N6_COMPLETION),
    "n11_completion_sha256": sha256_file(N11_COMPLETION),
    "n12_completion_sha256": sha256_file(N12_COMPLETION),
    "frozen_score_file_sha256": sha256_file(SCORE_FILE),
    "outcome_source_sha256": sha256_file(OUTCOME_SOURCE),
    "test_predictor_source_sha256": sha256_file(TEST_PREDICTORS),
    "analysis_role": (
        "Post-Test descriptive integrity audit and post-hoc clinical benchmark; "
        "not model selection or threshold optimization."
    ),
    "alpha_review_count": alpha_review_n,
    "uacr_loaded_or_reported": False,
    "model_or_preprocessing_changed": False,
    "test_scores_or_labels_changed": False,
    "clinical_categories_created": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

atomic_json(OUTPUT_DIR / "context.json", context)
atomic_json(OUTPUT_DIR / "alpha_review_integrity_summary.json", integrity_summary)
atomic_json(OUTPUT_DIR / "temporal_audit_status.json", temporal_audit)
atomic_csv(band_table, OUTPUT_DIR / "alpha_review_current_egfr_bands.csv")
atomic_csv(alpha_quantiles, OUTPUT_DIR / "alpha_review_egfr_quantiles.csv")
atomic_csv(
    continuous_table,
    OUTPUT_DIR / "posthoc_continuous_egfr_clinical_benchmark.csv",
)
atomic_csv(rule_table, OUTPUT_DIR / "posthoc_current_egfr_rule_benchmark.csv")
atomic_csv(delta_table, OUTPUT_DIR / "paired_bootstrap_egfr_vs_model_ap.csv")
atomic_csv(
    deidentified_cases,
    PRIVATE_DIR / "alpha_review_deidentified_renal_audit.csv",
)
atomic_parquet(
    alpha_review[
        [
            ID_COL,
            "_PATIENT_KEY",
            "audit_case_code",
            *[column for column in public_case_columns if column != "audit_case_code"],
        ]
    ],
    PRIVATE_DIR / "alpha_review_private_renal_audit.parquet",
)
atomic_parquet(
    bootstrap,
    PRIVATE_DIR / "paired_bootstrap_replicates.parquet",
)

completion = {
    **context,
    "status": "ALPHA_REVIEW_INTEGRITY_AND_EGFR_BENCHMARK_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(OUTPUT_DIR),
    "integrity_summary": integrity_summary,
    "temporal_audit": temporal_audit,
    "posthoc_benchmark_interpretation": (
        "The eGFR-only comparators quantify incremental ranking performance "
        "relative to simple renal-function signals. They were evaluated after "
        "Test outcome opening and cannot alter the locked primary model."
    ),
    "model_changed": False,
    "next_step": (
        "Use the locked scores and this integrity audit to freeze a blinded, "
        "two-phase clinician-review sample; do not alter the model or Test results."
    ),
}
atomic_json(OUTPUT_DIR / "completion.json", completion)

print("=" * 116)
print("STEP 5.12N13 — ALPHA-REVIEW INTEGRITY AND eGFR BENCHMARK COMPLETE")
print("=" * 116)
print("\nALPHA-REVIEW INTEGRITY SUMMARY")
print(json.dumps(integrity_summary, indent=2, ensure_ascii=False))
print("\nCURRENT eGFR BANDS IN THE 56-CASE ALPHA REVIEW SET")
display(band_table)
print("\neGFR DISTRIBUTION IN THE ALPHA REVIEW SET")
display(alpha_quantiles)
print("\nPOST-HOC CONTINUOUS CLINICAL BENCHMARK")
display(continuous_table)
print("\nPOST-HOC REFERENCE-DEFINED eGFR RULES")
display(rule_table)
print("\nPAIRED BOOTSTRAP AP DIFFERENCE VERSUS LOCKED MODEL")
display(delta_table)
print("\nTEMPORAL AUDIT STATUS")
print(json.dumps(temporal_audit, indent=2, ensure_ascii=False))
print(f"\nSaved run: {OUTPUT_DIR}")
print("uACR loaded or reported: No")
print("Model, preprocessing, scores, or labels changed: No")
print("Clinical categories created: No")
print("Benchmark role: Post-hoc descriptive clinical comparator only")
