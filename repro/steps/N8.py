# STEP 5.12N8 — LOCKED VALIDATION CHARACTERIZATION AND WORKLOAD LOCK
#
# This is the intentional locked characterization of Validation outcomes after
# the Step 5.12N7 score vector was frozen. Validation had previously been opened
# for the superseded Step 5.12H candidate in Step 5.12K; it was not used to
# select or tune the present final pipeline. This step characterizes the immutable model,
# reports patient-bootstrap uncertainty, and locks rank-based review workloads.
# It does not train, tune, recalibrate, or access Test.
#
# Locked review-priority rule (independent of Validation outcomes):
#   High     = highest 10% of scores
#   Moderate = next 40% of scores
#   Low      = remaining 50% of scores
# The score is a review-priority ranking, not a calibrated CKD probability.

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


EXPECTED_VALIDATION_PATIENTS = 14_923
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 51_290
ID_COL = "patient_id"
SCORE_COL = "locked_pulscar_mean_xgb_validation_score"
LABEL_COL = "OBSERVED_CKD_LABEL"
WORKLOAD_FRACTIONS = [0.02, 0.05, 0.10]
HIGH_FRACTION = 0.10
MODERATE_PLUS_HIGH_FRACTION = 0.50

ROOT = Path(
    "/content/drive/MyDrive/CKD_Project/outputs/"
    "section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
if not ROOT.exists():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass
if not ROOT.exists():
    raise FileNotFoundError(f"Modeling root was not found: {ROOT}")

N7_ROOT = ROOT / "05_12N7_outcome_blind_locked_pulscar_mean_xgb_validation_scoring"
n7_candidates = []
for candidate in N7_ROOT.rglob("completion.json"):
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("status") == "OUTCOME_BLIND_LOCKED_PULSCAR_MEAN_XGB_VALIDATION_SCORING_COMPLETE":
        n7_candidates.append((candidate, payload))
if not n7_candidates:
    raise FileNotFoundError("Run Step 5.12N7 before N8.")
M8_COMPLETION, m8 = max(n7_candidates, key=lambda item: item[0].stat().st_mtime)
M8_RUN = M8_COMPLETION.parent
SCORE_FILE = Path(m8["score_file"])
VALIDATION_SOURCE = Path(
    "/content/drive/MyDrive/CKD_Project/outputs/section_5_modeling/"
    "step_5_8_longitudinal_experiment/"
    "03b_longitudinal_egfr_development_data_90d.parquet"
)


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


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


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
    return series.astype("string").str.strip()


def metric_values(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.clip(np.asarray(scores, dtype=float), 1e-15, 1 - 1e-15)
    return {
        "Average_precision": float(average_precision_score(labels, scores)),
        "ROC_AUC": float(roc_auc_score(labels, scores)),
        "Brier_score": float(brier_score_loss(labels, scores)),
        "Log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "Observed_prevalence": float(np.mean(labels)),
        "Mean_score": float(np.mean(scores)),
    }


def stable_rank(frame):
    ranked = frame.copy()
    ranked["_TIE_HASH"] = ranked["_PATIENT_KEY"].map(
        lambda value: hashlib.sha256(
            f"N8|locked-rank|{value}".encode("utf-8")
        ).hexdigest()
    )
    return ranked.sort_values(
        [SCORE_COL, "_TIE_HASH"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


for path in [M8_COMPLETION, SCORE_FILE, VALIDATION_SOURCE]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact was not found: {path}")

if m8.get("status") != "OUTCOME_BLIND_LOCKED_PULSCAR_MEAN_XGB_VALIDATION_SCORING_COMPLETE":
    raise ValueError("Step 5.12N7 is not complete.")
if m8.get("validation_outcomes_loaded") is not False:
    raise ValueError("Step 5.12N7 was not outcome-blind.")
if m8.get("validation_metrics_calculated") is not False:
    raise ValueError("Step 5.12N7 calculated Validation metrics prematurely.")
if m8.get("threshold_selected") or m8.get("priority_categories_selected"):
    raise ValueError("Step 5.12N7 selected a threshold or category prematurely.")
if m8.get("test_accessed"):
    raise ValueError("Step 5.12N7 reports Test access.")
if sha256_file(SCORE_FILE) != m8["score_file_sha256"]:
    raise ValueError("The frozen outcome-blind Validation score file changed.")

scores = pd.read_parquet(SCORE_FILE)
if list(scores.columns) != [ID_COL, SCORE_COL]:
    raise ValueError(f"Unexpected frozen score schema: {list(scores.columns)}")
if len(scores) != EXPECTED_VALIDATION_PATIENTS:
    raise ValueError("Unexpected number of frozen Validation scores.")
scores["_PATIENT_KEY"] = patient_key(scores[ID_COL])
if scores["_PATIENT_KEY"].isna().any() or scores["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Frozen Validation score identifiers are invalid.")

schema_names = pq.ParquetFile(VALIDATION_SOURCE).schema_arrow.names
patient_column = resolve_name(
    schema_names, ["patient_id", "PATIENT_ID"], [("patient",), ("id",)], "patient ID"
)
split_column = resolve_name(
    schema_names, ["DATA_SPLIT", "data_split", "SPLIT"], [("split",)], "data split"
)
label_column = resolve_name(
    schema_names,
    ["CKD_CLASS", "OBSERVED_CKD_LABEL", "CKD_LABEL", "LABEL"],
    [("ckd",), ("class", "label")],
    "documented CKD label",
)
supporting_fields = [
    "DERIVED_EGFR_CKD_EPI_2021",
    "EGFR_PREVIOUS_90D",
    "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS",
]
supporting_fields = [field for field in supporting_fields if field in schema_names]

# Intentional locked opening for the final score vector. Test is never read.
source = pd.read_parquet(
    VALIDATION_SOURCE,
    columns=[patient_column, split_column, label_column] + supporting_fields,
)
validation_mask = source[split_column].astype("string").str.strip().str.lower().isin(
    ["validation", "valid", "val"]
)
validation = source.loc[validation_mask].reset_index(drop=True).copy()
validation["_PATIENT_KEY"] = patient_key(validation[patient_column])
if validation["_PATIENT_KEY"].isna().any() or validation["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Validation outcome-source patient identifiers are invalid.")
validation[LABEL_COL] = pd.to_numeric(validation[label_column], errors="raise").astype(int)
if not validation[LABEL_COL].isin([0, 1]).all():
    raise ValueError("Validation labels are not binary.")

evaluation = scores.merge(
    validation[["_PATIENT_KEY", LABEL_COL] + supporting_fields],
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
if len(evaluation) != EXPECTED_VALIDATION_PATIENTS:
    raise ValueError("Evaluation row count changed during Validation linkage.")
if evaluation[LABEL_COL].isna().any():
    raise ValueError("Some frozen-score patients lack a Validation outcome.")

labels = evaluation[LABEL_COL].to_numpy(int)
score_array = pd.to_numeric(evaluation[SCORE_COL], errors="raise").to_numpy(float)
if not np.isfinite(score_array).all() or np.any((score_array < 0) | (score_array > 1)):
    raise ValueError("Frozen Validation scores are invalid.")
positive_count = int(labels.sum())
unlabeled_count = int((labels == 0).sum())
if positive_count == 0 or unlabeled_count == 0:
    raise ValueError("Both observed-label groups must be represented.")

metrics = metric_values(labels, score_array)
metric_table = pd.DataFrame(
    [{"Method": "Official PULSCAR + mean-imputed XGBoost", **metrics}]
)

# Stratified patient bootstrap for uncertainty. It does not alter the model,
# ranking rule, or workload definitions.
positive_positions = np.flatnonzero(labels == 1)
unlabeled_positions = np.flatnonzero(labels == 0)
rng = np.random.default_rng(BOOTSTRAP_SEED)
bootstrap_rows = []
started = time.perf_counter()
for replicate in range(1, BOOTSTRAP_REPLICATES + 1):
    sampled = np.concatenate(
        [
            rng.choice(positive_positions, len(positive_positions), replace=True),
            rng.choice(unlabeled_positions, len(unlabeled_positions), replace=True),
        ]
    )
    bootstrap_rows.append(
        {"Replicate": replicate, **metric_values(labels[sampled], score_array[sampled])}
    )
    if replicate % 100 == 0:
        print(f"Bootstrap {replicate}/{BOOTSTRAP_REPLICATES}")
bootstrap = pd.DataFrame(bootstrap_rows)
ci_table = pd.DataFrame(
    [
        {
            "Measure": measure,
            "Point_estimate": metrics[measure],
            "Bootstrap_95CI_lower": float(bootstrap[measure].quantile(0.025)),
            "Bootstrap_95CI_upper": float(bootstrap[measure].quantile(0.975)),
        }
        for measure in ["Average_precision", "ROC_AUC", "Brier_score", "Log_loss"]
    ]
)

# Equal-frequency calibration description only. No recalibration is performed.
calibration = evaluation[[SCORE_COL, LABEL_COL]].copy()
calibration["Calibration_decile"] = pd.qcut(
    calibration[SCORE_COL].rank(method="first"), q=10, labels=False
) + 1
calibration_table = (
    calibration.groupby("Calibration_decile", observed=True)
    .agg(
        Patients=(LABEL_COL, "size"),
        Mean_score=(SCORE_COL, "mean"),
        Minimum_score=(SCORE_COL, "min"),
        Maximum_score=(SCORE_COL, "max"),
        Observed_documented_CKD_rate=(LABEL_COL, "mean"),
        Observed_documented_CKD=(LABEL_COL, "sum"),
    )
    .reset_index()
)

ordered = stable_rank(evaluation)
workload_rows = []
for fraction in WORKLOAD_FRACTIONS:
    count = max(1, int(np.ceil(fraction * len(ordered))))
    selected = ordered.head(count)
    captured = int(selected[LABEL_COL].sum())
    workload_rows.append(
        {
            "Workload_percent": int(round(100 * fraction)),
            "Patients_reviewed": count,
            "Observed_documented_CKD_captured": captured,
            "Observed_documented_CKD_recall_percent": 100.0 * captured / positive_count,
            "Observed_documented_CKD_precision_percent": 100.0 * captured / count,
            "Minimum_selected_score_descriptive_only": float(selected[SCORE_COL].min()),
        }
    )
workload_table = pd.DataFrame(workload_rows)

# Lock exact rank-capacity categories. The rule uses no outcomes and transports
# to Test by rank fraction, not by a numeric probability cutoff.
high_count = int(np.ceil(HIGH_FRACTION * len(ordered)))
moderate_plus_high_count = int(
    np.ceil(MODERATE_PLUS_HIGH_FRACTION * len(ordered))
)
ordered["REVIEW_PRIORITY_CATEGORY"] = "Low"
ordered.loc[: moderate_plus_high_count - 1, "REVIEW_PRIORITY_CATEGORY"] = "Moderate"
ordered.loc[: high_count - 1, "REVIEW_PRIORITY_CATEGORY"] = "High"
ordered["REVIEW_RANK"] = np.arange(1, len(ordered) + 1, dtype=int)
ordered["REVIEW_PERCENTILE"] = 100.0 * (
    len(ordered) - ordered["REVIEW_RANK"] + 1
) / len(ordered)

priority_table = (
    ordered.groupby("REVIEW_PRIORITY_CATEGORY", observed=True)
    .agg(
        Patients=(LABEL_COL, "size"),
        Minimum_score_descriptive_only=(SCORE_COL, "min"),
        Maximum_score_descriptive_only=(SCORE_COL, "max"),
        Observed_documented_CKD=(LABEL_COL, "sum"),
        Observed_documented_CKD_rate=(LABEL_COL, "mean"),
    )
    .reindex(["Low", "Moderate", "High"])
    .reset_index()
)
priority_table["Observed_documented_CKD_recall_percent"] = (
    100.0 * priority_table["Observed_documented_CKD"] / positive_count
)

# Supporting renal evidence among high-ranked originally unlabeled patients.
# This is descriptive and does not modify scores, labels, or categories.
latest = pd.to_numeric(
    ordered.get(
        "DERIVED_EGFR_CKD_EPI_2021", pd.Series(np.nan, index=ordered.index)
    ),
    errors="coerce",
)
previous = pd.to_numeric(
    ordered.get("EGFR_PREVIOUS_90D", pd.Series(np.nan, index=ordered.index)),
    errors="coerce",
)
gap = pd.to_numeric(
    ordered.get(
        "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS",
        pd.Series(np.nan, index=ordered.index),
    ),
    errors="coerce",
)
ordered["PERSISTENT_LOW_EGFR_90D"] = latest.lt(60) & previous.lt(60) & gap.ge(90)
ordered["LATEST_EGFR_LT30"] = latest.lt(30)
ordered["LATEST_EGFR_LT15"] = latest.lt(15)
unlabeled_ordered = ordered.loc[ordered[LABEL_COL].eq(0)].reset_index(drop=True)
clinical_rows = []
for fraction in WORKLOAD_FRACTIONS:
    count = max(1, int(np.ceil(fraction * len(unlabeled_ordered))))
    selected = unlabeled_ordered.head(count)
    clinical_rows.append(
        {
            "Unlabeled_workload_percent": int(round(100 * fraction)),
            "Unlabeled_patients_reviewed": count,
            "Persistent_low_eGFR_90d": int(selected["PERSISTENT_LOW_EGFR_90D"].sum()),
            "Persistent_low_eGFR_90d_percent": float(
                100.0 * selected["PERSISTENT_LOW_EGFR_90D"].mean()
            ),
            "Latest_eGFR_below_30": int(selected["LATEST_EGFR_LT30"].sum()),
            "Latest_eGFR_below_15": int(selected["LATEST_EGFR_LT15"].sum()),
        }
    )
clinical_table = pd.DataFrame(clinical_rows)

workload_lock = {
    "score_interpretation": (
        "Review-priority ranking only; not a calibrated probability or CKD diagnosis."
    ),
    "transport_rule": "Apply rank fractions separately within each locked cohort.",
    "tie_breaker": "SHA256 of the private patient key with the fixed N8 salt.",
    "priority_categories": {
        "High": "highest 10% of scores",
        "Moderate": "next 40% of scores",
        "Low": "remaining 50% of scores",
    },
    "descriptive_workloads_percent": [2, 5, 10],
    "primary_high_priority_workload_percent": 10,
    "numeric_score_cutoff_transported_to_test": False,
    "recalibration_permitted_before_test": False,
    "model_change_permitted_before_test": False,
}

context = {
    "step": "5.12N8",
    "status": "LOCKED_VALIDATION_CHARACTERIZATION_AND_WORKLOAD_LOCK",
    "score_file_sha256": sha256_file(SCORE_FILE),
    "step_5_12N7_completion_sha256": sha256_file(M8_COMPLETION),
    "validation_outcome_source": str(VALIDATION_SOURCE),
    "validation_outcome_source_sha256": sha256_file(VALIDATION_SOURCE),
    "patients_evaluated": len(evaluation),
    "documented_ckd": positive_count,
    "unlabeled": unlabeled_count,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "validation_outcomes_loaded": True,
    "validation_previously_accessed_for_superseded_candidate_in_step_5_12K": True,
    "model_trained_tuned_or_changed": False,
    "recalibrated": False,
    "rank_based_workload_rule_locked": True,
    "test_accessed": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = ROOT / "05_12N8_validation_characterization_and_workload_lock" / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
atomic_json(OUTPUT_DIR / "context.json", context)

metric_table.to_csv(OUTPUT_DIR / "validation_observed_label_metrics.csv", index=False)
ci_table.to_csv(OUTPUT_DIR / "validation_bootstrap_confidence_intervals.csv", index=False)
calibration_table.to_csv(OUTPUT_DIR / "validation_calibration_deciles.csv", index=False)
workload_table.to_csv(OUTPUT_DIR / "validation_fixed_workload_capture.csv", index=False)
priority_table.to_csv(OUTPUT_DIR / "validation_priority_category_summary.csv", index=False)
clinical_table.to_csv(OUTPUT_DIR / "validation_unlabeled_renal_evidence.csv", index=False)
atomic_json(OUTPUT_DIR / "locked_workload_and_priority_policy.json", workload_lock)
bootstrap.to_parquet(PRIVATE_DIR / "bootstrap_replicates.parquet", index=False)
ordered[
    [
        ID_COL,
        "_PATIENT_KEY",
        SCORE_COL,
        LABEL_COL,
        "REVIEW_RANK",
        "REVIEW_PERCENTILE",
        "REVIEW_PRIORITY_CATEGORY",
        "PERSISTENT_LOW_EGFR_90D",
        "LATEST_EGFR_LT30",
        "LATEST_EGFR_LT15",
    ]
].to_parquet(
    PRIVATE_DIR / "patient_level_validation_characterization.parquet", index=False
)

completion = {
    **context,
    "status": "LOCKED_N8_VALIDATION_CHARACTERIZATION_AND_WORKLOAD_LOCK_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "metrics": metrics,
    "bootstrap_runtime_minutes": (time.perf_counter() - started) / 60.0,
    "workload_and_priority_policy": workload_lock,
    "output_dir": str(OUTPUT_DIR),
    "model_replaced": False,
    "clinician_review_sample_replaced": False,
    "next_step": (
        "Write and freeze the one-time Test protocol, then score Test outcome-blind "
        "before loading Test outcomes exactly once."
    ),
    "interpretation": (
        "Validation performance is against documented CKD versus unlabeled status, "
        "not an adjudicated latent-CKD reference standard."
    ),
}
atomic_json(OUTPUT_DIR / "completion.json", completion)

print("=" * 108)
print("STEP 5.12N8 — LOCKED VALIDATION CHARACTERIZATION AND WORKLOAD LOCK COMPLETE")
print("=" * 108)
print("\nVALIDATION COHORT")
print(
    json.dumps(
        {
            "patients": len(evaluation),
            "documented_ckd": positive_count,
            "unlabeled": unlabeled_count,
        },
        indent=2,
    )
)
print("\nOBSERVED-LABEL METRICS")
display(metric_table)
print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")
display(ci_table)
print("\nCALIBRATION DECILES — DESCRIPTION ONLY")
display(calibration_table)
print("\nFIXED-WORKLOAD CAPTURE")
display(workload_table)
print("\nLOCKED PRIORITY-CATEGORY SUMMARY")
display(priority_table)
print("\nRENAL EVIDENCE AMONG HIGH-RANKED UNLABELED PATIENTS")
display(clinical_table)
print("\nLOCKED WORKLOAD AND PRIORITY POLICY")
print(json.dumps(workload_lock, indent=2, ensure_ascii=False))
print(f"\nSaved run: {OUTPUT_DIR}")
print("Validation outcomes loaded: Yes — locked characterization of the final score vector")
print("Prior Validation access: Yes — superseded candidate in Step 5.12K")
print("Model trained, tuned, recalibrated, or changed: No")
print("Rank-based workload and priority categories locked: Yes")
print("Test accessed: No")
print("Clinician-review sample replaced: No")
