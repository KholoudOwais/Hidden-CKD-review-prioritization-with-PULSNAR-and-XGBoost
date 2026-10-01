"""STEP 5.12N12 — Locked Internal Test evaluation.

Opens Internal Test outcomes only after the immutable N11 continuous-score
artifact has been written and hash-locked.  It evaluates the continuous model
output against observed documented-CKD versus unlabeled status, and applies the
secondary PULSCAR-alpha review-set rule frozen in N10.

This step does not train, tune, recalibrate, switch, or modify the model.  It
does not create High/Moderate/Low categories, transport a numeric score cutoff,
or report the historical 2%, 5%, and 10% workload splits as decision rules.
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
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


EXPECTED_TEST_PATIENTS = 14_927
EXPECTED_TEST_DOCUMENTED_CKD = 1_071
EXPECTED_TEST_UNLABELED = 13_856
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 51_212

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
N8_ROOT = ROOT / "05_12N8_validation_characterization_and_workload_lock"
N10_ROOT = ROOT / (
    "05_12N10_reference_grounded_reporting_and_internal_test_protocol_freeze"
)
N11_ROOT = ROOT / (
    "05_12N11_outcome_blind_locked_pulscar_mean_xgb_internal_test_scoring"
)
OUTPUT_ROOT = ROOT / "05_12N12_locked_internal_test_evaluation"

# This is the previously verified authoritative linkage source.  Only the ID
# and documented-CKD label are loaded from it.
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


N6_COMPLETION, n6 = latest_completed(
    N6_ROOT, "FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED"
)
N8_COMPLETION, n8 = latest_completed(
    N8_ROOT, "LOCKED_N8_VALIDATION_CHARACTERIZATION_AND_WORKLOAD_LOCK_COMPLETE"
)
N10_COMPLETION, n10 = latest_completed(
    N10_ROOT,
    "REFERENCE_GROUNDED_REPORTING_AND_INTERNAL_TEST_PROTOCOL_FREEZE_COMPLETE",
)
N11_COMPLETION, n11 = latest_completed(
    N11_ROOT, "OUTCOME_BLIND_LOCKED_INTERNAL_TEST_SCORING_COMPLETE"
)

SCORE_FILE = Path(n11["score_file"])
TEST_PREDICTORS = Path(n11["internal_test_predictors"])

for required in [
    N6_COMPLETION,
    N8_COMPLETION,
    N10_COMPLETION,
    N11_COMPLETION,
    SCORE_FILE,
    TEST_PREDICTORS,
    OUTCOME_SOURCE,
]:
    if not required.is_file():
        raise FileNotFoundError(required)

# Immutable-state checks before the outcome column is opened.
if n6.get("pipeline") != "official PULSCAR discovery -> mean-imputed XGBoost":
    raise ValueError("The locked N6 pipeline identity changed.")
if n6.get("uacr_used_anywhere_in_modeling") is not False:
    raise ValueError("N6 reports uACR use.")
if n10.get("model_changed") or n10.get("preprocessing_changed"):
    raise ValueError("N10 reports a forbidden pipeline change.")
if n10.get("numeric_threshold_selected"):
    raise ValueError("N10 unexpectedly selected a numeric threshold.")
if n10.get("clinical_priority_categories_selected"):
    raise ValueError("N10 unexpectedly selected clinical priority categories.")
if n11.get("internal_test_outcomes_loaded") is not False:
    raise ValueError("N11 was not outcome-blind.")
if n11.get("internal_test_metrics_calculated") is not False:
    raise ValueError("N11 calculated Internal Test metrics prematurely.")
if n11.get("alpha_matched_review_set_applied") is not False:
    raise ValueError("N11 applied the alpha review rule before outcome linkage.")
if n11.get("priority_categories_created") is not False:
    raise ValueError("N11 created forbidden priority categories.")
if n11.get("model_or_preprocessing_changed") is not False:
    raise ValueError("N11 reports a forbidden pipeline change.")
if sha256_file(SCORE_FILE) != n11["score_file_sha256"]:
    raise ValueError("The frozen N11 score file changed.")
if sha256_file(TEST_PREDICTORS) != n11["internal_test_predictors_sha256"]:
    raise ValueError("The authoritative Internal Test predictor table changed.")

scores = pd.read_parquet(SCORE_FILE)
expected_score_columns = [ID_COL, SCORE_COL, PERCENT_COL, PERCENTILE_COL]
if list(scores.columns) != expected_score_columns:
    raise ValueError(f"Unexpected N11 score schema: {list(scores.columns)}")
if len(scores) != EXPECTED_TEST_PATIENTS:
    raise ValueError("Unexpected number of frozen Internal Test scores.")

scores["_PATIENT_KEY"] = patient_key(scores[ID_COL])
if scores["_PATIENT_KEY"].isna().any() or scores["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Frozen-score patient identifiers are invalid.")
score_array = pd.to_numeric(scores[SCORE_COL], errors="raise").to_numpy(float)
percent_array = pd.to_numeric(scores[PERCENT_COL], errors="raise").to_numpy(float)
percentile_array = pd.to_numeric(
    scores[PERCENTILE_COL], errors="raise"
).to_numpy(float)
if not np.isfinite(score_array).all() or np.any((score_array < 0) | (score_array > 1)):
    raise ValueError("Frozen Internal Test scores are invalid.")
if not np.allclose(percent_array, 100.0 * score_array, rtol=0, atol=1e-10):
    raise ValueError("Frozen percentage displays no longer equal 100 times score.")
if not np.isfinite(percentile_array).all() or np.any(
    (percentile_array <= 0) | (percentile_array > 100)
):
    raise ValueError("Frozen within-cohort percentiles are invalid.")

# Intentional one-time outcome linkage after all score-integrity checks pass.
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
outcomes = pd.read_parquet(
    OUTCOME_SOURCE, columns=[outcome_id_col, outcome_label_col]
).reset_index(drop=True)
outcomes["_PATIENT_KEY"] = patient_key(outcomes[outcome_id_col])
if outcomes["_PATIENT_KEY"].isna().any() or outcomes["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Internal Test outcome-source identifiers are invalid.")
outcomes[LABEL_COL] = pd.to_numeric(
    outcomes[outcome_label_col], errors="raise"
).astype(int)
if not outcomes[LABEL_COL].isin([0, 1]).all():
    raise ValueError("Internal Test observed labels are not binary.")

evaluation = scores.merge(
    outcomes[["_PATIENT_KEY", LABEL_COL]],
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
if len(evaluation) != EXPECTED_TEST_PATIENTS:
    raise ValueError("Evaluation row count changed during Internal Test linkage.")
if evaluation[LABEL_COL].isna().any():
    missing_n = int(evaluation[LABEL_COL].isna().sum())
    raise ValueError(f"{missing_n} frozen-score patients lack an Internal Test outcome.")

labels = evaluation[LABEL_COL].to_numpy(int)
documented_ckd = int(labels.sum())
unlabeled_n = int((labels == 0).sum())
if documented_ckd != EXPECTED_TEST_DOCUMENTED_CKD:
    raise ValueError(
        f"Expected {EXPECTED_TEST_DOCUMENTED_CKD:,} documented CKD records; "
        f"found {documented_ckd:,}."
    )
if unlabeled_n != EXPECTED_TEST_UNLABELED:
    raise ValueError(
        f"Expected {EXPECTED_TEST_UNLABELED:,} unlabeled records; found {unlabeled_n:,}."
    )

metrics = metric_values(labels, evaluation[SCORE_COL].to_numpy(float))
metric_table = pd.DataFrame(
    [{"Method": "Official PULSCAR + mean-imputed XGBoost", **metrics}]
)

# Prespecified stratified patient bootstrap.
positive_positions = np.flatnonzero(labels == 1)
unlabeled_positions = np.flatnonzero(labels == 0)
rng = np.random.default_rng(BOOTSTRAP_SEED)
bootstrap_rows = []
bootstrap_started = time.perf_counter()
for replicate in range(1, BOOTSTRAP_REPLICATES + 1):
    sampled = np.concatenate(
        [
            rng.choice(positive_positions, len(positive_positions), replace=True),
            rng.choice(unlabeled_positions, len(unlabeled_positions), replace=True),
        ]
    )
    bootstrap_rows.append(
        {
            "Replicate": replicate,
            **metric_values(
                labels[sampled], evaluation[SCORE_COL].to_numpy(float)[sampled]
            ),
        }
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

# Descriptive score deciles only.  These are not clinical risk categories.
calibration = evaluation[[SCORE_COL, LABEL_COL]].copy()
calibration["Score_decile"] = (
    pd.qcut(calibration[SCORE_COL].rank(method="first"), q=10, labels=False) + 1
)
decile_table = (
    calibration.groupby("Score_decile", observed=True)
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

precision, recall, pr_thresholds = precision_recall_curve(
    labels, evaluation[SCORE_COL].to_numpy(float)
)
pr_curve = pd.DataFrame(
    {
        "Recall": recall,
        "Precision": precision,
        "Threshold": np.append(pr_thresholds, np.nan),
    }
)

# Validation-to-Test comparison is descriptive and cannot change the model.
validation_metrics = n8.get("metrics")
if not isinstance(validation_metrics, dict):
    raise ValueError("N8 completion lacks locked Validation metrics.")
comparison_table = pd.DataFrame(
    [
        {
            "Measure": measure,
            "Validation": float(validation_metrics[measure]),
            "Internal_Test": float(metrics[measure]),
            "Internal_Test_minus_Validation": (
                float(metrics[measure]) - float(validation_metrics[measure])
            ),
        }
        for measure in ["Average_precision", "ROC_AUC", "Brier_score", "Log_loss"]
    ]
)

# Secondary method-grounded review set.  Labels are used only to identify the
# unlabeled pool; the ranking itself uses only the already frozen N11 scores.
alpha = float(n6["estimated_alpha"])
if not (0 < alpha < 1):
    raise ValueError(f"Invalid locked Training PULSCAR alpha: {alpha}")
alpha_review_n = int(math.floor(alpha * unlabeled_n))
if alpha_review_n < 1:
    raise ValueError("The alpha-matched Internal Test review count is zero.")

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

# A model-feature-only descriptive audit; uACR and excluded longitudinal fields
# are deliberately not loaded.  Imputed values are never represented as observed.
predictor_schema = pq.ParquetFile(TEST_PREDICTORS).schema_arrow.names
EGFR_COL = "DERIVED_EGFR_CKD_EPI_2021"
if ID_COL not in predictor_schema or EGFR_COL not in predictor_schema:
    raise KeyError(f"Authoritative Test predictors lack {ID_COL!r} or {EGFR_COL!r}.")
egfr = pd.read_parquet(TEST_PREDICTORS, columns=[ID_COL, EGFR_COL])
egfr["_PATIENT_KEY"] = patient_key(egfr[ID_COL])
if egfr["_PATIENT_KEY"].isna().any() or egfr["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Authoritative Test predictor identifiers are invalid.")
egfr["current_egfr_observed"] = pd.to_numeric(egfr[EGFR_COL], errors="coerce")
alpha_review = alpha_review.merge(
    egfr[["_PATIENT_KEY", "current_egfr_observed"]],
    on="_PATIENT_KEY",
    how="left",
    validate="one_to_one",
)
observed_egfr = alpha_review["current_egfr_observed"]

alpha_summary = {
    "rule": "K_test = floor(alpha_training * N_unlabeled,test)",
    "training_alpha": alpha,
    "internal_test_unlabeled": unlabeled_n,
    "selected_for_secondary_review": alpha_review_n,
    "selection_fraction_of_unlabeled_percent": 100.0 * alpha_review_n / unlabeled_n,
    "minimum_selected_score": float(alpha_review[SCORE_COL].min()),
    "maximum_selected_score": float(alpha_review[SCORE_COL].max()),
    "minimum_selected_percentile": float(alpha_review[PERCENTILE_COL].min()),
    "maximum_selected_percentile": float(alpha_review[PERCENTILE_COL].max()),
    "current_egfr_observed": int(observed_egfr.notna().sum()),
    "current_egfr_below_60": int(observed_egfr.lt(60).sum()),
    "current_egfr_below_30": int(observed_egfr.lt(30).sum()),
    "current_egfr_below_15": int(observed_egfr.lt(15).sum()),
    "interpretation": (
        "Secondary prevalence-matched review set among unlabeled records; "
        "not a CKD diagnosis, clinical risk category, or calibrated probability."
    ),
}
alpha_summary_table = pd.DataFrame([alpha_summary])

context = {
    "step": "5.12N12",
    "status": "LOCKED_INTERNAL_TEST_EVALUATION",
    "n6_completion_sha256": sha256_file(N6_COMPLETION),
    "n8_completion_sha256": sha256_file(N8_COMPLETION),
    "n10_completion_sha256": sha256_file(N10_COMPLETION),
    "n11_completion_sha256": sha256_file(N11_COMPLETION),
    "frozen_score_file": str(SCORE_FILE),
    "frozen_score_file_sha256": sha256_file(SCORE_FILE),
    "outcome_source": str(OUTCOME_SOURCE),
    "outcome_source_sha256": sha256_file(OUTCOME_SOURCE),
    "patients_evaluated": len(evaluation),
    "documented_ckd": documented_ckd,
    "unlabeled": unlabeled_n,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "training_pulscar_alpha": alpha,
    "alpha_matched_review_count": alpha_review_n,
    "internal_test_previously_accessed_for_superseded_pipeline": True,
    "analysis_role": (
        "Locked final-pipeline internal evaluation after outcome-blind N11 "
        "score freezing; the same partition had earlier exploratory use with "
        "a superseded pipeline."
    ),
    "internal_test_outcomes_loaded_in_this_step": True,
    "model_trained_tuned_recalibrated_switched_or_changed": False,
    "numeric_threshold_optimized": False,
    "high_moderate_low_categories_created": False,
    "uacr_loaded_or_reported": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

atomic_json(OUTPUT_DIR / "context.json", context)
atomic_csv(metric_table, OUTPUT_DIR / "internal_test_observed_label_metrics.csv")
atomic_csv(ci_table, OUTPUT_DIR / "internal_test_bootstrap_confidence_intervals.csv")
atomic_csv(decile_table, OUTPUT_DIR / "internal_test_score_deciles_descriptive.csv")
atomic_csv(pr_curve, OUTPUT_DIR / "internal_test_precision_recall_curve.csv")
atomic_csv(comparison_table, OUTPUT_DIR / "validation_internal_test_comparison.csv")
atomic_csv(alpha_summary_table, OUTPUT_DIR / "pulscar_alpha_review_set_summary.csv")
atomic_parquet(bootstrap, PRIVATE_DIR / "bootstrap_replicates.parquet")
atomic_parquet(
    evaluation[
        [
            ID_COL,
            "_PATIENT_KEY",
            SCORE_COL,
            PERCENT_COL,
            PERCENTILE_COL,
            LABEL_COL,
        ]
    ],
    PRIVATE_DIR / "patient_level_internal_test_evaluation.parquet",
)
atomic_parquet(
    alpha_review[
        [
            ID_COL,
            "_PATIENT_KEY",
            "alpha_review_rank",
            SCORE_COL,
            PERCENT_COL,
            PERCENTILE_COL,
            "current_egfr_observed",
        ]
    ],
    PRIVATE_DIR / "pulscar_alpha_matched_unlabeled_review_set.parquet",
)

completion = {
    **context,
    "status": "LOCKED_INTERNAL_TEST_EVALUATION_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "metrics": metrics,
    "bootstrap_runtime_minutes": (
        time.perf_counter() - bootstrap_started
    ) / 60.0,
    "alpha_review_summary": alpha_summary,
    "output_dir": str(OUTPUT_DIR),
    "internal_modeling_evaluation_closed": True,
    "model_changed": False,
    "clinician_review_sample_created_or_replaced": False,
    "interpretation": (
        "Observed-label performance compares documented CKD with unlabeled "
        "status; it is not diagnostic performance against adjudicated latent CKD."
    ),
    "next_step": (
        "Freeze final results, then create the blinded two-phase clinician-review "
        "sample and observed-model-feature-only SHAP materials without changing "
        "the model or Internal Test evaluation."
    ),
}
atomic_json(OUTPUT_DIR / "completion.json", completion)

print("=" * 116)
print("STEP 5.12N12 — LOCKED INTERNAL TEST EVALUATION COMPLETE")
print("=" * 116)
print("\nINTERNAL TEST COHORT")
print(
    json.dumps(
        {
            "patients": len(evaluation),
            "documented_ckd": documented_ckd,
            "unlabeled": unlabeled_n,
        },
        indent=2,
    )
)
print("\nOBSERVED-LABEL METRICS")
display(metric_table)
print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")
display(ci_table)
print("\nSCORE DECILES — DESCRIPTION ONLY, NOT CLINICAL CATEGORIES")
display(decile_table)
print("\nVALIDATION TO INTERNAL-TEST COMPARISON")
display(comparison_table)
print("\nSECONDARY PULSCAR-ALPHA REVIEW SET")
print(json.dumps(alpha_summary, indent=2, ensure_ascii=False))
print(f"\nSaved run: {OUTPUT_DIR}")
print("Internal Test outcomes loaded: Yes — after N11 score hash lock")
print("Model trained, tuned, recalibrated, switched, or changed: No")
print("Numeric threshold optimized on Internal Test: No")
print("High/Moderate/Low categories created: No")
print("uACR loaded or reported: No")
print("Internal modeling evaluation closed: Yes")
print("Clinician-review sample created or replaced: No")
