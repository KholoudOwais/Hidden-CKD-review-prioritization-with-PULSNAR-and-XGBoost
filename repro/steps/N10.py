"""STEP 5.12N10 — Reference-grounded reporting and Internal Test protocol freeze.

This step performs no modeling and reads no Internal Test predictors or outcomes.
It supersedes the clinician-facing 10/40/50 priority-category presentation with:

  * continuous locked-model output;
  * within-cohort percentile rank; and
  * a secondary prevalence-matched review set whose size is determined only by
    the official PULSCAR Training estimate alpha.

The 2%, 5%, and 10% Validation workload results remain descriptive sensitivity
analyses in their historical artifact; they are not transported as thresholds,
clinical categories, or clinician-facing labels.
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


PROJECT_ROOT = Path("/content/drive/MyDrive/CKD_Project")
ROOT = PROJECT_ROOT / (
    "outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
N6_ROOT = ROOT / "05_12N6_full_training_pulscar_mean_xgb_final_refit_and_lock"
N8_ROOT = ROOT / "05_12N8_validation_characterization_and_workload_lock"
N9_ROOT = ROOT / (
    "05_12N9_validation_low_priority_documented_ckd_safety_audit"
)
OUTPUT_ROOT = ROOT / (
    "05_12N10_reference_grounded_reporting_and_internal_test_protocol_freeze"
)

PULSCAR_REFERENCE = {
    "title": (
        "Positive Unlabeled Learning Selected Not At Random (PULSNAR): "
        "class proportion estimation without the selected completely at "
        "random assumption"
    ),
    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11622864/",
    "methodological_role": (
        "PULSCAR estimates alpha, the fraction of positives among unlabeled "
        "examples, and uses the highest-ranked alpha-times-|U| unlabeled "
        "examples as probable positives."
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


N6_COMPLETION, n6 = latest_completed(
    N6_ROOT, "FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED"
)
N8_COMPLETION, n8 = latest_completed(
    N8_ROOT, "LOCKED_N8_VALIDATION_CHARACTERIZATION_AND_WORKLOAD_LOCK_COMPLETE"
)
N9_COMPLETION, n9 = latest_completed(
    N9_ROOT, "VALIDATION_LOW_PRIORITY_DOCUMENTED_CKD_SAFETY_AUDIT_COMPLETE"
)

if n6.get("pipeline") != "official PULSCAR discovery -> mean-imputed XGBoost":
    raise ValueError("The locked N6 pipeline changed.")
if n6.get("threshold_selected") or n6.get("priority_categories_selected"):
    raise ValueError("N6 unexpectedly selected a threshold or categories.")
if n6.get("uacr_used_anywhere_in_modeling") is not False:
    raise ValueError("N6 reports uACR use.")
if n6.get("validation_accessed_in_this_step") or n6.get("test_accessed_in_this_step"):
    raise ValueError("N6 reports held-out access.")
if n8.get("test_accessed") is not False:
    raise ValueError("N8 reports Internal Test access.")
if n9.get("test_accessed") is not False:
    raise ValueError("N9 reports Internal Test access.")
if any(
    n9.get(field) is not False
    for field in [
        "model_changed",
        "preprocessing_changed",
        "scores_changed",
        "priority_categories_changed",
    ]
):
    raise ValueError("N9 reports a forbidden model or score change.")

alpha = float(n6["estimated_alpha"])
training_unlabeled = int(n6["unlabeled"])
training_selected = int(n6["selected_probable_positive"])
if not (0 < alpha < 1):
    raise ValueError(f"Invalid locked PULSCAR alpha: {alpha}")

# The official run used integer truncation of alpha * |U|. Because alpha is
# stored at the package precision, allow at most one record of rounding drift.
alpha_implied_training_count = int(math.floor(alpha * training_unlabeled))
if abs(alpha_implied_training_count - training_selected) > 1:
    raise ValueError(
        "The locked selected probable-positive count is inconsistent with alpha."
    )

validation_unlabeled = int(n8["unlabeled"])
validation_alpha_matched_count = int(math.floor(alpha * validation_unlabeled))
if validation_alpha_matched_count < 1:
    raise ValueError("The alpha-matched Validation count is unexpectedly zero.")

reporting_policy = {
    "primary_model_output": {
        "raw_output": "XGBoost positive-class output on the 0-to-1 scale",
        "percentage_display": "100 times the raw model output",
        "required_label": "Uncalibrated model-estimated positive-class output",
        "forbidden_label": "Clinical probability of CKD",
    },
    "relative_rank": {
        "display": "Within-cohort percentile rank",
        "definition": (
            "Percentage of cohort members with a model score less than or "
            "equal to the patient's score, with a deterministic hash tie-break."
        ),
        "interpretation": "Ranking only; not disease probability or KDIGO stage.",
    },
    "clinician_facing_categories": None,
    "high_moderate_low_labels_permitted": False,
    "historical_fixed_workloads": {
        "values_percent": [2, 5, 10],
        "role": (
            "Descriptive Validation sensitivity analyses only; not thresholds, "
            "not transported to Internal Test, and not shown as clinical categories."
        ),
    },
    "pulscar_alpha_matched_review_set": {
        "role": "Secondary method-grounded candidate review set among unlabeled records.",
        "alpha_source": "Locked official PULSCAR Training estimate",
        "alpha": alpha,
        "formula": "K_c = floor(alpha_training * N_unlabeled,c)",
        "ranking_rule": "Select the K_c highest locked-model scores among unlabeled records.",
        "numeric_score_cutoff_transport": False,
        "validation_expected_count": validation_alpha_matched_count,
        "interpretation": (
            "Prevalence-matched probable-positive candidate set; not a CKD "
            "diagnosis and not an individual calibrated probability."
        ),
        "assumption": (
            "The Training PULSCAR alpha is transported as a class-proportion "
            "estimate to the target cohort; this assumption must be disclosed."
        ),
    },
}

clinician_review_policy = {
    "phase_a_independent_assessment": {
        "model_score_visible": False,
        "percentile_visible": False,
        "shap_visible": False,
        "uacr_visible": False,
        "nonmodel_imaging_proteinuria_pathology_visible": False,
        "features": "Observed values from the locked 20-feature model panel only.",
        "missing_values": "Display as Not available; never display imputed values as observed.",
        "interpretation": (
            "Independent clinical face-validity assessment using the same "
            "structured information domain available to the model."
        ),
    },
    "phase_b_model_reveal": {
        "model_score_visible": True,
        "model_output_percentage_visible": True,
        "percentile_visible": True,
        "shap_visible": True,
        "shap_features": "Observed model features only.",
        "uacr_visible": False,
        "priority_category_visible": False,
    },
    "not_claimed": [
        "Gold-standard CKD adjudication",
        "Validation of a calibrated clinical CKD probability",
        "KDIGO diagnosis or staging",
    ],
}

internal_test_protocol = [
    {
        "order": 1,
        "stage": "Verify immutable artifacts",
        "operation": (
            "Verify the hashes of the locked N6 model, preprocessing artifact, "
            "and transformed-feature manifest."
        ),
    },
    {
        "order": 2,
        "stage": "Locate predictor-only Internal Test table",
        "operation": "Do not load an outcome or documented-CKD label column.",
    },
    {
        "order": 3,
        "stage": "Outcome-blind scoring",
        "operation": (
            "Apply the unchanged mean-imputation preprocessor and locked XGBoost "
            "model; save patient IDs, raw outputs, and original availability."
        ),
    },
    {
        "order": 4,
        "stage": "Freeze continuous outputs",
        "operation": (
            "Save a hashed score artifact before any Internal Test outcome is opened."
        ),
    },
    {
        "order": 5,
        "stage": "Open Internal Test outcomes once",
        "operation": (
            "Link the frozen score artifact to documented CKD versus unlabeled status."
        ),
    },
    {
        "order": 6,
        "stage": "Primary continuous evaluation",
        "operation": (
            "Report average precision with stratified patient-bootstrap 95% CI; "
            "report ROC AUC, Brier score, and log loss as supporting metrics."
        ),
    },
    {
        "order": 7,
        "stage": "Rank evaluation",
        "operation": (
            "Report the precision-recall curve and percentile distribution; do "
            "not create High/Moderate/Low categories."
        ),
    },
    {
        "order": 8,
        "stage": "Secondary PULSCAR-alpha review set",
        "operation": (
            "Among records confirmed to be unlabeled after outcome linkage, "
            "select floor(alpha_training * N_unlabeled,test) highest scores and "
            "report the count and descriptive supporting audit separately."
        ),
    },
    {
        "order": 9,
        "stage": "Close internal evaluation",
        "operation": (
            "No retraining, recalibration, feature change, method switching, or "
            "post-Test threshold optimization."
        ),
    },
]

context = {
    "step": "5.12N10",
    "status": "REFERENCE_GROUNDED_REPORTING_AND_INTERNAL_TEST_PROTOCOL_FREEZE",
    "n6_completion_sha256": sha256_file(N6_COMPLETION),
    "n8_completion_sha256": sha256_file(N8_COMPLETION),
    "n9_completion_sha256": sha256_file(N9_COMPLETION),
    "locked_training_alpha": alpha,
    "reference": PULSCAR_REFERENCE,
    "validation_outcomes_previously_opened": True,
    "internal_test_previously_accessed_for_superseded_pipeline": True,
    "internal_test_predictors_accessed_in_this_step": False,
    "internal_test_outcomes_accessed_in_this_step": False,
    "model_or_preprocessing_changed": False,
    "scores_changed": False,
    "clinician_review_materials_changed_in_this_step": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

atomic_json(OUTPUT_DIR / "context.json", context)
atomic_json(OUTPUT_DIR / "locked_reporting_policy.json", reporting_policy)
atomic_json(OUTPUT_DIR / "locked_clinician_review_policy.json", clinician_review_policy)
atomic_json(OUTPUT_DIR / "frozen_internal_test_protocol.json", internal_test_protocol)

completion = {
    **context,
    "status": (
        "REFERENCE_GROUNDED_REPORTING_AND_INTERNAL_TEST_PROTOCOL_FREEZE_COMPLETE"
    ),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(OUTPUT_DIR),
    "reporting_policy": reporting_policy,
    "clinician_review_policy": clinician_review_policy,
    "internal_test_protocol": internal_test_protocol,
    "model_changed": False,
    "preprocessing_changed": False,
    "numeric_threshold_selected": False,
    "clinical_priority_categories_selected": False,
    "test_accessed": False,
    "next_step": (
        "Outcome-blind scoring of the immutable N6 model on the Internal Test "
        "predictor-only table; save and hash scores before opening outcomes."
    ),
}
atomic_json(OUTPUT_DIR / "completion.json", completion)

print("=" * 120)
print(
    "STEP 5.12N10 — REFERENCE-GROUNDED REPORTING AND INTERNAL TEST "
    "PROTOCOL FREEZE COMPLETE"
)
print("=" * 120)
print("\nMETHOD-GROUNDED SELECTION BASIS")
print(
    json.dumps(
        {
            "official_PULSCAR_training_alpha": alpha,
            "training_unlabeled": training_unlabeled,
            "training_selected_probable_positive": training_selected,
            "formula": "K_c = floor(alpha_training * N_unlabeled,c)",
            "validation_unlabeled": validation_unlabeled,
            "validation_alpha_matched_count": validation_alpha_matched_count,
        },
        indent=2,
        ensure_ascii=False,
    )
)
print("\nLOCKED REPORTING POLICY")
print(json.dumps(reporting_policy, indent=2, ensure_ascii=False))
print("\nLOCKED CLINICIAN-REVIEW POLICY")
print(json.dumps(clinician_review_policy, indent=2, ensure_ascii=False))
print("\nFROZEN INTERNAL TEST PROTOCOL")
for item in internal_test_protocol:
    print(f"{item['order']}. {item['stage']}: {item['operation']}")
print(f"\nSaved run: {OUTPUT_DIR}")
print("High/Moderate/Low clinician-facing categories: Removed")
print("2%, 5%, and 10% transported as decision thresholds: No")
print("Selection basis: Official PULSCAR Training alpha")
print("uACR visible to clinicians or SHAP: No")
print("Internal Test predictors accessed: No")
print("Internal Test outcomes accessed: No")
print("Model or preprocessing changed: No")
