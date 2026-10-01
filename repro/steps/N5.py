"""STEP 5.12N5 — Training-only post-tuning pipeline selection.

Consumes only the nested OOF predictions from N4. No model is fitted or tuned,
and Validation/Test are not loaded. Official PULSCAR replaces naive PN only if
it passes all three locked gates: AP loss <=0.010, documented-label recall at
5% within 1 percentage point, and top-2% unlabeled kidney-evidence yield at
least as high as naive PN.
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


SEED = 71_000_045
BOOTSTRAP_REPLICATES = 500
WORKLOADS = [0.02, 0.05, 0.10]
PRIMARY_WORKLOAD = 0.02
RECALL_WORKLOAD = 0.05
AP_DELTA_MINIMUM = -0.010
RECALL_TOLERANCE_PP = 1.0

EXPECTED_PATIENTS = 69_701
EXPECTED_POSITIVES = 5_042
EXPECTED_UNLABELED = 64_659
EXPECTED_FOLDS = 5
EXPECTED_UNLABELED_WITH_EVIDENCE = 459
METHODS = ["naive_pn", "official_pulscar"]

UACR_WINDOW_DAYS = 730
CHRONICITY_MIN_DAYS = 90
UACR_A2_THRESHOLD_MG_MMOL = 3.0
EGFR_G3_THRESHOLD = 60.0

PROJECT_ROOT = Path("/content/drive/MyDrive/CKD_Project")
ROOT = PROJECT_ROOT / "outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling"
N4_ROOT = ROOT / "05_12N4_equal_budget_nested_training_only_pulscar_tuning"
OUTPUT_ROOT = ROOT / "05_12N5_training_only_post_tuning_pipeline_selection"
CHECKPOINT_SOURCE = PROJECT_ROOT / "checkpoints/step_3_15_6_20260903_230014"
PATIENT_PATH = CHECKPOINT_SOURCE / "patient_level_data.parquet"
ENCOUNTER_PATH = CHECKPOINT_SOURCE / "df.parquet"


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
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str
    ).encode("utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(json_safe(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalize_patient_id(series):
    values = series.astype("string").str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    if int(numeric.notna().sum()) == int(values.notna().sum()):
        finite = numeric.dropna()
        if np.all(np.abs(finite - np.round(finite)) < 1e-9):
            return numeric.round().astype("Int64").astype("string")
    return values


def read_oof(path, expected_ids, id_col):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_parquet(path)
    if list(table.columns) != [id_col, "probability"]:
        raise ValueError(f"Unexpected OOF schema: {path}")
    if not np.array_equal(table[id_col].to_numpy(), np.asarray(expected_ids)):
        raise ValueError(f"OOF patient order changed: {path}")
    probability = pd.to_numeric(table["probability"], errors="coerce").to_numpy(float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"Invalid OOF probabilities: {path}")
    return probability


def metrics(target, probability):
    clipped = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    return {
        "Average_precision": float(average_precision_score(target, clipped)),
        "ROC_AUC": float(roc_auc_score(target, clipped)),
        "Brier_score": float(brier_score_loss(target, clipped)),
        "Log_loss": float(log_loss(target, clipped, labels=[0, 1])),
    }


def value_at(table, method, workload_column, workload, value_column):
    match = table.loc[
        table["Method"].eq(method)
        & np.isclose(table[workload_column], 100.0 * workload),
        value_column,
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one {value_column} row for {method}.")
    return float(match.iloc[0])


# Restore the latest completed N4 and its exact upstream chain.
n4_candidates = []
if N4_ROOT.exists():
    for completion_path in N4_ROOT.rglob("completion.json"):
        payload = load_json(completion_path)
        if payload and payload.get("status") == "EQUAL_BUDGET_NESTED_TRAINING_ONLY_PULSCAR_TUNING_COMPLETE":
            private = completion_path.parent / "PRIVATE_INTERNAL_DO_NOT_SHARE"
            required = [private / f"{method}__nested_tuned_complete_oof.parquet" for method in METHODS]
            if all(path.is_file() for path in required):
                n4_candidates.append((completion_path, payload))
if not n4_candidates:
    raise FileNotFoundError("Run Step 5.12N4 before Step 5.12N5.")

N4_FILE, n4 = max(n4_candidates, key=lambda item: item[0].stat().st_mtime)
N4_RUN = N4_FILE.parent
N4_PRIVATE = N4_RUN / "PRIVATE_INTERNAL_DO_NOT_SHARE"
N4_SUMMARY = N4_RUN / "nested_tuned_oof_summary.csv"
if not N4_SUMMARY.is_file():
    raise FileNotFoundError(N4_SUMMARY)
if n4.get("methods") != METHODS:
    raise ValueError("N4 method set/order changed.")
if n4.get("trials_per_outer_pipeline") != 20 or n4.get("total_completed_trials_per_pipeline") != 100:
    raise ValueError("N4 equal tuning budget is incomplete.")
if not n4.get("mean_imputation_used") or n4.get("missingness_indicators_used") or n4.get("native_nan_routing_used"):
    raise ValueError("N4 preprocessing policy is incompatible.")
if n4.get("automatic_final_winner_selected") or n4.get("validation_accessed_in_this_step") or n4.get("test_accessed_in_this_step"):
    raise ValueError("N4 reports prohibited selection or held-out access.")

N3_FILE = Path(n4["n3_completion"])
n3 = load_json(N3_FILE)
if n3 is None or n3.get("status") != "TRAINING_ONLY_HIDDEN_CASE_UTILITY_AND_SHORTLIST_COMPLETE":
    raise ValueError("N4 points to an invalid N3 completion.")
if sha256_file(N3_FILE) != n4["n3_completion_sha256"]:
    raise ValueError("N3 completion changed after N4.")
if n3.get("shortlisted_for_equal_budget_nested_training_only_tuning") != ["official_pulscar"]:
    raise ValueError("The locked N3 shortlist changed.")

N0_FILE = Path(n4["n0_policy_lock"])
n0 = load_json(N0_FILE)
if n0 is None or n0.get("status") != "TRAINING_ONLY_HIGH_MISSINGNESS_POLICY_LOCKED":
    raise ValueError("N4 points to an invalid N0 policy lock.")
if sha256_file(N0_FILE) != n4["n0_policy_lock_sha256"]:
    raise ValueError("N0 policy lock changed after N4.")

SNAPSHOT_FILE = Path(n4["training_snapshot"])
if not SNAPSHOT_FILE.is_file() or sha256_file(SNAPSHOT_FILE) != n4["training_snapshot_sha256"]:
    raise ValueError("The frozen Training snapshot is missing or changed.")
d0 = load_json(n0["source_protocol"])
if d0 is None:
    raise ValueError("The N0 source protocol is unreadable.")
ID_COL = d0["cohort"]["patient_id_column"]
LABEL_COL = d0["cohort"]["observed_label_column"]
FOLD_COL = d0["cohort"]["locked_fold_column"]

for required in [PATIENT_PATH, ENCOUNTER_PATH]:
    if not required.is_file():
        raise FileNotFoundError(required)

training = pd.read_parquet(
    SNAPSHOT_FILE,
    columns=[
        ID_COL, LABEL_COL, FOLD_COL, "DERIVED_EGFR_CKD_EPI_2021",
        "EGFR_PREVIOUS_90D", "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS",
    ],
)
if len(training) != EXPECTED_PATIENTS or training[ID_COL].isna().any() or training[ID_COL].duplicated().any():
    raise ValueError("Unexpected frozen Training cohort.")
labels = pd.to_numeric(training[LABEL_COL], errors="raise").astype(int).to_numpy()
folds = pd.to_numeric(training[FOLD_COL], errors="raise").astype(int).to_numpy()
if int(labels.sum()) != EXPECTED_POSITIVES or int((labels == 0).sum()) != EXPECTED_UNLABELED:
    raise ValueError("Training label counts changed.")
if sorted(np.unique(folds).tolist()) != list(range(1, EXPECTED_FOLDS + 1)):
    raise ValueError("Locked Training folds changed.")
patient_ids = training[ID_COL].to_numpy()
probabilities = {
    method: read_oof(
        N4_PRIVATE / f"{method}__nested_tuned_complete_oof.parquet",
        patient_ids, ID_COL,
    )
    for method in METHODS
}

# Verify exact reproduction of the saved N4 AP values.
n4_summary = pd.read_csv(N4_SUMMARY)
if set(n4_summary["Method"]) != set(METHODS):
    raise ValueError("N4 summary method set changed.")
for method in METHODS:
    saved_ap = float(n4_summary.loc[n4_summary["Method"].eq(method), "Average_precision"].iloc[0])
    reproduced_ap = float(average_precision_score(labels, probabilities[method]))
    if not np.isclose(saved_ap, reproduced_ap, rtol=0.0, atol=1e-12):
        raise ValueError(f"N4 AP cannot be reproduced for {method}.")


# Reconstruct the identical locked kidney-evidence phenotype.
training_keys = normalize_patient_id(training[ID_COL])
if training_keys.isna().any() or training_keys.duplicated().any():
    raise ValueError("Training identifiers do not normalize uniquely.")
current_egfr = pd.to_numeric(training["DERIVED_EGFR_CKD_EPI_2021"], errors="coerce")
previous_egfr = pd.to_numeric(training["EGFR_PREVIOUS_90D"], errors="coerce")
egfr_days = pd.to_numeric(training["EGFR_DAYS_BETWEEN_90D_MEASUREMENTS"], errors="coerce")
persistent_low_egfr = (
    current_egfr.lt(EGFR_G3_THRESHOLD)
    & previous_egfr.lt(EGFR_G3_THRESHOLD)
    & egfr_days.ge(CHRONICITY_MIN_DAYS)
)

patient = pd.read_parquet(PATIENT_PATH, columns=[ID_COL, "PATIENT_INDEX_DATE"])
patient["_key"] = normalize_patient_id(patient[ID_COL])
patient["PATIENT_INDEX_DATE"] = pd.to_datetime(patient["PATIENT_INDEX_DATE"], errors="coerce")
patient = patient.dropna(subset=["_key", "PATIENT_INDEX_DATE"]).drop_duplicates("_key")
index_date_map = patient.set_index("_key")["PATIENT_INDEX_DATE"]
encounters = pd.read_parquet(
    ENCOUNTER_PATH, columns=[ID_COL, "ENCOUNTER_DATE", "UACR_FINAL_MG_MMOL"]
)
encounters["_key"] = normalize_patient_id(encounters[ID_COL])
encounters = encounters[encounters["_key"].isin(set(training_keys))].copy()
encounters["ENCOUNTER_DATE"] = pd.to_datetime(encounters["ENCOUNTER_DATE"], errors="coerce")
encounters["UACR_FINAL_MG_MMOL"] = pd.to_numeric(encounters["UACR_FINAL_MG_MMOL"], errors="coerce")
encounters["PATIENT_INDEX_DATE"] = encounters["_key"].map(index_date_map)
encounters = encounters[
    encounters["ENCOUNTER_DATE"].notna()
    & encounters["PATIENT_INDEX_DATE"].notna()
    & encounters["UACR_FINAL_MG_MMOL"].ge(UACR_A2_THRESHOLD_MG_MMOL)
    & encounters["ENCOUNTER_DATE"].le(encounters["PATIENT_INDEX_DATE"])
    & encounters["ENCOUNTER_DATE"].ge(
        encounters["PATIENT_INDEX_DATE"] - pd.Timedelta(days=UACR_WINDOW_DAYS)
    )
].copy()
uacr_span = encounters.groupby("_key")["ENCOUNTER_DATE"].agg(["min", "max", "count"])
repeated_high_uacr_keys = set(
    uacr_span.index[
        uacr_span["count"].ge(2)
        & (uacr_span["max"] - uacr_span["min"]).dt.days.ge(CHRONICITY_MIN_DAYS)
    ]
)
persistent_high_uacr = training_keys.isin(repeated_high_uacr_keys)
kidney_evidence = persistent_low_egfr.to_numpy(bool) | persistent_high_uacr.to_numpy(bool)
evidence_audit = {
    "persistent_low_egfr_90d": int(persistent_low_egfr.sum()),
    "persistent_high_uacr_90d": int(persistent_high_uacr.sum()),
    "either_computable_kidney_evidence": int(kidney_evidence.sum()),
    "unlabeled_with_either_evidence": int(((labels == 0) & kidney_evidence).sum()),
    "documented_with_either_evidence": int(((labels == 1) & kidney_evidence).sum()),
    "interpretation": (
        "Pre-index computable KDIGO-aligned laboratory concordance only; not "
        "clinical adjudication, verified latent CKD, or a diagnosis."
    ),
    "model_input_overlap_disclosure": (
        "Current eGFR is retained by the model; previous eGFR, interval, and uACR "
        "are excluded. This is supportive concordance, not independent validation."
    ),
}
if evidence_audit["unlabeled_with_either_evidence"] != EXPECTED_UNLABELED_WITH_EVIDENCE:
    raise ValueError("The locked kidney-evidence phenotype changed.")


# Freeze the selection protocol before calculating its outputs.
context = {
    "step": "5.12N5",
    "status": "TRAINING_ONLY_POST_TUNING_PIPELINE_SELECTION",
    "n4_completion": str(N4_FILE),
    "n4_completion_sha256": sha256_file(N4_FILE),
    "n4_summary_sha256": sha256_file(N4_SUMMARY),
    "n4_oof_sha256": {
        method: sha256_file(N4_PRIVATE / f"{method}__nested_tuned_complete_oof.parquet")
        for method in METHODS
    },
    "training_snapshot_sha256": sha256_file(SNAPSHOT_FILE),
    "patient_source_sha256": sha256_file(PATIENT_PATH),
    "encounter_source_sha256": sha256_file(ENCOUNTER_PATH),
    "methods": METHODS,
    "workloads": WORKLOADS,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "selection_gates": {
        "delta_AP_vs_naive_minimum": AP_DELTA_MINIMUM,
        "documented_recall_at_5pct_within_pp_of_naive": RECALL_TOLERANCE_PP,
        "top2pct_unlabeled_kidney_evidence_yield_at_least_naive": True,
    },
    "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
    "validation_previously_accessed_for_superseded_pipeline": True,
    "test_previously_accessed_for_superseded_pipeline": True,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_FILE = OUTPUT_DIR / "context.json"
if CONTEXT_FILE.is_file():
    if load_json(CONTEXT_FILE) != json_safe(context):
        raise ValueError("Existing N5 context is incompatible.")
else:
    atomic_json(CONTEXT_FILE, context)
atomic_json(OUTPUT_DIR / "computable_kidney_evidence_audit.json", evidence_audit)


observed_summary = pd.DataFrame([
    {"Method": method, **metrics(labels, probabilities[method])} for method in METHODS
]).sort_values("Average_precision", ascending=False, kind="stable").reset_index(drop=True)

capture_rows = []
for method in METHODS:
    ranking = np.argsort(-probabilities[method], kind="stable")
    for workload in WORKLOADS:
        count = int(np.ceil(workload * EXPECTED_PATIENTS))
        selected = ranking[:count]
        captured = int(labels[selected].sum())
        capture_rows.append({
            "Method": method,
            "Workload_percent": 100.0 * workload,
            "Patients_reviewed": count,
            "Documented_CKD_captured": captured,
            "Documented_CKD_recall_percent": 100.0 * captured / EXPECTED_POSITIVES,
            "Documented_CKD_precision_percent": 100.0 * captured / count,
        })
documented_capture = pd.DataFrame(capture_rows)

unlabeled_positions = np.flatnonzero(labels == 0)
positive_positions = np.flatnonzero(labels == 1)
utility_rows = []
for method in METHODS:
    ranking = unlabeled_positions[
        np.argsort(-probabilities[method][unlabeled_positions], kind="stable")
    ]
    for workload in WORKLOADS:
        count = int(np.ceil(workload * EXPECTED_UNLABELED))
        selected = ranking[:count]
        evidence_count = int(kidney_evidence[selected].sum())
        utility_rows.append({
            "Method": method,
            "Unlabeled_workload_percent": 100.0 * workload,
            "Unlabeled_reviewed": count,
            "Persistent_low_eGFR_90d": int(persistent_low_egfr.iloc[selected].sum()),
            "Persistent_high_uACR_90d": int(persistent_high_uacr.iloc[selected].sum()),
            "Either_computable_kidney_evidence": evidence_count,
            "Evidence_yield_percent": 100.0 * evidence_count / count,
            "Evidence_recall_percent": 100.0 * evidence_count / EXPECTED_UNLABELED_WITH_EVIDENCE,
        })
hidden_utility = pd.DataFrame(utility_rows)

fold_rows = []
for fold in range(1, EXPECTED_FOLDS + 1):
    fold_unlabeled = np.flatnonzero((folds == fold) & (labels == 0))
    for method in METHODS:
        count = int(np.ceil(PRIMARY_WORKLOAD * len(fold_unlabeled)))
        selected = fold_unlabeled[
            np.argsort(-probabilities[method][fold_unlabeled], kind="stable")[:count]
        ]
        fold_rows.append({
            "Fold": fold,
            "Method": method,
            "Unlabeled_in_fold": len(fold_unlabeled),
            "Reviewed_at_2pct": count,
            "Evidence_count": int(kidney_evidence[selected].sum()),
            "Evidence_yield_percent": 100.0 * float(kidney_evidence[selected].mean()),
        })
fold_utility = pd.DataFrame(fold_rows)


# Paired patient bootstrap of the two decision differences.
rng = np.random.default_rng(SEED)
ap_difference = np.empty(BOOTSTRAP_REPLICATES)
yield_difference = np.empty(BOOTSTRAP_REPLICATES)
for replicate in range(BOOTSTRAP_REPLICATES):
    sampled_positive = rng.choice(positive_positions, len(positive_positions), replace=True)
    sampled_unlabeled = rng.choice(unlabeled_positions, len(unlabeled_positions), replace=True)
    sampled_all = np.concatenate([sampled_positive, sampled_unlabeled])
    ap_difference[replicate] = (
        average_precision_score(labels[sampled_all], probabilities["official_pulscar"][sampled_all])
        - average_precision_score(labels[sampled_all], probabilities["naive_pn"][sampled_all])
    )
    count = int(np.ceil(PRIMARY_WORKLOAD * len(sampled_unlabeled)))
    selected = {}
    for method in METHODS:
        selected[method] = sampled_unlabeled[
            np.argsort(-probabilities[method][sampled_unlabeled], kind="stable")[:count]
        ]
    yield_difference[replicate] = 100.0 * (
        float(kidney_evidence[selected["official_pulscar"]].mean())
        - float(kidney_evidence[selected["naive_pn"]].mean())
    )
    if (replicate + 1) % 100 == 0:
        print(f"Bootstrap {replicate + 1}/{BOOTSTRAP_REPLICATES}")

naive_ap = float(observed_summary.loc[observed_summary["Method"].eq("naive_pn"), "Average_precision"].iloc[0])
pulscar_ap = float(observed_summary.loc[observed_summary["Method"].eq("official_pulscar"), "Average_precision"].iloc[0])
naive_yield = value_at(hidden_utility, "naive_pn", "Unlabeled_workload_percent", PRIMARY_WORKLOAD, "Evidence_yield_percent")
pulscar_yield = value_at(hidden_utility, "official_pulscar", "Unlabeled_workload_percent", PRIMARY_WORKLOAD, "Evidence_yield_percent")
paired_bootstrap = pd.DataFrame([
    {
        "Comparison": "PULSCAR_minus_naive_observed_AP",
        "Point_difference": pulscar_ap - naive_ap,
        "Bootstrap_95CI_lower": float(np.quantile(ap_difference, 0.025)),
        "Bootstrap_95CI_upper": float(np.quantile(ap_difference, 0.975)),
        "Replicates": BOOTSTRAP_REPLICATES,
    },
    {
        "Comparison": "PULSCAR_minus_naive_top2pct_evidence_yield_pp",
        "Point_difference": pulscar_yield - naive_yield,
        "Bootstrap_95CI_lower": float(np.quantile(yield_difference, 0.025)),
        "Bootstrap_95CI_upper": float(np.quantile(yield_difference, 0.975)),
        "Replicates": BOOTSTRAP_REPLICATES,
    },
])


naive_recall = value_at(documented_capture, "naive_pn", "Workload_percent", RECALL_WORKLOAD, "Documented_CKD_recall_percent")
pulscar_recall = value_at(documented_capture, "official_pulscar", "Workload_percent", RECALL_WORKLOAD, "Documented_CKD_recall_percent")
ap_gate = pulscar_ap - naive_ap >= AP_DELTA_MINIMUM
recall_gate = pulscar_recall >= naive_recall - RECALL_TOLERANCE_PP
evidence_gate = pulscar_yield >= naive_yield
passes_all = bool(ap_gate and recall_gate and evidence_gate)
selected_method = "official_pulscar" if passes_all else "naive_pn"

gate_audit = pd.DataFrame([
    {
        "Candidate": "official_pulscar",
        "Delta_observed_AP_vs_naive": pulscar_ap - naive_ap,
        "AP_loss_no_more_than_0_010": ap_gate,
        "Candidate_documented_recall_at_5pct": pulscar_recall,
        "Naive_documented_recall_at_5pct": naive_recall,
        "Documented_recall_within_1pp_of_naive": recall_gate,
        "Candidate_renal_evidence_yield_at_2pct": pulscar_yield,
        "Naive_renal_evidence_yield_at_2pct": naive_yield,
        "Renal_evidence_yield_at_least_naive": evidence_gate,
        "Passes_all_locked_selection_gates": passes_all,
    }
])

decision = {
    "step": "5.12N5",
    "status": "TRAINING_ONLY_PIPELINE_SELECTED_FOR_FULL_TRAINING_REFIT",
    "selected_method": selected_method,
    "candidate_replaces_naive_reference": passes_all,
    "selection_gates": context["selection_gates"],
    "all_locked_gates_passed_by_pulscar": passes_all,
    "bootstrap_role": "supporting uncertainty analysis; not an additional post-hoc gate",
    "kidney_evidence_role": evidence_audit["interpretation"],
    "model_input_overlap_disclosure": evidence_audit["model_input_overlap_disclosure"],
    "selected_pipeline_is_not_yet_a_full_training_fitted_model": True,
    "next_step": (
        f"Tune {selected_method} once on the full Training cohort using the locked "
        "five folds, then fit and hash the final full-Training pipeline before "
        "any new held-out scoring."
    ),
    "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
    "model_fitted_tuned_promoted_or_replaced_in_this_step": False,
}

observed_summary.to_csv(OUTPUT_DIR / "nested_tuned_observed_label_summary.csv", index=False)
documented_capture.to_csv(OUTPUT_DIR / "fixed_workload_documented_capture.csv", index=False)
hidden_utility.to_csv(OUTPUT_DIR / "unlabeled_hidden_case_utility.csv", index=False)
fold_utility.to_csv(OUTPUT_DIR / "foldwise_top2pct_hidden_case_utility.csv", index=False)
paired_bootstrap.to_csv(OUTPUT_DIR / "paired_pulscar_vs_naive_bootstrap.csv", index=False)
gate_audit.to_csv(OUTPUT_DIR / "post_tuning_selection_gate_audit.csv", index=False)
atomic_json(OUTPUT_DIR / "pipeline_selection.json", decision)
atomic_json(
    OUTPUT_DIR / "completion.json",
    {
        **context,
        **decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
    },
)

print("=" * 112)
print("STEP 5.12N5 — TRAINING-ONLY POST-TUNING PIPELINE SELECTION COMPLETE")
print("=" * 112)
print("\nCOMPUTABLE KIDNEY-EVIDENCE AUDIT")
print(json.dumps(evidence_audit, indent=2, ensure_ascii=False))
print("\nNESTED TUNED OBSERVED-LABEL SUMMARY")
display(observed_summary)
print("\nFIXED-WORKLOAD DOCUMENTED CKD CAPTURE")
display(documented_capture)
print("\nUNLABELED HIDDEN-CASE UTILITY")
display(hidden_utility)
print("\nFOLDWISE TOP-2% HIDDEN-CASE UTILITY")
display(fold_utility)
print("\nPAIRED PULSCAR VERSUS NAIVE BOOTSTRAP")
display(paired_bootstrap)
print("\nPOST-TUNING SELECTION GATE AUDIT")
display(gate_audit)
print("\nPIPELINE SELECTION")
print(json.dumps(decision, indent=2, ensure_ascii=False))
print(f"\nSaved run: {OUTPUT_DIR}")
print("Validation accessed in this step: No")
print("Test accessed in this step: No")
print("Model fitted, tuned, or replaced in this step: No")
