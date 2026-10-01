"""STEP 5.12N3 — Training-only hidden-case utility and shortlist.

This step does not fit or tune a model. It consumes only the frozen Training
snapshot and the completed out-of-fold predictions from Step 5.12N2. It never
loads Validation or Test.

Prespecified shortlist gates (all must pass):
1. Observed-label AP loss versus naive PN is no greater than 0.010.
2. Documented-CKD recall at a 5% review workload is within 1 percentage point
   of naive PN.
3. At the primary 2% unlabeled workload, the yield of computable pre-index
   kidney evidence is at least that of naive PN.

The kidney-evidence phenotype is a supporting proxy, not adjudicated latent
CKD: either two eGFR values <60 at least 90 days apart, or two uACR values
>=3 mg/mmol at least 90 days apart within the 730 days before index.
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


SEED = 71_000_043
BOOTSTRAP_REPLICATES = 500
WORKLOADS = [0.02, 0.05, 0.10]
PRIMARY_WORKLOAD = 0.02
DOCUMENTED_RECALL_WORKLOAD = 0.05

EXPECTED_PATIENTS = 69_701
EXPECTED_POSITIVES = 5_042
EXPECTED_UNLABELED = 64_659
EXPECTED_FOLDS = 5
EXPECTED_UNLABELED_WITH_EVIDENCE = 459

METHODS = [
    "naive_pn",
    "standard_bagging_pu",
    "spy_consensus_rn",
    "official_pulscar",
    "official_pulsnar",
]
CANDIDATES = [method for method in METHODS if method != "naive_pn"]

AP_DELTA_MINIMUM = -0.010
RECALL_TOLERANCE_PERCENTAGE_POINTS = 1.0
UACR_WINDOW_DAYS = 730
CHRONICITY_MIN_DAYS = 90
UACR_A2_THRESHOLD_MG_MMOL = 3.0
EGFR_G3_THRESHOLD = 60.0

PROJECT_ROOT = Path("/content/drive/MyDrive/CKD_Project")
ROOT = PROJECT_ROOT / (
    "outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
N2_ROOT = ROOT / "05_12N2_training_only_mean_imputed_five_method_pu_screen"
OUTPUT_ROOT = ROOT / "05_12N3_training_only_hidden_case_utility_and_shortlist"
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
            json.dump(
                json_safe(payload), stream, indent=2, ensure_ascii=False,
                allow_nan=False,
            )
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


def metric_row(method, target, probability):
    clipped = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    return {
        "Method": method,
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
        raise ValueError(
            f"Expected one {value_column} row for {method} at {100*workload:g}%."
        )
    return float(match.iloc[0])


# ---------------------------------------------------------------------------
# Locate and verify the one completed N2 source run selected by recency.
# ---------------------------------------------------------------------------
if not N2_ROOT.exists():
    raise FileNotFoundError("Run Step 5.12N2 before Step 5.12N3.")

n2_candidates = []
for completion_path in N2_ROOT.rglob("completion.json"):
    payload = load_json(completion_path)
    if payload and payload.get("status") == "TRAINING_ONLY_MEAN_IMPUTED_FIVE_METHOD_PU_SCREEN_COMPLETE":
        private_dir = completion_path.parent / "PRIVATE_INTERNAL_DO_NOT_SHARE"
        required_oof = [private_dir / f"{method}__complete_oof.parquet" for method in METHODS]
        if all(path.is_file() for path in required_oof):
            n2_candidates.append((completion_path, payload))
if not n2_candidates:
    raise FileNotFoundError("No complete N2 run with all five OOF vectors was found.")

N2_FILE, n2 = max(n2_candidates, key=lambda item: item[0].stat().st_mtime)
N2_RUN = N2_FILE.parent
N2_PRIVATE = N2_RUN / "PRIVATE_INTERNAL_DO_NOT_SHARE"
N2_SUMMARY_FILE = N2_RUN / "mean_imputed_pu_oof_summary.csv"
if not N2_SUMMARY_FILE.is_file():
    raise FileNotFoundError(N2_SUMMARY_FILE)

if n2.get("validation_accessed_in_this_step") or n2.get("test_accessed_in_this_step"):
    raise ValueError("The selected N2 run reports held-out access.")
if n2.get("imputation_policy_locked") != "mean":
    raise ValueError("The selected N2 run did not lock mean imputation.")
if n2.get("native_nan_routing_used") or n2.get("missingness_indicators_used"):
    raise ValueError("The selected N2 run used a prohibited missingness route.")
if n2.get("methods") != METHODS:
    raise ValueError(f"Unexpected N2 method set/order: {n2.get('methods')}")
if not n2.get("no_method_promoted_yet", False):
    raise ValueError("N2 unexpectedly reports that a method was already promoted.")

N0_FILE = Path(n2["n0_policy_lock"])
n0 = load_json(N0_FILE)
if n0 is None or n0.get("status") != "TRAINING_ONLY_HIGH_MISSINGNESS_POLICY_LOCKED":
    raise ValueError("N2 points to an invalid N0 policy lock.")
if sha256_file(N0_FILE) != n2["n0_policy_lock_sha256"]:
    raise ValueError("The N0 policy lock changed after N2.")

SNAPSHOT_FILE = Path(n2["training_snapshot"])
if not SNAPSHOT_FILE.is_file() or sha256_file(SNAPSHOT_FILE) != n2["training_snapshot_sha256"]:
    raise ValueError("The frozen Training snapshot is missing or changed.")

d0 = load_json(n0["source_protocol"])
if d0 is None:
    raise ValueError("The source protocol recorded by N0 is unreadable.")
ID_COL = d0["cohort"]["patient_id_column"]
LABEL_COL = d0["cohort"]["observed_label_column"]
FOLD_COL = d0["cohort"]["locked_fold_column"]

for required in [PATIENT_PATH, ENCOUNTER_PATH]:
    if not required.is_file():
        raise FileNotFoundError(required)

required_training_columns = [
    ID_COL,
    LABEL_COL,
    FOLD_COL,
    "DERIVED_EGFR_CKD_EPI_2021",
    "EGFR_PREVIOUS_90D",
    "EGFR_DAYS_BETWEEN_90D_MEASUREMENTS",
]
training = pd.read_parquet(SNAPSHOT_FILE, columns=required_training_columns)
if len(training) != EXPECTED_PATIENTS:
    raise ValueError(f"Training row count changed: {len(training):,}")
if training[ID_COL].isna().any() or training[ID_COL].duplicated().any():
    raise ValueError("Training identifiers are missing or duplicated.")

labels = pd.to_numeric(training[LABEL_COL], errors="raise").astype(int).to_numpy()
folds = pd.to_numeric(training[FOLD_COL], errors="raise").astype(int).to_numpy()
if int(labels.sum()) != EXPECTED_POSITIVES or int((labels == 0).sum()) != EXPECTED_UNLABELED:
    raise ValueError("Training label counts changed.")
if sorted(np.unique(folds).tolist()) != list(range(1, EXPECTED_FOLDS + 1)):
    raise ValueError("The five locked Training folds changed.")
patient_ids = training[ID_COL].to_numpy()

probabilities = {
    method: read_oof(
        N2_PRIVATE / f"{method}__complete_oof.parquet", patient_ids, ID_COL
    )
    for method in METHODS
}

# Reproduce the saved N2 AP values from the exact OOF vectors before using them.
n2_summary = pd.read_csv(N2_SUMMARY_FILE)
if set(n2_summary["Method"]) != set(METHODS):
    raise ValueError("The saved N2 summary has an unexpected method set.")
for method in METHODS:
    saved_ap = float(
        n2_summary.loc[n2_summary["Method"].eq(method), "Average_precision"].iloc[0]
    )
    reproduced_ap = float(average_precision_score(labels, probabilities[method]))
    if not np.isclose(saved_ap, reproduced_ap, rtol=0.0, atol=1e-12):
        raise ValueError(f"N2 AP cannot be reproduced for {method}.")


# ---------------------------------------------------------------------------
# Reconstruct the prespecified computable kidney-evidence phenotype.
# These columns are used only for the utility audit, not as N2 model inputs.
# ---------------------------------------------------------------------------
training_keys = normalize_patient_id(training[ID_COL])
if training_keys.isna().any() or training_keys.duplicated().any():
    raise ValueError("Training patient identifiers do not normalize uniquely.")

current_egfr = pd.to_numeric(training["DERIVED_EGFR_CKD_EPI_2021"], errors="coerce")
previous_egfr = pd.to_numeric(training["EGFR_PREVIOUS_90D"], errors="coerce")
egfr_days = pd.to_numeric(
    training["EGFR_DAYS_BETWEEN_90D_MEASUREMENTS"], errors="coerce"
)
persistent_low_egfr = (
    current_egfr.lt(EGFR_G3_THRESHOLD)
    & previous_egfr.lt(EGFR_G3_THRESHOLD)
    & egfr_days.ge(CHRONICITY_MIN_DAYS)
)

patient = pd.read_parquet(PATIENT_PATH, columns=[ID_COL, "PATIENT_INDEX_DATE"])
patient["_key"] = normalize_patient_id(patient[ID_COL])
patient["PATIENT_INDEX_DATE"] = pd.to_datetime(
    patient["PATIENT_INDEX_DATE"], errors="coerce"
)
patient = patient.dropna(subset=["_key", "PATIENT_INDEX_DATE"]).drop_duplicates("_key")
index_date_map = patient.set_index("_key")["PATIENT_INDEX_DATE"]

encounters = pd.read_parquet(
    ENCOUNTER_PATH,
    columns=[ID_COL, "ENCOUNTER_DATE", "UACR_FINAL_MG_MMOL"],
)
encounters["_key"] = normalize_patient_id(encounters[ID_COL])
encounters = encounters[encounters["_key"].isin(set(training_keys))].copy()
encounters["ENCOUNTER_DATE"] = pd.to_datetime(
    encounters["ENCOUNTER_DATE"], errors="coerce"
)
encounters["UACR_FINAL_MG_MMOL"] = pd.to_numeric(
    encounters["UACR_FINAL_MG_MMOL"], errors="coerce"
)
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
uacr_date_span = encounters.groupby("_key")["ENCOUNTER_DATE"].agg(["min", "max", "count"])
repeated_high_uacr_keys = set(
    uacr_date_span.index[
        uacr_date_span["count"].ge(2)
        & (uacr_date_span["max"] - uacr_date_span["min"])
        .dt.days.ge(CHRONICITY_MIN_DAYS)
    ]
)
persistent_high_uacr = training_keys.isin(repeated_high_uacr_keys)
kidney_evidence = (
    persistent_low_egfr.to_numpy(bool) | persistent_high_uacr.to_numpy(bool)
)

evidence_audit = {
    "training_patients": EXPECTED_PATIENTS,
    "persistent_low_egfr_90d": int(persistent_low_egfr.sum()),
    "persistent_high_uacr_90d": int(persistent_high_uacr.sum()),
    "either_computable_kidney_evidence": int(kidney_evidence.sum()),
    "unlabeled_with_either_evidence": int(((labels == 0) & kidney_evidence).sum()),
    "documented_with_either_evidence": int(((labels == 1) & kidney_evidence).sum()),
    "model_input_overlap_disclosure": (
        "Current eGFR is retained by the model; previous eGFR, interval, and uACR "
        "are excluded from the new model. This is supportive clinical concordance, "
        "not independent validation."
    ),
    "interpretation": (
        "Pre-index computable KDIGO-aligned laboratory concordance only; not "
        "clinical adjudication, verified latent CKD, or a diagnosis."
    ),
}
if evidence_audit["unlabeled_with_either_evidence"] != EXPECTED_UNLABELED_WITH_EVIDENCE:
    raise ValueError(
        "The locked computable kidney-evidence phenotype changed: expected "
        f"{EXPECTED_UNLABELED_WITH_EVIDENCE}, found "
        f"{evidence_audit['unlabeled_with_either_evidence']}."
    )


# ---------------------------------------------------------------------------
# Freeze context before looking at utility results.
# ---------------------------------------------------------------------------
context = {
    "step": "5.12N3",
    "status": "TRAINING_ONLY_HIDDEN_CASE_UTILITY_AND_SHORTLIST",
    "n2_completion": str(N2_FILE),
    "n2_completion_sha256": sha256_file(N2_FILE),
    "n2_summary": str(N2_SUMMARY_FILE),
    "n2_summary_sha256": sha256_file(N2_SUMMARY_FILE),
    "n2_oof_sha256": {
        method: sha256_file(N2_PRIVATE / f"{method}__complete_oof.parquet")
        for method in METHODS
    },
    "training_snapshot": str(SNAPSHOT_FILE),
    "training_snapshot_sha256": sha256_file(SNAPSHOT_FILE),
    "patient_source": str(PATIENT_PATH),
    "patient_source_sha256": sha256_file(PATIENT_PATH),
    "encounter_source": str(ENCOUNTER_PATH),
    "encounter_source_sha256": sha256_file(ENCOUNTER_PATH),
    "methods": METHODS,
    "candidates": CANDIDATES,
    "workloads": WORKLOADS,
    "primary_unlabeled_workload": PRIMARY_WORKLOAD,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "ranking_tie_rule": "stable frozen Training row order",
    "shortlist_gates": {
        "delta_AP_vs_naive_minimum": AP_DELTA_MINIMUM,
        "documented_recall_at_5pct_within_pp_of_naive": RECALL_TOLERANCE_PERCENTAGE_POINTS,
        "top2pct_unlabeled_kidney_evidence_yield_at_least_naive": True,
    },
    "selection_prohibited_in_this_step": True,
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
    existing_context = load_json(CONTEXT_FILE)
    if existing_context != json_safe(context):
        raise ValueError("Existing N3 context is incompatible.")
else:
    atomic_json(CONTEXT_FILE, context)
atomic_json(OUTPUT_DIR / "computable_kidney_evidence_audit.json", evidence_audit)


# ---------------------------------------------------------------------------
# Observed-label metrics and fixed-workload documented-label capture.
# ---------------------------------------------------------------------------
observed_summary = pd.DataFrame(
    [metric_row(method, labels, probabilities[method]) for method in METHODS]
).sort_values(
    ["Average_precision", "Log_loss", "Brier_score"],
    ascending=[False, True, True], kind="stable",
).reset_index(drop=True)

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
            "Minimum_selected_score_descriptive_only": float(
                probabilities[method][selected].min()
            ),
        })
documented_capture = pd.DataFrame(capture_rows)


# ---------------------------------------------------------------------------
# Hidden-case utility among originally unlabeled patients.
# ---------------------------------------------------------------------------
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
            "Evidence_recall_percent": (
                100.0 * evidence_count
                / evidence_audit["unlabeled_with_either_evidence"]
            ),
            "Minimum_selected_score_descriptive_only": float(
                probabilities[method][selected].min()
            ),
        })
hidden_utility = pd.DataFrame(utility_rows)

fold_rows = []
for fold in range(1, EXPECTED_FOLDS + 1):
    fold_unlabeled = np.flatnonzero((folds == fold) & (labels == 0))
    for method in METHODS:
        ranking = fold_unlabeled[
            np.argsort(-probabilities[method][fold_unlabeled], kind="stable")
        ]
        count = int(np.ceil(PRIMARY_WORKLOAD * len(fold_unlabeled)))
        selected = ranking[:count]
        evidence_count = int(kidney_evidence[selected].sum())
        fold_rows.append({
            "Fold": fold,
            "Method": method,
            "Unlabeled_in_fold": len(fold_unlabeled),
            "Reviewed_at_2pct": count,
            "Evidence_count": evidence_count,
            "Evidence_yield_percent": 100.0 * evidence_count / count,
        })
fold_utility = pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# Paired patient bootstrap versus naive PN (supporting uncertainty analysis).
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
bootstrap_rows = []
bootstrap_ap = {method: np.empty(BOOTSTRAP_REPLICATES) for method in CANDIDATES}
bootstrap_yield = {method: np.empty(BOOTSTRAP_REPLICATES) for method in CANDIDATES}

for replicate in range(BOOTSTRAP_REPLICATES):
    sampled_positive = rng.choice(positive_positions, len(positive_positions), replace=True)
    sampled_unlabeled = rng.choice(unlabeled_positions, len(unlabeled_positions), replace=True)
    sampled_all = np.concatenate([sampled_positive, sampled_unlabeled])
    naive_ap = average_precision_score(labels[sampled_all], probabilities["naive_pn"][sampled_all])

    count = int(np.ceil(PRIMARY_WORKLOAD * len(sampled_unlabeled)))
    naive_selected = sampled_unlabeled[
        np.argsort(-probabilities["naive_pn"][sampled_unlabeled], kind="stable")[:count]
    ]
    naive_yield = 100.0 * float(kidney_evidence[naive_selected].mean())

    for method in CANDIDATES:
        candidate_ap = average_precision_score(
            labels[sampled_all], probabilities[method][sampled_all]
        )
        bootstrap_ap[method][replicate] = candidate_ap - naive_ap
        selected = sampled_unlabeled[
            np.argsort(-probabilities[method][sampled_unlabeled], kind="stable")[:count]
        ]
        candidate_yield = 100.0 * float(kidney_evidence[selected].mean())
        bootstrap_yield[method][replicate] = candidate_yield - naive_yield

    if (replicate + 1) % 100 == 0:
        print(f"Bootstrap {replicate + 1}/{BOOTSTRAP_REPLICATES}")

naive_ap_point = float(
    observed_summary.loc[observed_summary["Method"].eq("naive_pn"), "Average_precision"].iloc[0]
)
naive_yield_point = value_at(
    hidden_utility, "naive_pn", "Unlabeled_workload_percent",
    PRIMARY_WORKLOAD, "Evidence_yield_percent",
)
for method in CANDIDATES:
    method_ap_point = float(
        observed_summary.loc[observed_summary["Method"].eq(method), "Average_precision"].iloc[0]
    )
    method_yield_point = value_at(
        hidden_utility, method, "Unlabeled_workload_percent",
        PRIMARY_WORKLOAD, "Evidence_yield_percent",
    )
    bootstrap_rows.extend([
        {
            "Method": method,
            "Measure": "Observed_AP_difference_vs_naive",
            "Point_difference": method_ap_point - naive_ap_point,
            "Bootstrap_95CI_lower": float(np.quantile(bootstrap_ap[method], 0.025)),
            "Bootstrap_95CI_upper": float(np.quantile(bootstrap_ap[method], 0.975)),
            "Replicates": BOOTSTRAP_REPLICATES,
        },
        {
            "Method": method,
            "Measure": "Top2pct_evidence_yield_difference_vs_naive_pp",
            "Point_difference": method_yield_point - naive_yield_point,
            "Bootstrap_95CI_lower": float(np.quantile(bootstrap_yield[method], 0.025)),
            "Bootstrap_95CI_upper": float(np.quantile(bootstrap_yield[method], 0.975)),
            "Replicates": BOOTSTRAP_REPLICATES,
        },
    ])
paired_bootstrap = pd.DataFrame(bootstrap_rows)


# ---------------------------------------------------------------------------
# Apply the locked gates. Do not hard-code an expected shortlist.
# ---------------------------------------------------------------------------
naive_recall_5 = value_at(
    documented_capture, "naive_pn", "Workload_percent",
    DOCUMENTED_RECALL_WORKLOAD, "Documented_CKD_recall_percent",
)
gate_rows = []
for method in METHODS:
    method_ap = float(
        observed_summary.loc[observed_summary["Method"].eq(method), "Average_precision"].iloc[0]
    )
    recall_5 = value_at(
        documented_capture, method, "Workload_percent",
        DOCUMENTED_RECALL_WORKLOAD, "Documented_CKD_recall_percent",
    )
    yield_2 = value_at(
        hidden_utility, method, "Unlabeled_workload_percent",
        PRIMARY_WORKLOAD, "Evidence_yield_percent",
    )
    eligible = method != "naive_pn"
    exclusion = "" if eligible else "baseline_not_specialized_PU"
    ap_gate = method_ap - naive_ap_point >= AP_DELTA_MINIMUM
    recall_gate = recall_5 >= naive_recall_5 - RECALL_TOLERANCE_PERCENTAGE_POINTS
    evidence_gate = yield_2 >= naive_yield_point
    gate_rows.append({
        "Method": method,
        "Methodologically_eligible": eligible,
        "Methodological_exclusion": exclusion,
        "Delta_observed_AP_vs_naive": method_ap - naive_ap_point,
        "AP_loss_no_more_than_0_010": ap_gate,
        "Documented_recall_at_5pct": recall_5,
        "Documented_recall_within_1pp_of_naive": recall_gate,
        "Renal_evidence_yield_at_2pct": yield_2,
        "Renal_evidence_yield_at_least_naive": evidence_gate,
        "Passes_all_shortlist_gates": bool(
            eligible and ap_gate and recall_gate and evidence_gate
        ),
    })
gate_audit = pd.DataFrame(gate_rows)
shortlisted = gate_audit.loc[
    gate_audit["Passes_all_shortlist_gates"], "Method"
].tolist()

status = (
    "TRAINING_ONLY_HIDDEN_CASE_UTILITY_AND_SHORTLIST_COMPLETE"
    if shortlisted
    else "TRAINING_ONLY_SHORTLIST_COMPLETE_NO_PU_METHOD_PASSED"
)
decision = {
    "step": "5.12N3",
    "status": status,
    "baseline": "naive_pn",
    "shortlisted_for_equal_budget_nested_training_only_tuning": shortlisted,
    "shortlist_rule": context["shortlist_gates"],
    "shortlist_was_not_hard_coded": True,
    "no_pipeline_selected_in_this_step": True,
    "kidney_evidence_role": evidence_audit["interpretation"],
    "model_input_overlap_disclosure": evidence_audit["model_input_overlap_disclosure"],
    "next_step": (
        "Equal-budget nested Training-only tuning of exactly the shortlisted PU "
        "methods, with naive PN retained as reference."
        if shortlisted else
        "Stop PU redevelopment and document that no specialized PU method passed "
        "the locked Training-only gates; do not inspect Validation or Test."
    ),
    "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
    "model_fitted_tuned_promoted_or_replaced": False,
}

observed_summary.to_csv(OUTPUT_DIR / "observed_label_oof_summary.csv", index=False)
documented_capture.to_csv(OUTPUT_DIR / "fixed_workload_documented_capture.csv", index=False)
hidden_utility.to_csv(OUTPUT_DIR / "unlabeled_hidden_case_utility.csv", index=False)
fold_utility.to_csv(OUTPUT_DIR / "foldwise_top2pct_hidden_case_utility.csv", index=False)
paired_bootstrap.to_csv(OUTPUT_DIR / "paired_bootstrap_vs_naive.csv", index=False)
gate_audit.to_csv(OUTPUT_DIR / "pu_shortlist_gate_audit.csv", index=False)
atomic_json(OUTPUT_DIR / "shortlist_decision.json", decision)
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
print("STEP 5.12N3 — TRAINING-ONLY HIDDEN-CASE UTILITY AND SHORTLIST COMPLETE")
print("=" * 112)
print("\nCOMPUTABLE KIDNEY-EVIDENCE AUDIT")
print(json.dumps(evidence_audit, indent=2, ensure_ascii=False))
print("\nOBSERVED-LABEL OOF SUMMARY")
display(observed_summary)
print("\nFIXED-WORKLOAD DOCUMENTED CKD CAPTURE")
display(documented_capture)
print("\nUNLABELED HIDDEN-CASE UTILITY")
display(hidden_utility)
print("\nFOLDWISE TOP-2% HIDDEN-CASE UTILITY")
display(fold_utility)
print("\nPAIRED BOOTSTRAP VERSUS NAIVE PN")
display(paired_bootstrap)
print("\nSHORTLIST GATE AUDIT")
display(gate_audit)
print("\nSHORTLIST DECISION")
print(json.dumps(decision, indent=2, ensure_ascii=False))
print(f"\nSaved run: {OUTPUT_DIR}")
print("Validation accessed in this step: No")
print("Test accessed in this step: No")
print("Model fitted, tuned, promoted, or replaced: No")
