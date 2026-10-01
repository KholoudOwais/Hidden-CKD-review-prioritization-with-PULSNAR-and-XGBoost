"""STEP 5.12N15 — Locked global/local SHAP and Phase-B reveal package.

Computes exact XGBoost TreeSHAP for the complete locked Internal Test cohort and
for the 15 frozen N14 clinician-review cases. Global analysis retains the full
mean-imputed model representation and separately audits whether each raw value
was originally observed. Clinician-facing local explanations rank only features
that were originally observed for that patient.

This step does not alter Phase A, the model, preprocessing, scores, percentiles,
labels, or the clinician sample. It does not load or display uACR.
"""

from google.colab import drive

drive.mount("/content/drive")

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xgboost as xgb
from IPython.display import display


EXPECTED_TEST_PATIENTS = 14_927
EXPECTED_CASES = 15
EXPECTED_TRANSFORMED_FEATURES = 21
FIGURE_TOP_N = 15
LOCAL_TOP_N = 5
PLOT_SEED = 51_215
SHAP_ADDITIVITY_TOLERANCE = 5e-5

ID_COL = "patient_id"
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
N14_ROOT = ROOT / "05_12N14_locked_blinded_clinician_review_sample"
OUTPUT_ROOT = ROOT / "05_12N15_locked_global_local_shap_and_phase_b"

CLINICAL_LABELS = {
    "AGE_AT_INDEX": "Age at index",
    "BMI_MODEL": "Body mass index",
    "SBP": "Systolic blood pressure",
    "DBP": "Diastolic blood pressure",
    "PULSE_FINAL": "Pulse",
    "DERIVED_EGFR_CKD_EPI_2021": "Current derived eGFR",
    "BUN_FINAL_MMOL_L": "Blood urea nitrogen",
    "POTASSIUM_FINAL_MMOL_L": "Potassium",
    "SODIUM_FINAL_MMOL_L": "Sodium",
    "HEMOGLOBIN_MODEL_G_L": "Hemoglobin",
    "MCV_MODEL_FL": "Mean corpuscular volume",
    "RDW_CV_MODEL_PERCENT": "Red cell distribution width",
    "WBC_MODEL_10E9_L": "White blood cell count",
    "PLATELETS_MODEL_10E9_L": "Platelet count",
    "MPV_MODEL_FL": "Mean platelet volume",
    "LYMPHOCYTES_ABSOLUTE_MODEL_10E9_L": "Absolute lymphocyte count",
    "EOSINOPHILS_ABSOLUTE_MODEL_10E9_L": "Absolute eosinophil count",
    "EGFR_MIN_24M": "Minimum eGFR within the 24-month observation window",
    "EGFR_MEAN_24M": "Mean eGFR within the 24-month observation window",
    "GENDER": "Gender",
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


def patient_key(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        nonmissing = numeric.dropna()
        if (np.abs(nonmissing - np.round(nonmissing)) < 1e-9).all():
            return numeric.round().astype("Int64").astype("string")
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


N6_COMPLETION, n6 = latest_completed(
    N6_ROOT, "FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED"
)
N11_COMPLETION, n11 = latest_completed(
    N11_ROOT, "OUTCOME_BLIND_LOCKED_INTERNAL_TEST_SCORING_COMPLETE"
)
N14_COMPLETION, n14 = latest_completed(
    N14_ROOT, "LOCKED_BLINDED_15_CASE_CLINICIAN_REVIEW_SAMPLE_COMPLETE"
)

N6_RUN = N6_COMPLETION.parent
MODEL_FILE = N6_RUN / "locked_official_pulscar_mean_xgb_model.joblib"
PREPROCESSOR_FILE = N6_RUN / "locked_mean_imputation_preprocessor.joblib"
MANIFEST_FILE = N6_RUN / "locked_transformed_feature_manifest.csv"
SCORE_FILE = Path(n11["score_file"])
TEST_PREDICTORS = Path(n11["internal_test_predictors"])
PHASE_A_JSON = Path(n14["phase_a_json"])
PHASE_A_CSV = Path(n14["phase_a_csv"])
LINKAGE_FILE = Path(n14["private_linkage"])

for required in [
    N6_COMPLETION,
    N11_COMPLETION,
    N14_COMPLETION,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    MANIFEST_FILE,
    SCORE_FILE,
    TEST_PREDICTORS,
    PHASE_A_JSON,
    PHASE_A_CSV,
    LINKAGE_FILE,
]:
    if not required.is_file():
        raise FileNotFoundError(required)

expected_hashes = {
    MODEL_FILE: n6["artifact_hashes"]["model_joblib_sha256"],
    PREPROCESSOR_FILE: n6["artifact_hashes"]["downstream_preprocessor_sha256"],
    MANIFEST_FILE: n6["artifact_hashes"]["feature_manifest_sha256"],
    SCORE_FILE: n11["score_file_sha256"],
    PHASE_A_JSON: n14["phase_a_json_sha256"],
    PHASE_A_CSV: n14["phase_a_csv_sha256"],
    LINKAGE_FILE: n14["private_linkage_sha256"],
}
for path, expected in expected_hashes.items():
    if sha256_file(path) != expected:
        raise ValueError(f"Locked artifact changed: {path}")
if n6.get("uacr_used_anywhere_in_modeling") is not False:
    raise ValueError("N6 reports uACR use.")
if n14.get("patient_identifiers_in_phase_a_casebook") is not False:
    raise ValueError("N14 reports patient identifiers in Phase A.")
if n14.get("model_results_in_phase_a_casebook") is not False:
    raise ValueError("N14 reports model results in Phase A.")
if n14.get("model_changed") is not False:
    raise ValueError("N14 reports a model change.")

model = joblib.load(MODEL_FILE)
preprocessor = joblib.load(PREPROCESSOR_FILE)
manifest = pd.read_csv(MANIFEST_FILE)
test = pd.read_parquet(TEST_PREDICTORS)
scores = pd.read_parquet(SCORE_FILE)
phase_a = load_json(PHASE_A_JSON)
linkage = pd.read_parquet(LINKAGE_FILE).sort_values("presentation_order")

if len(test) != EXPECTED_TEST_PATIENTS or test[ID_COL].isna().any():
    raise ValueError("The authoritative Internal Test predictor cohort is invalid.")
if len(linkage) != EXPECTED_CASES or linkage["study_case_code"].nunique() != 15:
    raise ValueError("The N14 private linkage is invalid.")
if not isinstance(phase_a, dict) or len(phase_a.get("cases", [])) != EXPECTED_CASES:
    raise ValueError("The N14 Phase-A JSON is invalid.")
if [case["study_case_code"] for case in phase_a["cases"]] != linkage[
    "study_case_code"
].tolist():
    raise ValueError("Phase-A and private-linkage case order differs.")

test["_PATIENT_KEY"] = patient_key(test[ID_COL])
scores["_PATIENT_KEY"] = patient_key(scores[ID_COL])
linkage["_PATIENT_KEY"] = patient_key(linkage[ID_COL])
if test["_PATIENT_KEY"].duplicated().any():
    raise ValueError("Internal Test predictors contain duplicate patient keys.")

numeric_features = list(preprocessor["numeric_features"])
categorical_feature = str(preprocessor["categorical_feature"])
clinical_features = numeric_features + [categorical_feature]
transformed_features = list(preprocessor["transformed_features"])
if len(clinical_features) != 20 or set(clinical_features) != set(CLINICAL_LABELS):
    raise ValueError("Unexpected locked clinical feature panel.")
if len(transformed_features) != EXPECTED_TRANSFORMED_FEATURES:
    raise ValueError("Unexpected transformed feature count.")
if preprocessor.get("missingness_indicators") is not False:
    raise ValueError("The locked preprocessor unexpectedly uses missingness indicators.")
if preprocessor.get("native_nan_routing") is not False:
    raise ValueError("The locked preprocessor unexpectedly uses native-NaN routing.")
if any("uacr" in feature.lower() for feature in clinical_features):
    raise ValueError("uACR reached SHAP analysis.")

missing_features = sorted(set(clinical_features) - set(test.columns))
if missing_features:
    raise KeyError(f"Internal Test predictors lack: {missing_features}")

raw_clinical = test[clinical_features].copy()
original_observed = raw_clinical.notna().copy()
raw_numeric = (
    raw_clinical[numeric_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)
numeric_matrix = preprocessor["imputer"].transform(raw_numeric).astype(np.float32)
gender_matrix = preprocessor["gender_encoder"].transform(
    raw_clinical[[categorical_feature]].astype("string")
).astype(np.float32)
matrix = np.column_stack([numeric_matrix, gender_matrix]).astype(np.float32)
if matrix.shape != (EXPECTED_TEST_PATIENTS, EXPECTED_TRANSFORMED_FEATURES):
    raise ValueError(f"Unexpected transformed Test shape: {matrix.shape}")
if not np.isfinite(matrix).all():
    raise ValueError("Nonfinite value reached locked XGBoost.")
manifest_names = (
    manifest.sort_values("transformed_position")["transformed_feature"]
    .astype(str)
    .tolist()
)
if manifest_names != transformed_features:
    raise ValueError("Locked feature manifest differs from the preprocessor.")

# Verify exact reproduction of the frozen N11 output before explanation.
recomputed_scores = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
frozen_lookup = scores.set_index("_PATIENT_KEY")[SCORE_COL]
frozen_scores = test["_PATIENT_KEY"].map(frozen_lookup).to_numpy(float)
score_difference = float(np.max(np.abs(recomputed_scores - frozen_scores)))
if score_difference > 1e-7:
    raise ValueError(
        f"Reconstructed scores differ from N11 (max={score_difference:.3g})."
    )

# Exact TreeSHAP. The final contribution column is the model bias term.
booster = model.get_booster()
dmatrix = xgb.DMatrix(matrix, feature_names=transformed_features)
contribution_matrix = np.asarray(
    booster.predict(dmatrix, pred_contribs=True, approx_contribs=False),
    dtype=np.float64,
)
if contribution_matrix.shape != (
    EXPECTED_TEST_PATIENTS,
    EXPECTED_TRANSFORMED_FEATURES + 1,
):
    raise ValueError(f"Unexpected TreeSHAP shape: {contribution_matrix.shape}")
transformed_shap = contribution_matrix[:, :-1]
bias = contribution_matrix[:, -1]
raw_margin = np.asarray(booster.predict(dmatrix, output_margin=True), dtype=float)
additivity_error = float(
    np.max(np.abs(contribution_matrix.sum(axis=1) - raw_margin))
)
if additivity_error > SHAP_ADDITIVITY_TOLERANCE:
    raise ValueError(
        "TreeSHAP additivity exceeded the prespecified float32 tolerance "
        f"({additivity_error:.3g} > {SHAP_ADDITIVITY_TOLERANCE:.3g})."
    )

# Aggregate the two gender one-hot columns to the single clinical variable.
transformed_to_source = []
for name in transformed_features:
    if name.startswith("GENDER_"):
        transformed_to_source.append("GENDER")
    elif name in numeric_features:
        transformed_to_source.append(name)
    else:
        raise ValueError(f"Cannot map transformed feature to source: {name}")
aggregate_indices = {
    feature: [
        index
        for index, source in enumerate(transformed_to_source)
        if source == feature
    ]
    for feature in clinical_features
}
if any(not indices for indices in aggregate_indices.values()):
    raise ValueError("At least one clinical feature has no transformed mapping.")
clinical_shap = np.column_stack(
    [
        transformed_shap[:, aggregate_indices[feature]].sum(axis=1)
        for feature in clinical_features
    ]
)
aggregation_error = float(
    np.max(np.abs(clinical_shap.sum(axis=1) - transformed_shap.sum(axis=1)))
)
if aggregation_error > 1e-10:
    raise ValueError("Clinical-feature aggregation changed total SHAP contribution.")

# Global SHAP importance and observed/imputed-state audit. A contribution in an
# originally missing row reflects the fitted model's response to the Training-
# mean substituted value; it is not a direct missingness effect.
global_rows = []
for feature_index, feature in enumerate(clinical_features):
    values = clinical_shap[:, feature_index]
    observed = original_observed[feature].to_numpy(bool)
    missing = ~observed
    global_rows.append(
        {
            "feature": feature,
            "clinical_label": CLINICAL_LABELS[feature],
            "mean_absolute_shap_all": float(np.mean(np.abs(values))),
            "mean_signed_shap_all": float(np.mean(values)),
            "mean_absolute_shap_when_originally_observed": (
                float(np.mean(np.abs(values[observed]))) if observed.any() else np.nan
            ),
            "mean_absolute_shap_when_mean_imputed": (
                float(np.mean(np.abs(values[missing]))) if missing.any() else np.nan
            ),
            "originally_observed_patients": int(observed.sum()),
            "originally_observed_percent": float(100.0 * observed.mean()),
            "mean_imputed_patients": int(missing.sum()),
            "mean_imputed_percent": float(100.0 * missing.mean()),
        }
    )
global_summary = pd.DataFrame(global_rows).sort_values(
    "mean_absolute_shap_all", ascending=False
).reset_index(drop=True)
global_summary.insert(
    0,
    "global_importance_rank",
    np.arange(1, len(global_summary) + 1, dtype=int),
)
mapping_audit = pd.DataFrame(
    {
        "transformed_position": np.arange(len(transformed_features)),
        "transformed_feature": transformed_features,
        "source_clinical_feature": transformed_to_source,
        "clinical_label": [CLINICAL_LABELS[x] for x in transformed_to_source],
    }
)

context = {
    "step": "5.12N15",
    "status": "LOCKED_GLOBAL_LOCAL_SHAP_AND_PHASE_B",
    "model_sha256": sha256_file(MODEL_FILE),
    "preprocessor_sha256": sha256_file(PREPROCESSOR_FILE),
    "manifest_sha256": sha256_file(MANIFEST_FILE),
    "n11_score_sha256": sha256_file(SCORE_FILE),
    "n14_phase_a_json_sha256": sha256_file(PHASE_A_JSON),
    "n14_phase_a_csv_sha256": sha256_file(PHASE_A_CSV),
    "n14_linkage_sha256": sha256_file(LINKAGE_FILE),
    "test_predictors_sha256": sha256_file(TEST_PREDICTORS),
    "global_patients": EXPECTED_TEST_PATIENTS,
    "local_cases": EXPECTED_CASES,
    "shap_method": "Exact XGBoost TreeSHAP using pred_contribs",
    "shap_scale": "Raw model margin; not probability points or clinical effect size",
    "shap_additivity_tolerance": SHAP_ADDITIVITY_TOLERANCE,
    "global_missing_policy": (
        "Retain full mean-imputed-model contributions and separately identify "
        "whether each raw feature was originally observed."
    ),
    "local_display_policy": (
        "Rank and show only originally observed clinical features; missing raw "
        "features remain in the private full-model additivity audit."
    ),
    "clinician_terminology_version": "gender-v2",
    "gender_display_label": "Gender",
    "uacr_loaded_or_displayed": False,
    "phase_a_changed": False,
    "model_preprocessing_scores_percentiles_or_sample_changed": False,
}
context_id = hashlib.sha256(json_bytes(context)).hexdigest()[:24]
OUTPUT_DIR = OUTPUT_ROOT / context_id
PRIVATE_DIR = OUTPUT_DIR / "PRIVATE_INTERNAL_DO_NOT_SHARE"
FIGURE_DIR = OUTPUT_DIR / "manuscript_figures"
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
atomic_json(OUTPUT_DIR / "context.json", context)

GLOBAL_SUMMARY = OUTPUT_DIR / "global_shap_observed_imputed_audit.csv"
MAPPING_AUDIT = OUTPUT_DIR / "shap_feature_mapping_audit.csv"
atomic_csv(global_summary, GLOBAL_SUMMARY)
atomic_csv(mapping_audit, MAPPING_AUDIT)

PRIVATE_SHAP = PRIVATE_DIR / "internal_test_exact_treeshap_float32.npz"
np.savez_compressed(
    PRIVATE_SHAP,
    patient_id=test[ID_COL].astype(str).to_numpy(),
    transformed_feature_names=np.asarray(transformed_features, dtype=str),
    clinical_feature_names=np.asarray(clinical_features, dtype=str),
    transformed_shap=transformed_shap.astype(np.float32),
    clinical_shap=clinical_shap.astype(np.float32),
    bias=bias.astype(np.float32),
    raw_margin=raw_margin.astype(np.float32),
)

# Manuscript global mean-absolute-SHAP bar figure.
top = global_summary.head(FIGURE_TOP_N).iloc[::-1]
fig, ax = plt.subplots(figsize=(9.2, 7.0))
ax.barh(
    top["clinical_label"],
    top["mean_absolute_shap_all"],
    color="#176B73",
    edgecolor="white",
)
ax.set_xlabel("Mean absolute SHAP contribution (raw model margin)")
ax.set_ylabel("")
ax.set_title("Global feature influence in the locked internal test cohort")
ax.grid(axis="x", color="#DDE5E7", linewidth=0.8, alpha=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
BAR_PNG = FIGURE_DIR / "global_shap_mean_absolute_top15.png"
BAR_PDF = FIGURE_DIR / "global_shap_mean_absolute_top15.pdf"
fig.savefig(BAR_PNG, dpi=300, bbox_inches="tight")
fig.savefig(BAR_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)

# Beeswarm colors use raw observed values; originally missing values are gray.
plot_values = pd.DataFrame(index=test.index)
for feature in clinical_features:
    if feature == "GENDER":
        plot_values[feature] = (
            raw_clinical[feature]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({"female": 0.0, "male": 1.0})
        )
    else:
        plot_values[feature] = pd.to_numeric(raw_clinical[feature], errors="coerce")

rng = np.random.default_rng(PLOT_SEED)
ranked_features = global_summary.head(FIGURE_TOP_N)["feature"].tolist()
fig, ax = plt.subplots(figsize=(10.5, 8.0))
cmap = plt.get_cmap("coolwarm")
for plot_row, feature in enumerate(reversed(ranked_features)):
    feature_index = clinical_features.index(feature)
    x_values = clinical_shap[:, feature_index]
    feature_values = plot_values[feature].to_numpy(float)
    observed = original_observed[feature].to_numpy(bool) & np.isfinite(feature_values)
    jitter = rng.normal(0.0, 0.105, size=len(x_values))
    y_values = plot_row + np.clip(jitter, -0.32, 0.32)
    if observed.any():
        lower, upper = np.nanpercentile(feature_values[observed], [5, 95])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            colors = np.full(observed.sum(), 0.5)
        else:
            colors = np.clip(
                (feature_values[observed] - lower) / (upper - lower), 0, 1
            )
        ax.scatter(
            x_values[observed],
            y_values[observed],
            c=colors,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=7,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
    missing = ~observed
    if missing.any():
        ax.scatter(
            x_values[missing],
            y_values[missing],
            color="#A7ADB2",
            s=7,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )
ax.axvline(0, color="#5D6770", linewidth=0.9)
ax.set_yticks(np.arange(len(ranked_features)))
ax.set_yticklabels(
    [CLINICAL_LABELS[name] for name in reversed(ranked_features)]
)
ax.set_xlabel("SHAP contribution (raw model margin)")
ax.set_ylabel("")
ax.set_title("Global SHAP distribution in the locked internal test cohort")
ax.grid(axis="x", color="#E1E7E9", linewidth=0.7, alpha=0.8)
ax.set_axisbelow(True)
scalar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
scalar.set_array([])
colorbar = fig.colorbar(scalar, ax=ax, pad=0.015, fraction=0.035)
colorbar.set_label("Observed feature value (low to high)")
missing_handle = Line2D(
    [0],
    [0],
    marker="o",
    color="none",
    markerfacecolor="#A7ADB2",
    markeredgecolor="none",
    markersize=6,
    label="Originally missing; Training-mean substituted",
)
ax.legend(handles=[missing_handle], loc="lower right", frameon=False)
fig.tight_layout()
BEESWARM_PNG = FIGURE_DIR / "global_shap_beeswarm_top15.png"
BEESWARM_PDF = FIGURE_DIR / "global_shap_beeswarm_top15.pdf"
fig.savefig(BEESWARM_PNG, dpi=300, bbox_inches="tight")
fig.savefig(BEESWARM_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)

# Build a value/unit lookup from the immutable Phase-A casebook so Phase B uses
# exactly the same clinician-facing measurement text.
phase_a_lookup = {}
for case in phase_a["cases"]:
    code = case["study_case_code"]
    by_name = {
        item["clinical_measure"]: item
        for item in case["clinical_information"]
    }
    phase_a_lookup[code] = by_name

test_row_by_key = pd.Series(np.arange(len(test)), index=test["_PATIENT_KEY"])
score_by_key = scores.set_index("_PATIENT_KEY")
phase_b_cases = []
local_audit_rows = []
private_local_rows = []

for _, link in linkage.iterrows():
    case_code = str(link["study_case_code"])
    key = str(link["_PATIENT_KEY"])
    if key not in test_row_by_key.index:
        raise KeyError(f"Selected patient for {case_code} is absent from Test.")
    row_index = int(test_row_by_key.loc[key])
    row_contributions = clinical_shap[row_index]
    observed_mask = original_observed.iloc[row_index][clinical_features].to_numpy(bool)
    observed_indices = np.flatnonzero(observed_mask)
    if len(observed_indices) < LOCAL_TOP_N:
        raise ValueError(f"{case_code} has fewer than five observed model features.")
    ordered_observed = observed_indices[
        np.argsort(-np.abs(row_contributions[observed_indices]), kind="stable")
    ]
    displayed_indices = ordered_observed[:LOCAL_TOP_N]
    factors = []
    for factor_rank, feature_index in enumerate(displayed_indices, start=1):
        feature = clinical_features[int(feature_index)]
        label = CLINICAL_LABELS[feature]
        phase_a_item = phase_a_lookup[case_code].get(label)
        if phase_a_item is None:
            raise KeyError(f"Phase-A display value missing for {case_code}/{label}.")
        if phase_a_item["availability"] != "Observed":
            raise ValueError(f"A missing feature entered Phase B: {case_code}/{label}.")
        contribution = float(row_contributions[int(feature_index)])
        if contribution > 0:
            direction = "Moved the model output higher"
        elif contribution < 0:
            direction = "Moved the model output lower"
        else:
            direction = "Had no directional effect on the model output"
        factors.append(
            {
                "rank": factor_rank,
                "clinical_measure": label,
                "displayed_value": phase_a_item["value"],
                "unit": phase_a_item["unit"],
                "direction": direction,
                "raw_margin_contribution": round(contribution, 8),
                "relative_bar_length": round(
                    abs(contribution)
                    / max(abs(row_contributions[displayed_indices]).max(), 1e-12),
                    6,
                ),
            }
        )

    score_row = score_by_key.loc[key]
    raw_output = float(score_row[SCORE_COL])
    percentage = float(score_row[PERCENT_COL])
    percentile = float(score_row[PERCENTILE_COL])
    if not np.isclose(percentage, 100.0 * raw_output, rtol=0, atol=1e-10):
        raise ValueError(f"Score display inconsistency for {case_code}.")
    phase_b_cases.append(
        {
            "study_case_code": case_code,
            "model_result": {
                "model_name": "Official PULSCAR + mean-imputed XGBoost",
                "raw_positive_class_output": round(raw_output, 8),
                "model_output_out_of_100": round(percentage, 4),
                "within_cohort_percentile": round(percentile, 4),
                "interpretation": (
                    "The model output is an uncalibrated ranking score. It is "
                    "not the patient's clinical probability of CKD, a diagnosis, "
                    "or a KDIGO stage."
                ),
            },
            "five_observed_clinical_factors_with_strongest_model_influence": factors,
            "explanation_note": (
                "Only factors with an originally observed value are displayed. "
                "The five shown factors are a display subset and do not form the "
                "complete additive model explanation."
            ),
        }
    )

    unfiltered_top = np.argsort(-np.abs(row_contributions), kind="stable")[:LOCAL_TOP_N]
    local_audit_rows.append(
        {
            "study_case_code": case_code,
            "observed_clinical_features": int(observed_mask.sum()),
            "displayed_factor_count": len(factors),
            "displayed_missing_factor_count": 0,
            "unfiltered_top5_mean_imputed_factor_count": int(
                (~observed_mask[unfiltered_top]).sum()
            ),
            "full_additivity_error": float(
                abs(bias[row_index] + row_contributions.sum() - raw_margin[row_index])
            ),
        }
    )
    for feature_index, feature in enumerate(clinical_features):
        private_local_rows.append(
            {
                "study_case_code": case_code,
                ID_COL: link[ID_COL],
                "feature": feature,
                "originally_observed": bool(observed_mask[feature_index]),
                "raw_margin_contribution": float(row_contributions[feature_index]),
            }
        )

phase_b_questions = [
    {
        "question_id": "B1",
        "question": "Does the model result appear clinically plausible for this case?",
        "response_options": ["Yes", "Partly", "No", "Uncertain"],
    },
    {
        "question_id": "B2",
        "question": (
            "Are the displayed clinical factors and their directions clinically coherent?"
        ),
        "response_options": ["Yes", "Partly", "No", "Uncertain"],
    },
    {
        "question_id": "B3",
        "question": (
            "After seeing the model result, would you change your recommendation "
            "for additional record review or repeat testing?"
        ),
        "response_options": [
            "More likely to recommend",
            "No change",
            "Less likely to recommend",
            "Uncertain",
        ],
    },
    {
        "question_id": "B4",
        "question": "Optional comments about the model explanation",
        "response_options": ["Free text"],
    },
]

phase_b_bundle = {
    "study": "Clinical review of potential undocumented chronic kidney disease",
    "phase": "B — Model result revealed",
    "instructions": (
        "Complete this phase only after the corresponding Phase-A responses "
        "have been locked. Review the model output and the five strongest "
        "originally observed clinical factors."
    ),
    "case_count": EXPECTED_CASES,
    "local_explanation_method": (
        "Exact XGBoost TreeSHAP on the locked downstream model, reported on the "
        "raw-margin scale."
    ),
    "cases": phase_b_cases,
    "questions": phase_b_questions,
}
PHASE_B_JSON = OUTPUT_DIR / "phase_b_model_reveal_15_cases.json"
atomic_json(PHASE_B_JSON, phase_b_bundle)

serialized = json.dumps(phase_b_bundle, ensure_ascii=False).lower()
for forbidden in [
    "patient_id",
    "observed_ckd_label",
    "documented_ckd",
    "sampling_role",
    "persistent_low",
    "uacr",
    "high/moderate/low",
]:
    if forbidden in serialized:
        raise ValueError(f"Phase-B safety audit failed: {forbidden}")
if [case["study_case_code"] for case in phase_b_cases] != [
    case["study_case_code"] for case in phase_a["cases"]
]:
    raise ValueError("Phase-B case order differs from Phase A.")

local_audit = pd.DataFrame(local_audit_rows)
if int(local_audit["displayed_missing_factor_count"].sum()) != 0:
    raise ValueError("A missing feature entered a clinician-facing explanation.")
if not local_audit["displayed_factor_count"].eq(LOCAL_TOP_N).all():
    raise ValueError("At least one case lacks five displayed observed factors.")
LOCAL_AUDIT = OUTPUT_DIR / "local_observed_only_shap_safety_audit.csv"
PRIVATE_LOCAL = PRIVATE_DIR / "selected_cases_full_shap_audit.parquet"
atomic_csv(local_audit, LOCAL_AUDIT)
atomic_parquet(pd.DataFrame(private_local_rows), PRIVATE_LOCAL)

top15_print = global_summary.head(FIGURE_TOP_N)[
    [
        "global_importance_rank",
        "clinical_label",
        "mean_absolute_shap_all",
        "mean_absolute_shap_when_originally_observed",
        "mean_absolute_shap_when_mean_imputed",
        "originally_observed_percent",
    ]
]

completion = {
    **context,
    "status": "LOCKED_GLOBAL_LOCAL_SHAP_AND_PHASE_B_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(OUTPUT_DIR),
    "score_reproduction_max_absolute_difference": score_difference,
    "tree_shap_additivity_max_absolute_error": additivity_error,
    "clinical_aggregation_max_absolute_error": aggregation_error,
    "global_summary_sha256": sha256_file(GLOBAL_SUMMARY),
    "mapping_audit_sha256": sha256_file(MAPPING_AUDIT),
    "private_exact_shap_sha256": sha256_file(PRIVATE_SHAP),
    "global_bar_png_sha256": sha256_file(BAR_PNG),
    "global_bar_pdf_sha256": sha256_file(BAR_PDF),
    "global_beeswarm_png_sha256": sha256_file(BEESWARM_PNG),
    "global_beeswarm_pdf_sha256": sha256_file(BEESWARM_PDF),
    "phase_b_json": str(PHASE_B_JSON),
    "phase_b_json_sha256": sha256_file(PHASE_B_JSON),
    "local_safety_audit_sha256": sha256_file(LOCAL_AUDIT),
    "private_local_full_audit_sha256": sha256_file(PRIVATE_LOCAL),
    "displayed_missing_factors": int(
        local_audit["displayed_missing_factor_count"].sum()
    ),
    "cases_with_five_displayed_observed_factors": int(
        local_audit["displayed_factor_count"].eq(LOCAL_TOP_N).sum()
    ),
    "model_changed": False,
    "phase_a_changed": False,
    "next_step": (
        "Build the two-phase clinician-review site/package from the immutable "
        "N14 Phase-A JSON and N15 Phase-B JSON, preserving response locking."
    ),
}
atomic_json(OUTPUT_DIR / "completion.json", completion)

print("=" * 116)
print("STEP 5.12N15 — LOCKED GLOBAL/LOCAL SHAP AND PHASE-B REVEAL COMPLETE")
print("=" * 116)
print("\nGLOBAL SHAP — TOP 15")
display(top15_print)
print("\nLOCAL OBSERVED-ONLY SAFETY AUDIT")
display(local_audit)
print("\nTECHNICAL CHECKS")
print(
    json.dumps(
        {
            "patients_explained_globally": EXPECTED_TEST_PATIENTS,
            "frozen_cases_explained_locally": EXPECTED_CASES,
            "score_reproduction_max_absolute_difference": score_difference,
            "tree_shap_additivity_max_absolute_error": additivity_error,
            "clinical_aggregation_max_absolute_error": aggregation_error,
            "displayed_missing_factors": 0,
            "cases_with_five_displayed_observed_factors": int(
                local_audit["displayed_factor_count"].eq(LOCAL_TOP_N).sum()
            ),
        },
        indent=2,
    )
)
print(f"\nSaved run: {OUTPUT_DIR}")
print(f"Global SHAP summary: {GLOBAL_SUMMARY}")
print(f"Manuscript bar figure: {BAR_PNG}")
print(f"Manuscript beeswarm figure: {BEESWARM_PNG}")
print(f"Phase-B reveal JSON: {PHASE_B_JSON}")
print(f"Private exact SHAP archive: {PRIVATE_SHAP}")
print("Phase A changed: No")
print("Model, preprocessing, scores, percentiles, or sample changed: No")
print("Missing or mean-imputed factors displayed to clinicians: No")
print("uACR loaded or displayed: No")
