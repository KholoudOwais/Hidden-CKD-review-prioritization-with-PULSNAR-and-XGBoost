"""STEP 5.12N1 — Training-only mean versus multiple-MICE screen.

Run after Step 5.12N0 in the new redevelopment notebook. The comparison uses
the frozen five Training folds and observed labels only. PULSNAR is deliberately
not run yet so this step isolates the imputation policy. Validation and Test are
not accessed. Completed candidate/fold predictions are cached for safe resume.
"""

from google.colab import drive
drive.mount("/content/drive")

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


EXPECTED_PATIENTS = 69_701
EXPECTED_POSITIVES = 5_042
EXPECTED_UNLABELED = 64_659
EXPECTED_RETAINED_FEATURES = 20
EXPECTED_FOLDS = 5
MICE_IMPUTATIONS = 5
BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 71_000_042
WEIGHT_COL = "TEMPORAL_BALANCING_WEIGHT"

ROOT = Path(
    "/content/drive/MyDrive/CKD_Project/outputs/"
    "section_5_modeling/step_5_9_longitudinal_primary_modeling"
)
N0_ROOT = ROOT / "05_12N0_training_only_feature_availability_and_policy_lock"
OUTPUT_ROOT = ROOT / "05_12N1_training_only_mean_vs_multiple_mice_screen"


LOCKED_XGB_PARAMETERS = {
    "n_estimators": 500,
    "learning_rate": 0.04570147939250174,
    "reg_lambda": 10.204527160227807,
    "colsample_bytree": 0.7400473792766912,
    "max_depth": 5,
    "min_child_weight": 7.323551441473847,
    "subsample": 0.8377920943702721,
}


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def make_model(seed):
    return XGBClassifier(
        **LOCKED_XGB_PARAMETERS,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        missing=np.nan,
        verbosity=0,
    )


def metric_row(name, y_true, prediction):
    clipped = np.clip(np.asarray(prediction, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "Candidate": name,
        "Average_precision": average_precision_score(y_true, clipped),
        "ROC_AUC": roc_auc_score(y_true, clipped),
        "Brier_score": brier_score_loss(y_true, clipped),
        "Log_loss": log_loss(y_true, clipped, labels=[0, 1]),
    }


def stratified_bootstrap_delta(y_true, candidate, reference, replicates, seed):
    y_true = np.asarray(y_true, dtype=int)
    positive = np.flatnonzero(y_true == 1)
    unlabeled = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    deltas = []
    for replicate in range(replicates):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(unlabeled, size=len(unlabeled), replace=True),
            ]
        )
        rng.shuffle(sampled)
        delta = (
            average_precision_score(y_true[sampled], candidate[sampled])
            - average_precision_score(y_true[sampled], reference[sampled])
        )
        deltas.append(delta)
        if (replicate + 1) % 100 == 0:
            print(f"Bootstrap {replicate + 1}/{replicates}")
    return np.asarray(deltas)


# Load the latest completed N0 policy lock.
n0_candidates = []
for path in N0_ROOT.rglob("training_only_feature_policy_lock.json"):
    payload = load_json(path)
    if payload and payload.get("status") == "TRAINING_ONLY_HIGH_MISSINGNESS_POLICY_LOCKED":
        n0_candidates.append((path, payload))
if not n0_candidates:
    raise FileNotFoundError("Run Step 5.12N0 before Step 5.12N1.")

N0_FILE, n0 = max(n0_candidates, key=lambda item: item[0].stat().st_mtime)
SNAPSHOT_FILE = Path(n0["frozen_training_snapshot"])
EXPECTED_SNAPSHOT_HASH = n0["frozen_training_snapshot_sha256"]
FEATURES = list(n0["retained_features"])
EXCLUDED_FEATURES = list(n0["excluded_features"])

if len(FEATURES) != EXPECTED_RETAINED_FEATURES:
    raise ValueError(f"Expected 20 retained features, found {len(FEATURES)}.")
if "GENDER" not in FEATURES:
    raise ValueError("GENDER is missing from the retained feature list.")
if not SNAPSHOT_FILE.is_file():
    raise FileNotFoundError(SNAPSHOT_FILE)
if sha256_file(SNAPSHOT_FILE) != EXPECTED_SNAPSHOT_HASH:
    raise ValueError("Frozen Training snapshot hash changed.")

# Restore identifier, label and fold names from the source protocol recorded by N0.
d0 = load_json(Path(n0["source_protocol"]))
if d0 is None:
    raise ValueError("The source protocol recorded by N0 is unreadable.")
ID_COL = d0["cohort"]["patient_id_column"]
LABEL_COL = d0["cohort"]["observed_label_column"]
FOLD_COL = d0["cohort"]["locked_fold_column"]

training = pd.read_parquet(
    SNAPSHOT_FILE, columns=[ID_COL, LABEL_COL, FOLD_COL, WEIGHT_COL, *FEATURES]
)
if len(training) != EXPECTED_PATIENTS:
    raise ValueError(f"Training row count changed: {len(training):,}")
if training[ID_COL].isna().any() or training[ID_COL].duplicated().any():
    raise ValueError("Training patient identifiers are missing or duplicated.")

y = pd.to_numeric(training[LABEL_COL], errors="raise").astype(int).to_numpy()
if int(y.sum()) != EXPECTED_POSITIVES or int((y == 0).sum()) != EXPECTED_UNLABELED:
    raise ValueError("Training label counts changed.")
weights = pd.to_numeric(training[WEIGHT_COL], errors="raise").to_numpy(float)
if not np.isfinite(weights).all() or np.any(weights <= 0):
    raise ValueError("Temporal balancing weights are invalid.")
weights = weights / weights.mean()

fold_values = sorted(pd.Series(training[FOLD_COL]).dropna().unique().tolist())
if len(fold_values) != EXPECTED_FOLDS:
    raise ValueError(f"Expected five locked folds, found {fold_values}.")

NUMERIC_FEATURES = [feature for feature in FEATURES if feature != "GENDER"]
numeric = training[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
gender = training[["GENDER"]].astype("string")
if gender["GENDER"].isna().any():
    raise ValueError("GENDER unexpectedly contains missing values.")


run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_dir = OUTPUT_ROOT / run_id
private_dir = output_dir / "PRIVATE_INTERNAL_DO_NOT_SHARE"
cache_dir = private_dir / "resume_cache"
cache_dir.mkdir(parents=True, exist_ok=False)

candidates = ["mean_imputation", "multiple_mice_5"]
oof_predictions = {candidate: np.full(len(training), np.nan) for candidate in candidates}
fold_audits = []

for fold_number, fold_value in enumerate(fold_values, start=1):
    evaluation_mask = training[FOLD_COL].eq(fold_value).to_numpy()
    fitting_mask = ~evaluation_mask
    X_fit_num = numeric.loc[fitting_mask].reset_index(drop=True)
    X_eval_num = numeric.loc[evaluation_mask].reset_index(drop=True)
    X_fit_gender = gender.loc[fitting_mask].reset_index(drop=True)
    X_eval_gender = gender.loc[evaluation_mask].reset_index(drop=True)
    y_fit = y[fitting_mask]
    w_fit = weights[fitting_mask]

    gender_encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=np.float32
    )
    fit_gender = gender_encoder.fit_transform(X_fit_gender)
    eval_gender = gender_encoder.transform(X_eval_gender)
    gender_names = gender_encoder.get_feature_names_out(["GENDER"]).tolist()

    # Mean-imputed comparator.
    cache_file = cache_dir / f"mean_imputation_fold_{fold_number}.npz"
    if cache_file.is_file():
        cached = np.load(cache_file)
        prediction = cached["prediction"]
        if len(prediction) != int(evaluation_mask.sum()):
            raise ValueError(f"Invalid cached Mean fold {fold_number}.")
        print(f"Reused mean_imputation, fold {fold_number}.")
    else:
        started = time.time()
        imputer = SimpleImputer(strategy="mean", add_indicator=False)
        fit_num = imputer.fit_transform(X_fit_num).astype(np.float32)
        eval_num = imputer.transform(X_eval_num).astype(np.float32)
        X_fit = np.column_stack([fit_num, fit_gender]).astype(np.float32)
        X_eval = np.column_stack([eval_num, eval_gender]).astype(np.float32)
        if not np.isfinite(X_fit).all() or not np.isfinite(X_eval).all():
            raise ValueError("Mean pipeline passed non-finite values to XGBoost.")
        model = make_model(RANDOM_SEED + fold_number)
        model.fit(X_fit, y_fit, sample_weight=w_fit)
        prediction = model.predict_proba(X_eval)[:, 1]
        np.savez_compressed(cache_file, prediction=prediction)
        print(
            f"Completed mean_imputation, fold {fold_number}, "
            f"{(time.time() - started) / 60:.2f} minutes."
        )
    oof_predictions["mean_imputation"][evaluation_mask] = prediction

    # Proper multiple MICE: five posterior draws, five fitted models, pooled predictions.
    cache_file = cache_dir / f"multiple_mice_5_fold_{fold_number}.npz"
    audit_file = cache_dir / f"multiple_mice_5_fold_{fold_number}_audit.json"
    if cache_file.is_file() and audit_file.is_file():
        cached = np.load(cache_file)
        prediction = cached["prediction"]
        if len(prediction) != int(evaluation_mask.sum()):
            raise ValueError(f"Invalid cached MICE fold {fold_number}.")
        fold_audits.append(load_json(audit_file))
        print(f"Reused multiple_mice_5, fold {fold_number}.")
    else:
        started = time.time()
        imputation_predictions = []
        eval_imputations = []
        train_min = np.nanmin(X_fit_num.to_numpy(float), axis=0)
        train_max = np.nanmax(X_fit_num.to_numpy(float), axis=0)
        if not np.isfinite(train_min).all() or not np.isfinite(train_max).all():
            raise ValueError("A retained numeric feature is entirely missing in a fitting fold.")

        for imputation_index in range(MICE_IMPUTATIONS):
            seed = RANDOM_SEED + fold_number * 100 + imputation_index
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=10,
                sample_posterior=True,
                initial_strategy="mean",
                imputation_order="ascending",
                skip_complete=True,
                n_nearest_features=min(10, len(NUMERIC_FEATURES) - 1),
                min_value=train_min,
                max_value=train_max,
                random_state=seed,
            )
            fit_num = imputer.fit_transform(X_fit_num).astype(np.float32)
            eval_num = imputer.transform(X_eval_num).astype(np.float32)
            X_fit = np.column_stack([fit_num, fit_gender]).astype(np.float32)
            X_eval = np.column_stack([eval_num, eval_gender]).astype(np.float32)
            if not np.isfinite(X_fit).all() or not np.isfinite(X_eval).all():
                raise ValueError("MICE pipeline passed non-finite values to XGBoost.")
            model = make_model(seed)
            model.fit(X_fit, y_fit, sample_weight=w_fit)
            imputation_predictions.append(model.predict_proba(X_eval)[:, 1])
            eval_imputations.append(eval_num)
            print(
                f"  MICE fold {fold_number}, imputation "
                f"{imputation_index + 1}/{MICE_IMPUTATIONS} complete."
            )

        prediction = np.mean(np.vstack(imputation_predictions), axis=0)
        stacked_eval = np.stack(eval_imputations, axis=0)
        original_missing = X_eval_num.isna().to_numpy()
        per_cell_sd = np.std(stacked_eval, axis=0, ddof=1)
        missing_cell_sd = per_cell_sd[original_missing]
        fold_audit = {
            "fold": fold_number,
            "fitting_patients": int(fitting_mask.sum()),
            "evaluation_patients": int(evaluation_mask.sum()),
            "imputations": MICE_IMPUTATIONS,
            "all_transformed_values_finite": True,
            "missing_cells_in_evaluation_fold": int(original_missing.sum()),
            "mean_between_imputation_sd_for_missing_cells": (
                float(missing_cell_sd.mean()) if len(missing_cell_sd) else 0.0
            ),
            "missing_cells_with_nonzero_between_imputation_sd_percent": (
                float(100 * np.mean(missing_cell_sd > 0)) if len(missing_cell_sd) else 100.0
            ),
        }
        if len(missing_cell_sd) and not np.any(missing_cell_sd > 0):
            raise ValueError("Multiple MICE produced no between-imputation variation.")
        np.savez_compressed(cache_file, prediction=prediction)
        atomic_json(audit_file, fold_audit)
        fold_audits.append(fold_audit)
        print(
            f"Completed multiple_mice_5, fold {fold_number}, "
            f"{(time.time() - started) / 60:.2f} minutes."
        )
    oof_predictions["multiple_mice_5"][evaluation_mask] = prediction


for candidate, prediction in oof_predictions.items():
    if not np.isfinite(prediction).all():
        raise ValueError(f"Incomplete OOF predictions for {candidate}.")

summary = pd.DataFrame(
    [metric_row(candidate, y, oof_predictions[candidate]) for candidate in candidates]
)
summary["Delta_AP_vs_mean"] = (
    summary["Average_precision"]
    - float(summary.loc[summary["Candidate"] == "mean_imputation", "Average_precision"].iloc[0])
)

deltas = stratified_bootstrap_delta(
    y,
    oof_predictions["multiple_mice_5"],
    oof_predictions["mean_imputation"],
    BOOTSTRAP_REPLICATES,
    RANDOM_SEED + 9000,
)
bootstrap = pd.DataFrame(
    [
        {
            "Candidate": "multiple_mice_5",
            "Reference": "mean_imputation",
            "Delta_AP": float(
                average_precision_score(y, oof_predictions["multiple_mice_5"])
                - average_precision_score(y, oof_predictions["mean_imputation"])
            ),
            "Bootstrap_95CI_lower": float(np.quantile(deltas, 0.025)),
            "Bootstrap_95CI_upper": float(np.quantile(deltas, 0.975)),
            "Replicates": BOOTSTRAP_REPLICATES,
        }
    ]
)

summary.to_csv(output_dir / "training_oof_imputation_summary.csv", index=False)
bootstrap.to_csv(output_dir / "paired_bootstrap_ap_difference.csv", index=False)
pd.DataFrame(fold_audits).to_csv(output_dir / "multiple_mice_fold_audit.csv", index=False)

private_oof = training[[ID_COL, LABEL_COL, FOLD_COL]].copy()
private_oof["mean_imputation_oof_score"] = oof_predictions["mean_imputation"]
private_oof["multiple_mice_5_oof_score"] = oof_predictions["multiple_mice_5"]
private_oof.to_parquet(private_dir / "training_oof_imputation_scores.parquet", index=False)

completion = {
    "step": "5.12N1",
    "status": "TRAINING_ONLY_MEAN_VS_MULTIPLE_MICE_SCREEN_COMPLETE",
    "source_policy_lock": str(N0_FILE),
    "training_patients": int(len(training)),
    "retained_features": FEATURES,
    "excluded_features": EXCLUDED_FEATURES,
    "candidate_policies": {
        "mean_imputation": "fold-fitted single mean imputation; benchmark only",
        "multiple_mice_5": (
            "five fold-fitted Bayesian MICE posterior draws; five XGBoost models; "
            "mean pooled prediction"
        ),
    },
    "engine": "XGBoost fixed at the previously locked parameter values",
    "training_weight_policy": (
        "Previously locked temporal balancing weights, normalized to mean one; "
        "applied identically to both imputation candidates"
    ),
    "labels_for_this_screen": "documented CKD versus unlabeled; PU not yet applied",
    "automatic_winner_selected": False,
    "selection_note": (
        "Review predictive metrics, paired interval, runtime, and MICE variation "
        "before locking the imputation policy."
    ),
    "validation_accessed_in_this_step": False,
    "test_accessed_in_this_step": False,
    "model_promoted_or_replaced": False,
}
atomic_json(output_dir / "completion.json", completion)

print("=" * 108)
print("STEP 5.12N1 — TRAINING-ONLY MEAN VERSUS MULTIPLE-MICE SCREEN COMPLETE")
print("=" * 108)
print("\nOOF IMPUTATION SUMMARY")
display(summary)
print("\nPAIRED BOOTSTRAP AP DIFFERENCE")
display(bootstrap)
print("\nMULTIPLE-MICE TECHNICAL AUDIT")
display(pd.DataFrame(fold_audits))
print("\nSaved run:", output_dir)
print("Validation accessed: No")
print("Test accessed: No")
print("PULSNAR run in this step: No")
print("Model promoted or replaced: No")
print("Next: lock the clinically acceptable imputation policy, then rerun the Training-only PU screen.")
