"""Reusable, non-diagnostic components of the reported PULSCAR-XGBoost workflow.

PULSCAR alpha and record-level scores must be produced by the official upstream
PULSNAR implementation. This module performs deterministic top-m promotion,
mean-imputed XGBoost fitting, and outcome-blind scoring.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


NUMERIC_FEATURES = [
    "AGE_AT_INDEX",
    "BMI_MODEL",
    "SBP",
    "DBP",
    "PULSE_FINAL",
    "DERIVED_EGFR_CKD_EPI_2021",
    "BUN_FINAL_MMOL_L",
    "POTASSIUM_FINAL_MMOL_L",
    "SODIUM_FINAL_MMOL_L",
    "HEMOGLOBIN_MODEL_G_L",
    "MCV_MODEL_FL",
    "RDW_CV_MODEL_PERCENT",
    "WBC_MODEL_10E9_L",
    "PLATELETS_MODEL_10E9_L",
    "MPV_MODEL_FL",
    "LYMPHOCYTES_ABSOLUTE_MODEL_10E9_L",
    "EOSINOPHILS_ABSOLUTE_MODEL_10E9_L",
    "EGFR_MIN_24M",
    "EGFR_MEAN_24M",
]
ALL_FEATURES = ["AGE_AT_INDEX", "GENDER", *NUMERIC_FEATURES[1:]]

LOCKED_XGB_PARAMETERS = {
    "n_estimators": 700,
    "learning_rate": 0.04590659413126398,
    "reg_lambda": 13.008281483228425,
    "colsample_bytree": 0.7953856488673938,
    "max_depth": 5,
    "min_child_weight": 5.598456360046426,
    "subsample": 0.6746041940331612,
}


@dataclass
class FittedReviewModel:
    preprocessor: ColumnTransformer
    model: XGBClassifier
    alpha: float
    selected_probable_positive: int


def validate_input_frame(frame: pd.DataFrame, *, require_id: str | None = None) -> None:
    required = set(ALL_FEATURES)
    if require_id:
        required.add(require_id)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if require_id and frame[require_id].astype("string").duplicated().any():
        raise ValueError(f"{require_id} must contain one unique row per patient")


def promote_probable_positives(
    patient_ids,
    observed_labels,
    pulscar_scores,
    alpha: float,
    *,
    tie_salt: str = "full_pulscar",
) -> tuple[np.ndarray, np.ndarray]:
    """Promote round-half-up(alpha * N_unlabeled) highest-scored unlabeled rows."""
    ids = pd.Series(patient_ids, dtype="string").to_numpy()
    labels = np.asarray(observed_labels, dtype=int)
    scores = np.asarray(pulscar_scores, dtype=float)
    if not (len(ids) == len(labels) == len(scores)):
        raise ValueError("ids, labels, and scores must have equal length")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("observed_labels must be binary: 1=observed positive, 0=unlabeled")
    if not 0 <= alpha <= 1 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and within [0, 1]")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("pulscar_scores must be finite and within [0, 1]")

    unlabeled = np.flatnonzero(labels == 0)
    n_select = int(math.floor(alpha * len(unlabeled) + 0.5))
    ranking = pd.DataFrame({
        "position": unlabeled,
        "score": scores[unlabeled],
        "tie": [hashlib.sha256(f"{tie_salt}|{ids[i]}".encode()).hexdigest() for i in unlabeled],
    }).sort_values(["score", "tie"], ascending=[False, True], kind="stable")
    selected = np.zeros(len(labels), dtype=bool)
    selected[ranking.iloc[:n_select]["position"].to_numpy(dtype=int)] = True
    augmented = labels.copy()
    augmented[selected] = 1
    return augmented, selected


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="mean", keep_empty_features=True), NUMERIC_FEATURES),
            (
                "gender",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                ["GENDER"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_xgboost_after_pulscar(
    training: pd.DataFrame,
    observed_labels,
    pulscar_scores,
    alpha: float,
    *,
    patient_id_col: str = "PATIENT_ID",
    random_state: int = 60123,
    sample_weight=None,
) -> FittedReviewModel:
    """Fit the reported mean-imputed XGBoost scorer after PULSCAR promotion."""
    validate_input_frame(training, require_id=patient_id_col)
    augmented, selected = promote_probable_positives(
        training[patient_id_col], observed_labels, pulscar_scores, alpha
    )
    preprocessor = _preprocessor()
    x = preprocessor.fit_transform(training[ALL_FEATURES])
    model = XGBClassifier(
        **LOCKED_XGB_PARAMETERS,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x, augmented, sample_weight=sample_weight)
    return FittedReviewModel(preprocessor, model, float(alpha), int(selected.sum()))


def predict_review_priority(fitted: FittedReviewModel, predictors: pd.DataFrame) -> np.ndarray:
    """Return a ranking score. It is not a calibrated CKD probability or diagnosis."""
    validate_input_frame(predictors)
    x = fitted.preprocessor.transform(predictors[ALL_FEATURES])
    return fitted.model.predict_proba(x)[:, 1]

