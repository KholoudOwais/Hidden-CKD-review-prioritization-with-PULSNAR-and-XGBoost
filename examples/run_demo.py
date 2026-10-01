"""Interface demonstration; not a reproduction of the clinical results."""

from pathlib import Path

import pandas as pd

from ckd_pulscar_xgb import fit_xgboost_after_pulscar, predict_review_priority


data_file = Path(__file__).with_name("demo.csv")
data = pd.read_csv(data_file)
split = int(len(data) * 0.8)
training = data.iloc[:split].copy()
holdout = data.iloc[split:].copy()

fitted = fit_xgboost_after_pulscar(
    training,
    observed_labels=training["OBSERVED_CKD"],
    pulscar_scores=training["PULSCAR_SCORE_DEMO"],
    alpha=0.05,
)
holdout["REVIEW_PRIORITY_SCORE"] = predict_review_priority(fitted, holdout)
print({
    "training_rows": len(training),
    "selected_probable_positive": fitted.selected_probable_positive,
    "holdout_rows_scored": len(holdout),
})
print(holdout[["PATIENT_ID", "REVIEW_PRIORITY_SCORE"]].head())
print("These scores are for software demonstration only and are not clinical outputs.")
