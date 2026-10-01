"""Generate non-clinical synthetic rows for interface testing only."""

from pathlib import Path

import numpy as np
import pandas as pd


rng = np.random.default_rng(20261001)
n = 500
egfr = np.clip(rng.normal(82, 27, n), 5, 145)
hidden_signal = -0.055 * (egfr - 60) + 0.018 * (rng.normal(55, 15, n) - 55)
latent = 1 / (1 + np.exp(-hidden_signal))
documented = ((rng.random(n) < latent * 0.18)).astype(int)

frame = pd.DataFrame({
    "PATIENT_ID": [f"SYN-{i:05d}" for i in range(n)],
    "OBSERVED_CKD": documented,
    "AGE_AT_INDEX": np.clip(rng.normal(55, 15, n), 18, 95),
    "GENDER": rng.choice(["Female", "Male"], n),
    "BMI_MODEL": np.clip(rng.normal(28, 6, n), 15, 55),
    "SBP": np.clip(rng.normal(128, 19, n), 75, 220),
    "DBP": np.clip(rng.normal(78, 12, n), 40, 135),
    "PULSE_FINAL": np.clip(rng.normal(78, 15, n), 35, 170),
    "DERIVED_EGFR_CKD_EPI_2021": egfr,
    "BUN_FINAL_MMOL_L": np.clip(rng.normal(5.5, 2.2, n), 1, 25),
    "POTASSIUM_FINAL_MMOL_L": np.clip(rng.normal(4.2, 0.55, n), 2.2, 7),
    "SODIUM_FINAL_MMOL_L": np.clip(rng.normal(138, 4, n), 118, 155),
    "HEMOGLOBIN_MODEL_G_L": np.clip(rng.normal(132, 21, n), 55, 190),
    "MCV_MODEL_FL": np.clip(rng.normal(85, 8, n), 55, 115),
    "RDW_CV_MODEL_PERCENT": np.clip(rng.normal(14, 2.2, n), 10, 28),
    "WBC_MODEL_10E9_L": np.clip(rng.normal(7.5, 2.7, n), 1, 35),
    "PLATELETS_MODEL_10E9_L": np.clip(rng.normal(270, 85, n), 40, 750),
    "MPV_MODEL_FL": np.clip(rng.normal(10.2, 1.2, n), 6, 15),
    "LYMPHOCYTES_ABSOLUTE_MODEL_10E9_L": np.clip(rng.normal(2.1, 0.9, n), 0.1, 8),
    "EOSINOPHILS_ABSOLUTE_MODEL_10E9_L": np.clip(rng.gamma(1.5, 0.15, n), 0, 2),
    "EGFR_MIN_24M": np.clip(egfr - rng.uniform(0, 18, n), 3, 145),
    "EGFR_MEAN_24M": np.clip(egfr + rng.normal(2, 5, n), 4, 145),
})

# Deliberately introduce missingness to demonstrate fold-fitted imputation.
for column in ["BMI_MODEL", "SBP", "PULSE_FINAL", "BUN_FINAL_MMOL_L"]:
    frame.loc[rng.random(n) < 0.18, column] = np.nan

# Demonstration-only ranking values; replace with official PULSCAR outputs.
frame["PULSCAR_SCORE_DEMO"] = np.clip(latent + rng.normal(0, 0.08, n), 0, 1)
output = Path(__file__).with_name("demo.csv")
frame.to_csv(output, index=False)
print(f"Wrote {len(frame):,} synthetic rows to {output}")
