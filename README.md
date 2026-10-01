# CKD PULSCAR-XGBoost review-prioritization workflow

This repository accompanies a manuscript describing a positive-unlabeled learning workflow for prioritizing patients for kidney review. Documented CKD records are treated as observed positives; all other records are unlabeled rather than confirmed negatives. PULSCAR estimates the hidden-positive proportion in the unlabeled training population, and a mean-imputed XGBoost model produces a continuous review-priority score.

> The model does not diagnose CKD. Its output is a review-priority score, not a calibrated patient-level probability of CKD. A low score does not rule out CKD.

## Repository contents

- `repro/`: the journal-facing notebook and exact numbered analysis scripts.
- `src/ckd_pulscar_xgb/`: reusable label-promotion, preprocessing, XGBoost fitting, and scoring functions.
- `examples/`: a fully synthetic interface demonstration containing no patient data.
- `docs/schema.csv`: names, units, and definitions for the 20 predictors.
- `results/`: aggregate tables and the completed run manifest; no patient-level scores.

## Quick start with synthetic data

Create an isolated Python environment, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python examples/make_demo.py
python examples/run_demo.py
```

The synthetic example uses a clearly marked demonstration score in place of official PULSCAR output. It verifies the public interface only and must not be interpreted as a clinical experiment or as reproduction of the reported results.

## Using official PULSCAR

The alpha estimate and record-level ranking scores should be generated with the official [PULSNAR repository](https://github.com/unmtransinfo/PULSNAR), using its PULSCAR mode (`scar=True`). Pass those scores and alpha to `fit_xgboost_after_pulscar`.

The study reproduction used a privately archived, hash-checked copy of the official wheel. That wheel is not redistributed here. CatBoost appears in `requirements.txt` only because the official PULSNAR package imports it as a software dependency; CatBoost was not trained or evaluated as a candidate model in the reported study. LightGBM is not used.

Minimal reusable pattern:

```python
from ckd_pulscar_xgb import fit_xgboost_after_pulscar, predict_review_priority

fitted = fit_xgboost_after_pulscar(
    training_frame,
    observed_labels=training_frame["OBSERVED_CKD"],
    pulscar_scores=pulscar_record_scores,
    alpha=pulscar_alpha,
)
scores = predict_review_priority(fitted, new_predictor_frame)
```

The function promotes `floor(alpha * N_unlabeled + 0.5)` top-ranked unlabeled records using deterministic hash-based tie breaking, fits mean imputation using training data only, and then fits the frozen XGBoost configuration reported in the manuscript.

## Study reproduction

Open `repro/reproduce.ipynb`. Its default `specification_check` mode checks the frozen study contract without requiring patient data. An authorized full rerun additionally requires the deidentified frozen inputs, software lock, and archived official PULSCAR wheel described inside the notebook.

The completed aggregate reproduction confirmed:

| Cohort | Average precision | ROC AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Validation | 0.890622 | 0.975408 | 0.019340 | 0.077229 |
| Locked internal test | 0.881898 | 0.968680 | 0.018833 | 0.079130 |

PULSCAR reproduced `alpha = 0.0041` and selected 265 probable-positive training records. Confidence intervals and verification details are provided in `results/`.

## Data availability and privacy

No patient-level data, patient identifiers, patient-level scores, fitted model artifacts, or clinician responses are included. The original data cannot be publicly released because of institutional governance and privacy restrictions. The synthetic example is generated independently and does not reproduce real patients.

Do not commit any of the following:

- `data/`, `private/`, or `restricted/` directories;
- `.parquet`, `.joblib`, or archived `.whl` files;
- patient-level scores, linkage files, or patient keys;
- the private input-preparation or authorized-Colab notebooks.

## Citation

Update the author names, manuscript citation, and GitHub URL in `CITATION.cff` before the first public release. After creating a GitHub release, archive it with Zenodo and cite the resulting DOI in the manuscript.

Suggested code-availability statement:

> The code used to reproduce the PULSCAR-XGBoost training and evaluation workflow, together with the analysis configuration, aggregate results, and a synthetic usage example, is available at https://github.com/KholoudOwais/Hidden-CKD-review-prioritization-with-PULSCAR-and-XGBoost. Patient-level data and fitted model artifacts are not publicly shared because of institutional data-governance and privacy restrictions. A permanent archival DOI will be added after the first versioned release is deposited in Zenodo.

## License

The repository is prepared under the MIT License. The official PULSNAR software remains subject to its own upstream license and is not redistributed in this repository.
