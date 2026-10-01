# Numbered analysis steps

The filenames are intentionally short so the archive extracts reliably on Windows. They correspond to the original locked analysis sequence:

- `N0.py`: training-only feature-policy lock
- `N1.py`: mean versus multiple-imputation screen
- `N2.py`: mean-imputed PU screen
- `N3.py`: hidden-case utility and shortlist
- `N4.py`: nested PULSCAR tuning
- `N5.py`: post-tuning pipeline selection
- `N6.py`: full-training PULSCAR-XGBoost refit and lock
- `N6A.py`: imputation sensitivity
- `N6B.py`: baseline model benchmark
- `N7.py`: outcome-blind validation scoring
- `N8.py`: validation characterization and workload lock
- `N9.py`: validation low-priority safety audit
- `N10.py`: reporting and internal-test protocol freeze
- `N11.py`: outcome-blind internal-test scoring
- `N12.py`: locked internal-test evaluation
- `N13.py`: alpha integrity and eGFR benchmark
- `N13A.py`: temporal eGFR replay
- `N14.py`: archived blinded-review sample step
- `N15.py`: global and local TreeSHAP

Run the applicable study steps in this order. The public reproducibility notebook remains the recommended entry point.
