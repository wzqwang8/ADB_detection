# Notebooks

The notebooks are grouped by role:

- `ADB_Intervals-Final.ipynb`: derives ADB event intervals.
- `HRV_data-multiple.ipynb`: processes HR/HRV data across participants.
- `frequency_domain.ipynb`: explores frequency-domain HRV features.
- `Plotting.ipynb`: creates figures for reports and presentations.
- `all_trip_data/`: prepares trip-level modelling data and trains machine-learning models.

Notebook outputs are stripped before commit. Re-run them locally after restoring the raw data described in `../data/README.md`.

For final reported model performance, prefer `../scripts/evaluate_models.py`; the original modelling notebooks contain leakage-prone exploratory splits documented in `../docs/modeling/leakage_audit.md`.
