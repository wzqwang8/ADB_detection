# Leakage Audit

The historical modelling notebooks are useful as exploratory work, but their reported scores should not be treated as final model performance.

**Update:** the fixes below (grouped splits, in-pipeline SMOTE, dropped timestamp
columns) still evaluated tiny, hand-built per-event *mean* tables. At the time
this was written, the original `Final_non_adb_means.csv` was believed lost, so
that path relied on an unrelated reconstructed fallback (see "Recovered Non-ADB
Means" below — it has since turned up and is now tracked at
`data/processed/Final_non_adb_means.csv`). Either way, sample
counts from the per-event mean tables are tiny (~370 ADB rows, ~9.5k non-ADB
rows, one row per hand-picked interval). `scripts/build_five_minute_windows.py`
now builds a real, non-overlapping 5-minute-window dataset directly from the
raw per-driver HRV streams and event log — see
`docs/modeling/five_minute_windows.md` for that pipeline and why it doesn't
reintroduce leakage. Both datasets are available if you want to compare scope
(per-event means vs. fixed 5-minute windows).

## Problems Found

1. `SMOTE` is applied before `train_test_split` in `notebooks/all_trip_data/ML_means.ipynb` and `ML_time_series.ipynb`.
   This leaks information because synthetic samples are generated using neighbours from the whole dataset, including rows that later become test rows.

2. The split is row-random rather than driver- or trip-grouped.
   HRV windows from the same driver and neighbouring time intervals are highly correlated, so random splitting can put near-duplicates in both train and test sets.

3. Timestamp and index-like columns are used as model features.
   Columns such as `Unnamed: 0`, `start_unix`, and `end_unix` can encode participant/session identity and collection order rather than physiology.

4. The notebook labels a `GradientBoostingClassifier` as `Xgboost`.
   That is not the XGBoost library. It is scikit-learn gradient boosting, and the naming makes the results hard to interpret.

5. Hyperparameter tuning is scored mainly by accuracy.
   With imbalanced ADB/non-ADB data, accuracy can hide weak positive-class recall. Balanced accuracy, F1, recall, precision, and ROC-AUC should be reported together.

## Fix Implemented

Use `scripts/evaluate_models.py` for defensible model evaluation:

```bash
PYTHONPATH=src python scripts/evaluate_models.py \
  --adb-csv data/processed/Final_adb_means.csv \
  --non-adb-csv data/processed/Final_non_adb_means.csv \
  --start-end-csv data/processed/start_end_data.csv
```

The new implementation:

- drops timestamp/index columns before training;
- infers driver groups from `start_end_data.csv` when possible;
- aligns ADB and non-ADB files to shared feature columns, preventing missing columns in one class from becoming a label signal;
- uses grouped train/test and grouped cross-validation when driver IDs are available;
- keeps `SMOTE` inside the imbalanced-learn pipeline, so it is fitted only on each training fold;
- compares logistic regression, random forest, gradient boosting, SVM, and real XGBoost when `xgboost` is installed, using balanced metrics.

## Recovered Non-ADB Means

The original `Final_non_adb_means.csv` was initially believed missing from the
backup. It has since been located (originally in `Data example/ALL_TRIP_DATA/`,
now reorganized into `Data example/derived_outputs_legacy/`) — 9,561 rows, same
31-column schema as `Final_adb_means.csv`, including the
merged `ODI-3%`/`CVHRI`/`CEI` sleep metrics — and is now tracked at
`data/processed/Final_non_adb_means.csv`, so the command above works with the
real original data rather than a reconstruction.

`scripts/make_frequency_non_adb.py` (a frequency-only fallback built from
`frequency_data/No_*.csv`) is kept for reference/sanity-checking but is no
longer needed as the primary non-ADB source.

Note that, like `Final_adb_means.csv`, every row here is a *mean over one
hand-picked interval* (5 minutes before/after an event, or a matched baseline),
not a real fixed window — see the caveats in `docs/modeling/five_minute_windows.md`
for why the windowed dataset is the more defensible option for anything beyond
a quick comparison.

If grouped scores are much lower than the old random-split scores, that is evidence that the old evaluation was measuring driver/session memorisation rather than generalisable ADB detection.

## Driver-Grouping Bug Fixed

Testing the recovered `Final_non_adb_means.csv` end-to-end (the "Fix Implemented"
path above had never actually been exercised with real non-ADB data, since that
file was missing until now) surfaced two bugs in
`load_mean_feature_dataset`/`add_driver_groups` that silently defeated the
grouped-split protection described above, falling back to plain stratified
splitting with no warning beyond "No usable driver groups found":

1. `add_driver_groups` was only called on the ADB frame, before concatenation.
   `align_common_columns` then only kept the `driver` column if it existed on
   *both* frames — since the non-ADB frame never had it, the column (and all
   of the ADB rows' correctly-inferred driver ids) was silently dropped.
2. The interval-to-unix-seconds conversion used `.astype("int64") // 10**9`,
   which assumes nanosecond-resolution datetimes. Depending on the pandas
   version, `pd.to_datetime` on these string timestamps can resolve to
   microsecond (or other) precision instead, making every computed interval
   boundary wrong by orders of magnitude and matching zero rows to any driver.

Both are fixed in `src/adb_detection/modeling.py`: driver inference now runs
once on the concatenated table (so both classes get a `driver` value), using
a resolution-agnostic `(timestamp - epoch) // pd.Timedelta("1s")` conversion
(the same approach `scripts/build_five_minute_windows.py` already used for
event timestamps). Re-running the command above now reports "Using grouped
evaluation across N drivers" and matches ~9,700 of 9,919 rows to 22 drivers,
instead of falling back to stratified splitting.
