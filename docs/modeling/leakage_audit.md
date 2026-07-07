# Leakage Audit

The historical modelling notebooks are useful as exploratory work, but their reported scores should not be treated as final model performance.

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
  --adb-csv data/raw/final_adb_means.csv \
  --non-adb-csv data/raw/final_non_adb_means.csv \
  --start-end-csv data/processed/start_end_data.csv
```

The new implementation:

- drops timestamp/index columns before training;
- infers driver groups from `start_end_data.csv` when possible;
- uses grouped train/test and grouped cross-validation when driver IDs are available;
- keeps `SMOTE` inside the imbalanced-learn pipeline, so it is fitted only on each training fold;
- compares logistic regression, random forest, gradient boosting, SVM, and real XGBoost when `xgboost` is installed, using balanced metrics.

If grouped scores are much lower than the old random-split scores, that is evidence that the old evaluation was measuring driver/session memorisation rather than generalisable ADB detection.
