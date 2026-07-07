from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 129

DEFAULT_DROP_COLUMNS = {
    "Unnamed: 0",
    "start_time",
    "end_time",
    "start_unix",
    "end_unix",
    "unixTime",
    "unixStart",
    "recordTime",
    "move_start_time",
}


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    best_params: dict
    cv_score: float
    holdout_metrics: dict
    confusion_matrix: list[list[int]]


def load_mean_feature_dataset(
    adb_csv: str | Path,
    non_adb_csv: str | Path,
    start_end_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load ADB/non-ADB mean HRV tables and add labels/groups.

    The historical notebooks sliced with ``iloc[:, 5:]``. This loader keeps
    named columns instead, labels rows explicitly, and optionally infers a
    driver group from the interval metadata in ``start_end_csv``.
    """

    adb = pd.read_csv(adb_csv).copy()
    non_adb = pd.read_csv(non_adb_csv).copy()
    adb["adb"] = 1
    non_adb["adb"] = 0
    data = pd.concat([adb, non_adb], ignore_index=True)

    if start_end_csv is not None and Path(start_end_csv).exists():
        data = add_driver_groups(data, pd.read_csv(start_end_csv))

    return data


def add_driver_groups(data: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Infer driver IDs from timestamp intervals where possible."""

    if "start_unix" not in data.columns:
        return data

    intervals = intervals.copy()
    for column in ("ecg_start", "ecg_end", "trip_start", "trip_end"):
        if column in intervals.columns:
            intervals[column] = pd.to_datetime(
                intervals[column], dayfirst=True, errors="coerce"
            )

    if {"driver", "ecg_start", "ecg_end"}.issubset(intervals.columns):
        unix_start = intervals["ecg_start"].astype("int64") // 10**9
        unix_end = intervals["ecg_end"].astype("int64") // 10**9
    elif {"driver", "trip_start", "trip_end"}.issubset(intervals.columns):
        unix_start = intervals["trip_start"].astype("int64") // 10**9
        unix_end = intervals["trip_end"].astype("int64") // 10**9
    else:
        return data

    data = data.copy()
    data["driver"] = np.nan
    timestamps = pd.to_numeric(data["start_unix"], errors="coerce")
    for driver, start, end in zip(intervals["driver"], unix_start, unix_end):
        if pd.isna(start) or pd.isna(end):
            continue
        mask = timestamps.between(start, end, inclusive="both")
        data.loc[mask, "driver"] = driver

    return data


def prepare_features(
    data: pd.DataFrame,
    target: str = "adb",
    group_column: str = "driver",
    drop_columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Return leakage-filtered numeric features, labels, and optional groups."""

    clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    y = clean[target].astype(int)
    groups = clean[group_column] if group_column in clean.columns else None

    columns_to_drop = set(drop_columns) | {target}
    if group_column in clean.columns:
        columns_to_drop.add(group_column)

    features = clean.drop(columns=[c for c in columns_to_drop if c in clean.columns])
    features = features.select_dtypes(include=[np.number])
    features = features.dropna(axis=1, how="all")

    row_mask = features.notna().any(axis=1)
    features = features.loc[row_mask]
    y = y.loc[row_mask]
    if groups is not None:
        groups = groups.loc[row_mask]
        if groups.isna().any() or groups.nunique() < 2:
            groups = None

    return features, y, groups


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    test_size: float = 0.2,
):
    """Create a holdout split without mixing drivers when groups are available."""

    if groups is not None and groups.nunique() >= 3:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(splitter.split(x, y, groups))
        return (
            x.iloc[train_idx],
            x.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            groups.iloc[train_idx],
            groups.iloc[test_idx],
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return x_train, x_test, y_train, y_test, None, None


def make_pipeline(estimator, use_smote: bool = True, scale: bool = False) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[("numeric", Pipeline(numeric_steps), selector_all_columns)],
        remainder="drop",
    )

    steps = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def selector_all_columns(x: pd.DataFrame) -> list[str]:
    return list(x.columns)


def candidate_models(use_smote: bool = True) -> dict[str, tuple[Pipeline, dict]]:
    """Conservative model grids suitable for a small biomedical dataset."""

    models = {
        "logistic_regression": (
            make_pipeline(
                LogisticRegression(max_iter=2000, class_weight="balanced"),
                use_smote=use_smote,
                scale=True,
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            make_pipeline(
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
                use_smote=use_smote,
            ),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [2, 4, None],
                "model__min_samples_leaf": [1, 5, 10],
            },
        ),
        "gradient_boosting": (
            make_pipeline(
                GradientBoostingClassifier(random_state=RANDOM_STATE),
                use_smote=use_smote,
            ),
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__subsample": [0.7, 1.0],
            },
        ),
        "svm_rbf": (
            make_pipeline(
                SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
                use_smote=use_smote,
                scale=True,
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", "auto"],
            },
        ),
    }

    try:
        from xgboost import XGBClassifier
    except ImportError:
        pass
    else:
        models["xgboost"] = (
            make_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_smote=use_smote,
            ),
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__subsample": [0.7, 1.0],
                "model__colsample_bytree": [0.7, 1.0],
                "model__reg_lambda": [1.0, 5.0],
            },
        )

    return models


def evaluate_models(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    use_smote: bool = True,
) -> list[EvaluationResult]:
    x_train, x_test, y_train, y_test, train_groups, _ = split_train_test(x, y, groups)

    if train_groups is not None and train_groups.nunique() >= 3:
        n_splits = min(5, train_groups.nunique())
        cv = GroupKFold(n_splits=n_splits)
        cv_groups = train_groups
    else:
        minority_count = int(y_train.value_counts().min())
        n_splits = max(2, min(5, minority_count))
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        cv_groups = None

    results: list[EvaluationResult] = []
    for name, (pipeline, grid) in candidate_models(use_smote=use_smote).items():
        search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train, groups=cv_groups)
        fitted = search.best_estimator_
        predictions = fitted.predict(x_test)
        probabilities = _positive_class_scores(fitted, x_test)
        metrics = compute_metrics(y_test, predictions, probabilities)
        results.append(
            EvaluationResult(
                model_name=name,
                best_params=search.best_params_,
                cv_score=float(search.best_score_),
                holdout_metrics=metrics,
                confusion_matrix=confusion_matrix(y_test, predictions).tolist(),
            )
        )

    return results


def _positive_class_scores(model, x_test: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return None


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and y_true.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return metrics
