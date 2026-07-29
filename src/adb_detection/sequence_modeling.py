from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .modeling import DEFAULT_DROP_COLUMNS

RANDOM_STATE = 129


def build_lookback_sequences(
    data: pd.DataFrame,
    lookback: int,
    target: str = "adb",
    group_column: str = "driver",
    time_column: str = "window_start_unix",
    drop_columns=DEFAULT_DROP_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Reshape a flat windows table into per-driver lookback sequences.

    For each window, the sequence is that driver's `lookback` most recently
    monitored windows up to and including itself (chronological order, however
    irregular the gaps — the subsampled negative grid means consecutive rows
    are not evenly spaced), plus a per-step "seconds since previous step"
    feature so the model can tell a 5-minute gap from a 2-hour one. Windows
    without `lookback` prior rows for their driver are dropped (loses each
    driver's first `lookback - 1` windows).

    Uses only columns already present in the existing five_minute_windows*.csv
    tables — no raw signal access required.
    """

    clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    columns_to_drop = set(drop_columns) | {target, group_column, time_column}
    feature_columns = [
        c
        for c in clean.select_dtypes(include=[np.number]).columns
        if c not in columns_to_drop
    ]

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    seq_groups: list[int] = []

    for driver, group in clean.groupby(group_column):
        group = group.sort_values(time_column).reset_index(drop=True)
        feats = group[feature_columns].to_numpy(dtype=float)
        times = group[time_column].to_numpy(dtype=float)
        y = group[target].to_numpy(dtype=int)

        if len(group) < lookback:
            continue

        for end in range(lookback - 1, len(group)):
            start = end - lookback + 1
            window_feats = feats[start : end + 1]
            window_times = times[start : end + 1]
            delta_t = np.diff(window_times, prepend=window_times[0])
            step = np.concatenate([window_feats, delta_t[:, None]], axis=1)
            sequences.append(step)
            labels.append(y[end])
            seq_groups.append(driver)

    x = np.stack(sequences)
    y = np.array(labels)
    groups = np.array(seq_groups)
    return x, y, groups, feature_columns + ["delta_t"]


class GRUClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        h = self.dropout(h[-1])
        return self.fc(h).squeeze(-1)


@dataclass
class SequenceFoldResult:
    held_out_driver: int
    n_test: int
    n_positive_test: int
    balanced_accuracy: float
    f1: float
    recall: float
    precision: float
    roc_auc: float | None


def _fit_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])
    median = np.nanmedian(flat, axis=0)
    median = np.where(np.isnan(median), 0.0, median)
    filled = np.where(np.isnan(flat), median, flat)
    std = filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    mean = filled.mean(axis=0)
    return mean, std, median


def _apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray, median: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(x), median, x)
    return (filled - mean) / std


def train_gru(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    epochs: int = 60,
    batch_size: int = 64,
    seed: int = RANDOM_STATE,
) -> tuple[GRUClassifier, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    torch.manual_seed(seed)
    mean, std, median = _fit_scaler(x_train)
    x_scaled = _apply_scaler(x_train, mean, std, median)

    model = GRUClassifier(n_features=x_train.shape[-1], hidden_size=hidden_size, dropout=dropout)
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    n = len(x_tensor)

    model.train()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_x = x_tensor[idx]
            batch_y = y_tensor[idx]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model, (mean, std, median)


def predict_gru(
    model: GRUClassifier, x: np.ndarray, scaler: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    mean, std, median = scaler
    x_scaled = _apply_scaler(x, mean, std, median)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_scaled, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def leave_one_group_out_evaluation_gru(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    hidden_size: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    epochs: int = 60,
) -> pd.DataFrame:
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    records: list[dict] = []
    for held_out in sorted(set(groups.tolist())):
        train_mask = groups != held_out
        test_mask = groups == held_out

        y_train = y[train_mask]
        if len(set(y_train.tolist())) < 2:
            continue

        model, scaler = train_gru(
            x[train_mask],
            y_train,
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        scores = predict_gru(model, x[test_mask], scaler)
        predictions = (scores >= 0.5).astype(int)
        y_test = y[test_mask]

        record = {
            "model": "gru_sequence",
            "held_out_driver": held_out,
            "n_test": int(test_mask.sum()),
            "n_positive_test": int(y_test.sum()),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "precision": precision_score(y_test, predictions, zero_division=0),
        }
        if len(set(y_test.tolist())) == 2:
            record["roc_auc"] = roc_auc_score(y_test, scores)
        records.append(record)

    return pd.DataFrame.from_records(records)
