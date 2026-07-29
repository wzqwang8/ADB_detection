from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adb_detection.modeling import load_five_minute_windows
from adb_detection.sequence_modeling import (
    build_lookback_sequences,
    leave_one_group_out_evaluation_gru,
)

METRIC_COLUMNS = ["balanced_accuracy", "f1", "recall", "precision", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-driver-out evaluation of a small GRU sequence model. "
            "Reshapes the existing five_minute_windows*.csv into per-driver "
            "lookback sequences (each driver's K most recent monitored "
            "windows, chronological, however irregular the gaps) instead of "
            "treating each window as an i.i.d. row - no raw signal access "
            "required, since it only reorders/regroups data already built."
        )
    )
    parser.add_argument(
        "--windows-csv",
        default="data/processed/five_minute_windows.csv",
        help="Windows table from scripts/build_five_minute_windows*.py.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=5,
        help="Number of most-recent monitored windows per sequence (including "
        "the window being classified). Each driver loses its first "
        "lookback-1 windows.",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--output",
        default="reports/model_evaluation_logo_sequence.csv",
        help="Where to write the per-driver-fold results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_five_minute_windows(args.windows_csv)
    x, y, groups, feature_names = build_lookback_sequences(dataset, lookback=args.lookback)

    n_drivers = len(set(groups.tolist()))
    print(
        f"Sequences: {len(x)} | Lookback: {args.lookback} | Features/step: {len(feature_names)} | "
        f"Drivers: {n_drivers} | Positive rate: {y.mean():.3f}"
    )
    print(f"(dropped each driver's first {args.lookback - 1} windows for lacking full history)")

    print(f"Running leave-one-driver-out evaluation across {n_drivers} drivers...")
    per_fold = leave_one_group_out_evaluation_gru(
        x,
        y,
        groups,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(output, index=False)

    available_metrics = [c for c in METRIC_COLUMNS if c in per_fold.columns]
    summary = per_fold.groupby("model")[available_metrics].agg(["mean", "std", "min", "max"])
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path)

    pd.set_option("display.width", 160)
    print(summary)
    print(f"Wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
