from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adb_detection.modeling import (
    leave_one_group_out_evaluation,
    load_five_minute_windows,
    load_mean_feature_dataset,
    normalize_within_driver,
    prepare_features,
    select_best_hyperparameters,
)

METRIC_COLUMNS = ["balanced_accuracy", "f1", "recall", "precision", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Robust leave-one-driver-out evaluation. Picks each model's "
            "hyperparameters once via grouped CV over all data, then refits "
            "with those fixed hyperparameters once per held-out driver, giving "
            "a distribution of holdout scores instead of a single split's "
            "point estimate (see docs/modeling/five_minute_windows.md, "
            "'Result instability at this sample size')."
        )
    )
    parser.add_argument(
        "--windows-csv",
        help="Windows table from scripts/build_five_minute_windows*.py. "
        "Mutually exclusive with --adb-csv/--non-adb-csv.",
    )
    parser.add_argument("--adb-csv", help="CSV containing ADB windows.")
    parser.add_argument("--non-adb-csv", help="CSV containing non-ADB windows.")
    parser.add_argument(
        "--start-end-csv",
        default="data/processed/start_end_data.csv",
        help="Optional driver interval metadata (only used with --adb-csv/--non-adb-csv).",
    )
    parser.add_argument(
        "--output",
        default="reports/model_evaluation_logo.csv",
        help="Where to write the per-driver-fold results.",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE and rely on class weights only.",
    )
    parser.add_argument(
        "--per-driver-normalize",
        action="store_true",
        help="Z-score each feature against that driver's own mean/std before "
        "modelling, to remove between-driver baseline differences.",
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Add univariate (ANOVA F-test) SelectKBest to each pipeline, "
        "tuning k (10/20/30/50/all) as a grid-search hyperparameter.",
    )
    args = parser.parse_args()

    if bool(args.windows_csv) == bool(args.adb_csv or args.non_adb_csv):
        raise SystemExit(
            "Pass exactly one of --windows-csv or --adb-csv/--non-adb-csv."
        )
    if bool(args.adb_csv) != bool(args.non_adb_csv):
        raise SystemExit("--adb-csv and --non-adb-csv must be given together.")

    return args


def main() -> None:
    args = parse_args()

    if args.windows_csv:
        dataset = load_five_minute_windows(args.windows_csv)
    else:
        dataset = load_mean_feature_dataset(
            args.adb_csv,
            args.non_adb_csv,
            args.start_end_csv if args.start_end_csv else None,
        )

    x, y, groups = prepare_features(dataset)
    if groups is None:
        raise SystemExit(
            "Leave-one-driver-out evaluation requires driver groups; none were found."
        )

    if args.per_driver_normalize:
        x = normalize_within_driver(x, groups)

    print(
        f"Rows: {len(x)} | Features: {x.shape[1]} | Drivers: {groups.nunique()} | "
        f"Positive rate: {y.mean():.3f}"
        + (" | per-driver normalized" if args.per_driver_normalize else "")
    )

    print("Selecting hyperparameters via grouped CV over all data...")
    best_params = select_best_hyperparameters(
        x,
        y,
        groups,
        use_smote=not args.no_smote,
        use_feature_selection=args.feature_selection,
    )
    for name, params in best_params.items():
        print(f"  {name}: {params}")

    print(f"Running leave-one-driver-out evaluation across {groups.nunique()} drivers...")
    per_fold = leave_one_group_out_evaluation(
        x,
        y,
        groups,
        best_params,
        use_smote=not args.no_smote,
        use_feature_selection=args.feature_selection,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(output, index=False)

    if "error" in per_fold.columns:
        n_errors = per_fold["error"].notna().sum()
        if n_errors:
            print(f"{n_errors} (model, held-out driver) folds raised an error; see {output}.")

    available_metrics = [c for c in METRIC_COLUMNS if c in per_fold.columns]
    summary = per_fold.groupby("model")[available_metrics].agg(["mean", "std", "min", "max"])
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path)

    pd.set_option("display.width", 160)
    print(summary)
    print(f"Wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
