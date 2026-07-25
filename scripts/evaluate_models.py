from __future__ import annotations

import argparse
import json
from pathlib import Path

from adb_detection.modeling import (
    evaluate_models,
    load_five_minute_windows,
    load_mean_feature_dataset,
    prepare_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ADB classifiers with leakage-aware splits."
    )
    parser.add_argument(
        "--windows-csv",
        help=(
            "Pre-built 5-minute-window table from "
            "scripts/build_five_minute_windows.py. Mutually exclusive with "
            "--adb-csv/--non-adb-csv."
        ),
    )
    parser.add_argument("--adb-csv", help="CSV containing ADB windows.")
    parser.add_argument("--non-adb-csv", help="CSV containing non-ADB windows.")
    parser.add_argument(
        "--start-end-csv",
        default="data/processed/start_end_data.csv",
        help="Optional driver interval metadata used for grouped evaluation "
        "(only used with --adb-csv/--non-adb-csv).",
    )
    parser.add_argument(
        "--output",
        default="reports/model_evaluation.json",
        help="Where to write JSON results.",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE and rely on class weights only.",
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

    if groups is not None:
        print(f"Using grouped evaluation across {groups.nunique()} drivers.")
    else:
        print("No usable driver groups found; falling back to stratified splitting.")

    print(f"Rows: {len(x)} | Features: {x.shape[1]} | Positive rate: {y.mean():.3f}")
    print("Dropped timestamp/index columns and kept only numeric HRV features.")

    results = evaluate_models(x, y, groups=groups, use_smote=not args.no_smote)
    serialisable = [
        {
            "model": r.model_name,
            "best_params": r.best_params,
            "cv_balanced_accuracy": r.cv_score,
            "holdout_metrics": r.holdout_metrics,
            "confusion_matrix": r.confusion_matrix,
        }
        for r in results
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serialisable, indent=2))

    for result in serialisable:
        metrics = result["holdout_metrics"]
        print(
            f"{result['model']}: "
            f"CV bal-acc={result['cv_balanced_accuracy']:.3f}, "
            f"holdout bal-acc={metrics['balanced_accuracy']:.3f}, "
            f"F1={metrics['f1']:.3f}, "
            f"ROC-AUC={metrics.get('roc_auc', float('nan')):.3f}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

