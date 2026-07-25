from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Columns present in both five_minute_windows.csv and five_minute_windows_ecg.csv
# (metadata + sleep summary) that would otherwise collide on merge. Keep the
# aggregate table's copy and drop the ECG table's duplicate before joining.
SHARED_METADATA_COLUMNS = ["adb", "event_count", "n_samples", "ODI-3%", "CVHRI", "CEI"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the aggregate (mean/std/min/max of the 30s-sliding stream) "
            "and raw-ECG-recomputed feature sets for the same 5-minute windows "
            "into one richer table."
        )
    )
    parser.add_argument("--aggregate-csv", default="data/processed/five_minute_windows.csv")
    parser.add_argument("--ecg-csv", default="data/processed/five_minute_windows_ecg.csv")
    parser.add_argument("--output", default="data/processed/five_minute_windows_combined.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = pd.read_csv(args.aggregate_csv)
    ecg = pd.read_csv(args.ecg_csv)

    ecg_features = ecg.drop(columns=[c for c in SHARED_METADATA_COLUMNS if c in ecg.columns])
    ecg_features = ecg_features.rename(
        columns={
            col: f"ecg_{col}"
            for col in ecg_features.columns
            if col not in {"driver", "window_start_unix", "window_end_unix", "n_beats"}
        }
    )

    combined = aggregate.merge(
        ecg_features, on=["driver", "window_start_unix", "window_end_unix"], how="inner"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)

    n_positive = int((combined["adb"] == 1).sum())
    n_total = len(combined)
    print(
        f"Wrote {output} with shape {combined.shape} "
        f"({n_positive}/{n_total} positive, rate={n_positive / n_total:.3f}) "
        f"across {combined['driver'].nunique()} drivers."
    )


if __name__ == "__main__":
    main()
