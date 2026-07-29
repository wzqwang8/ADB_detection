from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from adb_detection.modeling import load_sleep_summary
from build_five_minute_windows_from_ecg import (
    BEATS_PER_30S_MINIMUM,
    build_file_index,
    compute_hrv_features,
    extract_window_signal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute HRV features per 30-second sub-bin within each 5-minute "
            "window (instead of one aggregate vector per window), to give models "
            "a genuine short time series of the physiology leading up to each "
            "window's end instead of a single collapsed snapshot."
        )
    )
    parser.add_argument(
        "--windows-csv",
        default="data/processed/five_minute_windows.csv",
        help="Source window row-set (driver/window_start_unix/window_end_unix/adb).",
    )
    parser.add_argument(
        "--raw-hr-dir",
        default="Data example/Raw_HR",
        help="Folder containing No_<driver>(...)/measure/*/FilteredECG/{250,500}/*.txt",
    )
    parser.add_argument(
        "--summary-xlsx",
        default="Data example/Provided DB_Final.xlsx",
        help="Workbook containing the Summary-final sleep-metric sheet.",
    )
    parser.add_argument(
        "--no-sleep-summary",
        action="store_true",
        help="Skip merging per-driver ODI-3%%/CVHRI/CEI sleep metrics.",
    )
    parser.add_argument(
        "--sub-bin-seconds",
        type=float,
        default=30.0,
        help="Length of each sequence step. 5-minute windows are split into "
        "300/sub_bin_seconds equal, non-overlapping, chronologically ordered bins.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.6,
        help="Minimum fraction of expected raw ECG samples required to keep a sub-bin.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/five_minute_windows_sequence.csv",
        help="Where to write the recomputed sequence CSV (wide format, one row per "
        "window, columns suffixed _t0.._t{n-1} in chronological order).",
    )
    return parser.parse_args()


def build_sub_bins(window_start: float, window_end: float, sub_bin_seconds: float) -> list[tuple[float, float]]:
    n_bins = round((window_end - window_start) / sub_bin_seconds)
    return [
        (window_start + i * sub_bin_seconds, window_start + (i + 1) * sub_bin_seconds)
        for i in range(n_bins)
    ]


def main() -> None:
    args = parse_args()

    windows = pd.read_csv(args.windows_csv)
    file_index = build_file_index(Path(args.raw_hr_dir))

    sleep_summary = None
    if not args.no_sleep_summary:
        try:
            sleep_summary = load_sleep_summary(args.summary_xlsx)
        except (ValueError, KeyError) as exc:
            print(f"Skipping sleep-summary merge ({exc}).")

    rows: list[dict] = []
    drop_reasons: Counter[str] = Counter()
    feature_names: list[str] | None = None

    for driver, group in windows.groupby("driver"):
        driver = int(driver)
        files = file_index.get(driver)
        if not files:
            drop_reasons["no_raw_files_for_driver"] += len(group)
            print(f"driver {driver}: no raw ECG files found, skipping {len(group)} windows")
            continue

        group = group.sort_values("window_start_unix")
        kept = 0
        min_beats = max(1, round(BEATS_PER_30S_MINIMUM * (args.sub_bin_seconds / 30.0)))

        for _, row in group.iterrows():
            window_start = float(row["window_start_unix"])
            window_end = float(row["window_end_unix"])
            sub_bins = build_sub_bins(window_start, window_end, args.sub_bin_seconds)

            step_features: list[dict[str, float]] = []
            failed = False
            for bin_start, bin_end in sub_bins:
                extraction = extract_window_signal(files, bin_start, bin_end)
                if extraction is None:
                    drop_reasons["no_overlapping_raw_ecg"] += 1
                    failed = True
                    break
                timestamps, values, rate = extraction

                expected_samples = (bin_end - bin_start) * rate
                coverage = len(values) / expected_samples if expected_samples else 0.0
                if coverage < args.min_coverage:
                    drop_reasons["low_ecg_coverage"] += 1
                    failed = True
                    break

                features, _, reason = compute_hrv_features(values, rate, min_beats)
                if features is None:
                    drop_reasons[reason] += 1
                    failed = True
                    break
                step_features.append(features)

            if failed:
                continue

            if feature_names is None:
                feature_names = sorted(step_features[0].keys())

            record: dict[str, float | int] = {
                "driver": driver,
                "window_start_unix": window_start,
                "window_end_unix": window_end,
                "adb": int(row["adb"]),
                "event_count": int(row["event_count"]) if "event_count" in row else 0,
            }
            for step, features in enumerate(step_features):
                for name in feature_names:
                    record[f"{name}_t{step}"] = features[name]
            rows.append(record)
            kept += 1

        print(f"driver {driver}: {kept}/{len(group)} windows recomputed as sequences")

    if not rows:
        raise SystemExit("No windows were recomputed. Check --raw-hr-dir and --windows-csv.")

    result = pd.DataFrame(rows)
    if sleep_summary is not None:
        result = result.merge(sleep_summary, on="driver", how="left")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    n_positive = int((result["adb"] == 1).sum())
    n_total = len(result)
    n_steps = round(300 / args.sub_bin_seconds)
    print(
        f"\nWrote {output} with shape {result.shape} "
        f"({n_positive}/{n_total} positive, rate={n_positive / n_total:.3f}) "
        f"across {result['driver'].nunique()} drivers, "
        f"{n_steps} steps x {len(feature_names or [])} features per window."
    )
    print("Dropped window counts by reason (any sub-bin failing drops the whole window):")
    for reason, count in drop_reasons.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
