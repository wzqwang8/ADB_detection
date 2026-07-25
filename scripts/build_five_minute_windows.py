from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from adb_detection.modeling import load_sleep_summary

FEATURE_COLUMNS = [
    "mean_nni", "sdnn", "sdsd", "nni_50", "pnni_50", "nni_20", "pnni_20",
    "rmssd", "median_nni", "range_nni", "cvsd", "cvnni", "mean_hr", "max_hr",
    "min_hr", "std_hr", "lf", "hf", "lf_hf_ratio", "lfnu", "hfnu",
    "total_power", "vlf",
]

AGG_FUNCS = ["mean", "std", "min", "max"]

FINAL_DATA_PATTERN = re.compile(r"Final_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-aware 5-minute-window dataset for ADB prediction. "
            "Positive windows are the 5 minutes immediately before each recorded "
            "ADB event (pre-event forecasting); negative windows are a "
            "non-overlapping grid over the rest of each driver's recording, "
            "excluded around every event."
        )
    )
    parser.add_argument(
        "--final-data-dir",
        default="Data example/heart_rate_data/hrv_merged",
        help="Folder containing Final_<driver>.csv continuous HRV streams.",
    )
    parser.add_argument(
        "--events-xlsx",
        default="Data example/Provided DB_Final.xlsx",
        help="Workbook containing the Each_ADB_details (and Summary-final) sheets.",
    )
    parser.add_argument(
        "--events-sheet",
        default="Each_ADB_details",
        help="Sheet name with one row per ADB event (NO, recordTime).",
    )
    parser.add_argument(
        "--summary-xlsx",
        default=None,
        help=(
            "Workbook containing the Summary-final sleep-metric sheet. "
            "Defaults to --events-xlsx. Pass --no-sleep-summary to disable."
        ),
    )
    parser.add_argument(
        "--no-sleep-summary",
        action="store_true",
        help="Skip merging per-driver ODI-3%%/CVHRI/CEI sleep metrics.",
    )
    parser.add_argument("--window-minutes", type=float, default=5.0)
    parser.add_argument(
        "--post-event-buffer-minutes",
        type=float,
        default=5.0,
        help="Extra time excluded from the negative grid after each event.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.6,
        help="Minimum fraction of expected 1Hz samples required to keep a window.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=10.0,
        help="Negative:positive window ratio sampled per driver.",
    )
    parser.add_argument("--random-state", type=int, default=129)
    parser.add_argument(
        "--output",
        default="data/processed/five_minute_windows.csv",
        help="Where to write the windows CSV.",
    )
    return parser.parse_args()


def load_events(events_xlsx: str | Path, sheet: str) -> pd.DataFrame:
    events = pd.read_excel(events_xlsx, sheet_name=sheet)
    events = events.dropna(subset=["NO", "recordTime"]).copy()
    events["driver"] = events["NO"].astype(int)
    events["event_unix"] = (
        pd.to_datetime(events["recordTime"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")
    return events[["driver", "event_unix"]]


def driver_id_from_filename(path: Path) -> int | None:
    match = FINAL_DATA_PATTERN.match(path.stem)
    return int(match.group(1)) if match else None


def aggregate_windows(hrv: pd.DataFrame, left_idx: np.ndarray, right_idx: np.ndarray) -> list[dict[str, float]]:
    """Aggregate mean/std/min/max of FEATURE_COLUMNS for each [left, right) row range."""

    values = hrv[FEATURE_COLUMNS].to_numpy(dtype=float)
    rows = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for left, right in zip(left_idx, right_idx):
            chunk = values[left:right]
            stats = {}
            for func in AGG_FUNCS:
                reduced = getattr(np, f"nan{func}")(chunk, axis=0)
                for col, val in zip(FEATURE_COLUMNS, reduced):
                    stats[f"{col}_{func}"] = val
            rows.append(stats)
    return rows


def build_driver_windows(
    driver: int,
    hrv: pd.DataFrame,
    event_times: np.ndarray,
    window_seconds: float,
    buffer_seconds: float,
    min_coverage: float,
    negative_ratio: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict]:
    hrv = hrv.sort_values("start_unix").drop_duplicates(subset="start_unix").reset_index(drop=True)
    starts = hrv["start_unix"].to_numpy(dtype=float)
    min_samples = min_coverage * window_seconds

    stats = {
        "driver": driver,
        "n_events": len(event_times),
        "positives_kept": 0,
        "positives_dropped_low_coverage": 0,
        "negative_candidates": 0,
        "negatives_kept": 0,
    }

    # --- positives: 5 minutes immediately before each event ---
    pos_starts = event_times - window_seconds
    pos_ends = event_times
    left_idx = np.searchsorted(starts, pos_starts, side="left")
    right_idx = np.searchsorted(starts, pos_ends, side="left")
    n_samples = right_idx - left_idx
    keep = n_samples >= min_samples

    stats["positives_dropped_low_coverage"] = int((~keep).sum())
    stats["positives_kept"] = int(keep.sum())

    positive_rows = aggregate_windows(hrv, left_idx[keep], right_idx[keep])
    for row, w_start, w_end, n in zip(
        positive_rows, pos_starts[keep], pos_ends[keep], n_samples[keep]
    ):
        row.update(
            driver=driver,
            window_start_unix=w_start,
            window_end_unix=w_end,
            n_samples=int(n),
            adb=1,
            event_count=1,
        )

    n_positive = len(positive_rows)

    # --- negative grid: non-overlapping bins, excluded around every event ---
    stream_min, stream_max = starts.min(), starts.max()
    n_bins = int((stream_max - stream_min) // window_seconds)
    if n_bins <= 0 or n_positive == 0:
        return positive_rows, [], stats

    bin_starts = stream_min + np.arange(n_bins) * window_seconds
    bin_ends = bin_starts + window_seconds

    excl_starts = event_times - window_seconds
    excl_ends = event_times + buffer_seconds
    # A bin overlaps an event's exclusion span if bin_start < excl_end and bin_end > excl_start.
    overlap = (bin_starts[:, None] < excl_ends[None, :]) & (
        bin_ends[:, None] > excl_starts[None, :]
    )
    excluded = overlap.any(axis=1)

    grid_left = np.searchsorted(starts, bin_starts, side="left")
    grid_right = np.searchsorted(starts, bin_ends, side="left")
    grid_samples = grid_right - grid_left

    candidate_mask = (~excluded) & (grid_samples >= min_samples)
    candidate_idx = np.where(candidate_mask)[0]
    stats["negative_candidates"] = int(len(candidate_idx))

    target_n = int(round(negative_ratio * n_positive))
    if target_n and len(candidate_idx) > target_n:
        candidate_idx = rng.choice(candidate_idx, size=target_n, replace=False)
        candidate_idx.sort()

    stats["negatives_kept"] = int(len(candidate_idx))

    negative_rows = aggregate_windows(
        hrv, grid_left[candidate_idx], grid_right[candidate_idx]
    )
    for row, w_start, w_end, n in zip(
        negative_rows,
        bin_starts[candidate_idx],
        bin_ends[candidate_idx],
        grid_samples[candidate_idx],
    ):
        row.update(
            driver=driver,
            window_start_unix=w_start,
            window_end_unix=w_end,
            n_samples=int(n),
            adb=0,
            event_count=0,
        )

    return positive_rows, negative_rows, stats


def main() -> None:
    args = parse_args()
    window_seconds = args.window_minutes * 60.0
    buffer_seconds = args.post_event_buffer_minutes * 60.0

    events = load_events(args.events_xlsx, args.events_sheet)

    sleep_summary = None
    if not args.no_sleep_summary:
        summary_xlsx = args.summary_xlsx or args.events_xlsx
        try:
            sleep_summary = load_sleep_summary(summary_xlsx)
        except (ValueError, KeyError) as exc:
            print(f"Skipping sleep-summary merge ({exc}).")

    final_data_dir = Path(args.final_data_dir)
    rng = np.random.default_rng(args.random_state)

    all_rows: list[dict] = []
    driver_stats: list[dict] = []

    driver_files = sorted(
        (
            (driver_id_from_filename(path), path)
            for path in final_data_dir.glob("Final_*.csv")
        ),
        key=lambda pair: pair[0] if pair[0] is not None else -1,
    )

    for driver, path in driver_files:
        if driver is None:
            continue
        driver_events = events.loc[events["driver"] == driver, "event_unix"].to_numpy(dtype=float)
        if len(driver_events) == 0:
            continue

        hrv = pd.read_csv(path)
        positive_rows, negative_rows, stats = build_driver_windows(
            driver=driver,
            hrv=hrv,
            event_times=driver_events,
            window_seconds=window_seconds,
            buffer_seconds=buffer_seconds,
            min_coverage=args.min_coverage,
            negative_ratio=args.negative_ratio,
            rng=rng,
        )
        all_rows.extend(positive_rows)
        all_rows.extend(negative_rows)
        driver_stats.append(stats)
        print(
            f"driver {driver}: {stats['n_events']} events -> "
            f"{stats['positives_kept']} positive windows kept "
            f"({stats['positives_dropped_low_coverage']} dropped for coverage), "
            f"{stats['negatives_kept']}/{stats['negative_candidates']} negative windows"
        )

    if not all_rows:
        raise SystemExit("No windows were built. Check --final-data-dir and --events-xlsx.")

    windows = pd.DataFrame(all_rows)

    if sleep_summary is not None:
        windows = windows.merge(sleep_summary, on="driver", how="left")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output, index=False)

    n_positive = int((windows["adb"] == 1).sum())
    n_total = len(windows)
    print(
        f"\nWrote {output} with shape {windows.shape} "
        f"({n_positive}/{n_total} positive, rate={n_positive / n_total:.3f}) "
        f"across {windows['driver'].nunique()} drivers."
    )


if __name__ == "__main__":
    main()
