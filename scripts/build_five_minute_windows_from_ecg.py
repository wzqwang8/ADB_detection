from __future__ import annotations

import argparse
import bisect
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import biosppy.signals.ecg as biosppy_ecg
import hrvanalysis
import numpy as np
import pandas as pd

from adb_detection.modeling import load_sleep_summary

DRIVER_FOLDER_PATTERN = re.compile(r"No_(\d+)")
RATE_FOLDER_NAMES = {"250", "500"}
FILE_NOMINAL_DURATION = 600.0  # seconds; matches notebooks/HRV_data-multiple.ipynb
BEATS_PER_30S_MINIMUM = 15  # matches the notebook's "drop if <15 beats/30s" rule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute HRV features directly from raw ECG for the windows in an "
            "existing five_minute_windows.csv, instead of aggregating the "
            "precomputed 30s-sliding-window stream."
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
        "--min-coverage",
        type=float,
        default=0.6,
        help="Minimum fraction of expected raw ECG samples required to keep a window.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/five_minute_windows_ecg.csv",
        help="Where to write the recomputed windows CSV.",
    )
    return parser.parse_args()


def build_file_index(raw_hr_dir: Path) -> dict[int, list[tuple[float, Path, float]]]:
    """Map driver id -> sorted [(file_start_unix, path, sampling_rate), ...]."""

    index: dict[int, list[tuple[float, Path, float]]] = {}
    for driver_dir in raw_hr_dir.iterdir():
        if not driver_dir.is_dir():
            continue
        match = DRIVER_FOLDER_PATTERN.match(driver_dir.name)
        if not match:
            continue
        driver = int(match.group(1))

        files: list[tuple[float, Path, float]] = []
        for rate_dir in driver_dir.glob("measure/*/FilteredECG/*"):
            if rate_dir.name not in RATE_FOLDER_NAMES or not rate_dir.is_dir():
                continue
            rate = float(rate_dir.name)
            for txt_path in rate_dir.glob("*.txt"):
                try:
                    start_unix = float(txt_path.stem)
                except ValueError:
                    continue
                files.append((start_unix, txt_path, rate))

        files.sort(key=lambda item: item[0])
        if files:
            index[driver] = files

    return index


@lru_cache(maxsize=4)
def _load_ecg_file(path_str: str, rate: float) -> tuple[np.ndarray, np.ndarray]:
    raw_text = Path(path_str).read_text()
    cleaned = re.sub(r"[a-zA-Z]", "", raw_text).replace(",", "")
    values = np.array([float(token) for token in cleaned.split()], dtype=np.float64)
    start_unix = float(Path(path_str).stem)
    timestamps = np.linspace(start_unix, start_unix + FILE_NOMINAL_DURATION, len(values))
    return timestamps, values


def find_overlapping_files(
    files: list[tuple[float, Path, float]], window_start: float, window_end: float
) -> list[tuple[float, Path, float]]:
    starts = [item[0] for item in files]
    lo = max(0, bisect.bisect_right(starts, window_start - FILE_NOMINAL_DURATION) - 1)

    overlapping = []
    for start, path, rate in files[lo:]:
        if start >= window_end:
            break
        if start + FILE_NOMINAL_DURATION > window_start:
            overlapping.append((start, path, rate))
    return overlapping


def extract_window_signal(
    files: list[tuple[float, Path, float]], window_start: float, window_end: float
) -> tuple[np.ndarray, np.ndarray, float] | None:
    candidates = find_overlapping_files(files, window_start, window_end)
    if not candidates:
        return None

    rates = {rate for _, _, rate in candidates}
    if len(rates) > 1:
        # A window straddling a sampling-rate change; too ambiguous to merge.
        return None
    rate = candidates[0][2]

    timestamp_parts, value_parts = [], []
    for _, path, _ in candidates:
        timestamps, values = _load_ecg_file(str(path), rate)
        mask = (timestamps >= window_start) & (timestamps < window_end)
        if mask.any():
            timestamp_parts.append(timestamps[mask])
            value_parts.append(values[mask])

    if not timestamp_parts:
        return None

    timestamps = np.concatenate(timestamp_parts)
    values = np.concatenate(value_parts)
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], values[order], rate


def compute_hrv_features(
    values: np.ndarray, rate: float, min_beats: int
) -> tuple[dict[str, float] | None, int, str | None]:
    try:
        out = biosppy_ecg.ecg(signal=values, sampling_rate=rate, show=False)
    except Exception:  # noqa: BLE001 - biosppy raises assorted internal errors
        return None, 0, "ecg_failed"

    rpeaks = out["rpeaks"]
    if len(rpeaks) < 2:
        return None, len(rpeaks), "too_few_rpeaks"

    rr_intervals_ms = (np.diff(rpeaks) / rate) * 1000.0
    if len(rr_intervals_ms) < min_beats:
        return None, len(rr_intervals_ms), "too_few_beats"

    try:
        nn_intervals = hrvanalysis.preprocessing.get_nn_intervals(
            rr_intervals_ms.tolist(),
            low_rri=300,
            high_rri=2000,
            interpolation_method="linear",
            verbose=False,
        )
    except Exception:  # noqa: BLE001
        return None, len(rr_intervals_ms), "nn_cleaning_failed"

    nn_intervals = np.asarray(nn_intervals, dtype=float)
    nn_intervals = nn_intervals[~np.isnan(nn_intervals)]
    if len(nn_intervals) < min_beats:
        return None, len(nn_intervals), "too_few_nn"

    try:
        features = {
            **hrvanalysis.extract_features.get_time_domain_features(nn_intervals.tolist()),
            **hrvanalysis.extract_features.get_frequency_domain_features(nn_intervals.tolist()),
        }
    except Exception:  # noqa: BLE001
        return None, len(nn_intervals), "feature_extraction_failed"

    return features, len(nn_intervals), None


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

    for driver, group in windows.groupby("driver"):
        driver = int(driver)
        files = file_index.get(driver)
        if not files:
            drop_reasons["no_raw_files_for_driver"] += len(group)
            print(f"driver {driver}: no raw ECG files found, skipping {len(group)} windows")
            continue

        group = group.sort_values("window_start_unix")
        kept = 0
        for _, row in group.iterrows():
            window_start = float(row["window_start_unix"])
            window_end = float(row["window_end_unix"])
            window_seconds = window_end - window_start
            min_beats = max(1, round(BEATS_PER_30S_MINIMUM * (window_seconds / 30.0)))

            extraction = extract_window_signal(files, window_start, window_end)
            if extraction is None:
                drop_reasons["no_overlapping_raw_ecg"] += 1
                continue
            timestamps, values, rate = extraction

            expected_samples = window_seconds * rate
            coverage = len(values) / expected_samples if expected_samples else 0.0
            if coverage < args.min_coverage:
                drop_reasons["low_ecg_coverage"] += 1
                continue

            features, n_beats, reason = compute_hrv_features(values, rate, min_beats)
            if features is None:
                drop_reasons[reason] += 1
                continue

            record = dict(features)
            record.update(
                driver=driver,
                window_start_unix=window_start,
                window_end_unix=window_end,
                adb=int(row["adb"]),
                event_count=int(row["event_count"]) if "event_count" in row else 0,
                n_samples=len(values),
                n_beats=n_beats,
            )
            rows.append(record)
            kept += 1

        _load_ecg_file.cache_clear()
        print(f"driver {driver}: {kept}/{len(group)} windows recomputed from raw ECG")

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
    print(
        f"\nWrote {output} with shape {result.shape} "
        f"({n_positive}/{n_total} positive, rate={n_positive / n_total:.3f}) "
        f"across {result['driver'].nunique()} drivers."
    )
    print("Dropped window counts by reason:")
    for reason, count in drop_reasons.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
