from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frequency-only non-ADB fallback table from frequency_data/No_*.csv."
        )
    )
    parser.add_argument(
        "--frequency-dir",
        required=True,
        help="Folder containing No_*.csv frequency-domain HRV files.",
    )
    parser.add_argument(
        "--summary-xlsx",
        required=True,
        help="Provided DB_Final.xlsx workbook with Summary-final sheet.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/final_non_adb_means.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max-rows-per-driver",
        type=int,
        default=100,
        help="Cap rows sampled from each driver to avoid one driver dominating.",
    )
    return parser.parse_args()


def driver_from_filename(path: Path) -> int | None:
    match = re.match(r"No_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def load_sleep_summary(summary_xlsx: Path) -> pd.DataFrame:
    summary = pd.read_excel(summary_xlsx, sheet_name="Summary-final")
    summary = summary[["Name", "ODI-3%", "CVHRI", "CEI"]].dropna(subset=["Name"])
    summary["driver"] = summary["Name"].astype(int)
    return (
        summary.groupby("driver", as_index=False)[["ODI-3%", "CVHRI", "CEI"]]
        .median(numeric_only=True)
    )


def main() -> None:
    args = parse_args()
    frequency_dir = Path(args.frequency_dir)
    sleep_summary = load_sleep_summary(Path(args.summary_xlsx))

    frames: list[pd.DataFrame] = []
    for path in sorted(frequency_dir.glob("No_*.csv")):
        driver = driver_from_filename(path)
        if driver is None:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if args.max_rows_per_driver and len(frame) > args.max_rows_per_driver:
            frame = frame.sample(
                n=args.max_rows_per_driver, random_state=129
            ).sort_index()
        frame["driver"] = driver
        frames.append(frame)

    if not frames:
        raise SystemExit(f"No frequency CSV files found in {frequency_dir}")

    non_adb = pd.concat(frames, ignore_index=True)
    non_adb = non_adb.merge(sleep_summary, on="driver", how="left")
    non_adb = non_adb.dropna(axis=1, how="all")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    non_adb.to_csv(output, index=False)
    print(f"Wrote {output} with shape {non_adb.shape}")
    print(
        "Caveat: this is a frequency-only fallback table. The original timestamped "
        "non-ADB means file was not present in the backup."
    )


if __name__ == "__main__":
    main()
