# Data

This repository tracks only small, derived CSV files in `data/processed/`,
including `five_minute_windows.csv` built by `scripts/build_five_minute_windows.py`
(see `docs/modeling/five_minute_windows.md`).

Place the original large datasets locally under `data/raw/` when rerunning the notebooks. The source project used folders similar to:

```text
data/raw/
├── ALL_TRIP_DATA/
├── Raw_HR/
├── frequency_data/
└── Driverdata/
```

`scripts/build_five_minute_windows.py` instead expects the full local research
folder as-is at `Data example/` (repository root, gitignored) — specifically
`Data example/final_data/Final_<driver>.csv` (continuous per-driver HRV streams)
and `Data example/Provided DB_Final.xlsx` (`Each_ADB_details`/`Summary-final`
sheets). Point `--final-data-dir`/`--events-xlsx` elsewhere if your copy lives
somewhere else, or under `data/raw/` instead.

Raw data is excluded from git because it is large and may contain participant-level research data. If it needs to be shared, use a controlled storage location, GitHub Releases, or Git LFS after confirming the data-sharing policy.
