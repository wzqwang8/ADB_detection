# Data

This repository tracks only small, derived CSV files in `data/processed/`,
including `five_minute_windows.csv` built by `scripts/build_five_minute_windows.py`
(see `docs/modeling/five_minute_windows.md`).

`scripts/build_five_minute_windows.py` and `scripts/build_five_minute_windows_from_ecg.py`
expect the full local research folder at `Data example/` (repository root,
gitignored), organized by data category:

```text
Data example/
├── Provided DB_Final.xlsx              # event log, sleep summary, driving-behaviour sheets
├── heart_rate_data/
│   ├── raw_ecg/                        # Raw_HR/No_<driver>(...)/measure/*/FilteredECG/{250,500}/*.txt
│   ├── hrv_merged/                     # final_data/Final_<driver>.csv - continuous per-driver HRV stream
│   ├── hrv_per_session/                # Driverdata/ - pre-merge per-session HRV files
│   └── hrv_frequency_domain/           # frequency_data/No_<driver>.csv
├── driving_data/
│   ├── trip_by_day/                    # per-driver First_by_day_Mod trip/GPS xlsx files
│   └── excluded_drivers/               # drivers dropped from the final analysis
├── derived_outputs_legacy/             # legacy per-event mean/time-series tables predating
│                                       # data/processed/ (Final_adb_means.csv, etc.)
├── notebooks_original/                 # uncleaned exploratory notebooks (with outputs)
└── legacy_original_export_00491824/    # driver 1's original pre-renumbering raw export
```

`--final-data-dir` defaults to `Data example/heart_rate_data/hrv_merged`,
`--raw-hr-dir` to `Data example/heart_rate_data/raw_ecg`, and
`--events-xlsx`/`--summary-xlsx` to `Data example/Provided DB_Final.xlsx`. Pass
these flags explicitly if your copy lives somewhere else, or under `data/raw/`
instead.

Raw data is excluded from git because it is large and may contain participant-level research data. If it needs to be shared, use a controlled storage location, GitHub Releases, or Git LFS after confirming the data-sharing policy.
