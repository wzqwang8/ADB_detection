# Data

This repository tracks only small, derived CSV files in `data/processed/`.

Place the original large datasets locally under `data/raw/` when rerunning the notebooks. The source project used folders similar to:

```text
data/raw/
├── ALL_TRIP_DATA/
├── Raw_HR/
├── frequency_data/
└── Driverdata/
```

Raw data is excluded from git because it is large and may contain participant-level research data. If it needs to be shared, use a controlled storage location, GitHub Releases, or Git LFS after confirming the data-sharing policy.
