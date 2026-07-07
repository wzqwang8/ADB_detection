# ADB Detection

Machine-learning notebooks and report material for a final year project on predicting Aberrant Driving Behaviour (ADB) from heart-rate and heart-rate-variability signals collected during real-world taxi driving.

The project investigates whether physiological signals such as HR and HRV can help identify fatigue, sleepiness, and driving states associated with ADB events. The original dataset includes RootiRx chest-patch measurements, Unigo Plus/GPS driving records, and processed ADB interval labels.

## Repository layout

```text
.
├── data/
│   └── processed/              # Small derived CSVs included in git
├── docs/                       # Interim report, presentation, poster, and figures
├── notebooks/                  # Cleaned analysis notebooks
│   └── all_trip_data/          # Modelling notebooks for trip-level datasets
└── report/                     # Imperial LaTeX template/assets and report figures
```

Large raw datasets are intentionally excluded from git. Keep them locally under `data/raw/` using the structure described in `data/README.md`.

## Main workflow

1. Build ADB intervals from driving-event data:
   `notebooks/ADB_Intervals-Final.ipynb`
2. Extract and inspect HRV/frequency-domain features:
   `notebooks/HRV_data-multiple.ipynb` and `notebooks/frequency_domain.ipynb`
3. Prepare all-trip modelling tables:
   `notebooks/all_trip_data/ML_time_series_prep.ipynb`
4. Train and compare machine-learning models:
   `notebooks/all_trip_data/ML_time_series.ipynb`, `ML_means.ipynb`, and `SMOTE.ipynb`
5. Generate charts and presentation figures:
   `notebooks/Plotting.ipynb`

## Setup

Create a Python environment and install the analysis dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter lab
```

## Notes

- Notebook outputs have been stripped so code review stays readable.
- The original folder contained about 1.7 GB of raw/intermediate data and reference papers; only lightweight processed CSVs and project deliverables are tracked here.
- Some historical notebooks and duplicate checkpoint files were left out to keep the repository focused on the final workflow.
