# 5-Minute-Window ADB Prediction

This documents the windowed dataset built by `scripts/build_five_minute_windows.py`,
which replaces the old per-event mean tables (`Final_adb_means.csv`,
`Final_non_adb_means.csv`, `mean_adb.csv`) as the input to
`scripts/evaluate_models.py`. See `docs/modeling/leakage_audit.md` for the earlier
leakage findings this build resolves.

## Why the old tables were still a problem

`notebooks/ADB_Intervals-Final.ipynb` computed one *mean* HRV vector per hand-picked
interval (the 5 minutes before/after each ADB event, or a matched baseline),
giving only a few hundred ADB rows and ~9.5k non-ADB rows total — a small sample
with no real notion of a "5-minute window" to classify, just per-event summaries.

## What the new pipeline builds

Source data (see `data/README.md`):
- `Data example/final_data/Final_<driver>.csv` — a continuous, ~1 Hz stream per
  driver of HRV features computed over a trailing 30-second sliding window
  (`mean_nni`, `sdnn`, `rmssd`, `lf`, `hf`, ... 23 columns), spanning the driver's
  entire multi-day recording.
- `Data example/Provided DB_Final.xlsx`, sheet `Each_ADB_details` — one row per
  recorded ADB event (harsh braking, hard acceleration, speeding, aggressive
  steering, GPS drift) with a driver id and timestamp. Sheet `Summary-final` adds
  per-driver sleep-quality metrics (`ODI-3%`, `CVHRI`, `CEI`).

**Label definition — pre-event forecasting:** a 5-minute window is `adb=1` only if
it is the 5 minutes immediately *before* a recorded event (does the physiology in
the preceding 5 minutes forecast the event?), matching the original researcher's
baseline-vs-pre-event hypothesis. It is not "did an event happen during this
window" — that concurrent-detection framing was considered and rejected in favour
of the forecasting framing already implicit in the project's baseline/pre-event
comparison (`Final_compare.csv`).

**Negative windows** come from a non-overlapping 5-minute grid tiled across each
driver's whole recording, with every event excluded by a buffer: a grid window is
dropped if it overlaps `[event − 5 min, event + 5 min)` for *any* event from that
driver, so negatives are never adjacent to (or contaminated by) an event's
pre-event or post-event recovery physiology. Negatives are then randomly
subsampled per driver to a configurable ratio (default 10 negatives per positive)
since a full grid would be 50-100x the positive count over a ~117-hour recording —
keeping training tractable while remaining realistically imbalanced.

**Coverage filter:** both positive and negative windows are dropped if fewer than
60% of the expected ~300 one-second samples are actually present (sensor gaps,
events too close to the start of a recording, etc.), so aggregated features
aren't computed from a handful of scattered points.

**Features:** mean/std/min/max of each of the 23 HRV columns across the samples
in the window (92 columns), plus the per-driver sleep metrics broadcast onto every
window. Timestamps, row counts, and event counts are kept as `driver`,
`window_start_unix`, `window_end_unix`, `n_samples`, `event_count` for QC but are
dropped before modelling (`modeling.DEFAULT_DROP_COLUMNS`).

Current run (default settings, 29 drivers with recorded events — drivers 10 and
17 have none and are excluded): 3,817 windows, 347 positive (9.1%).

## Why this doesn't reintroduce leakage

- **Grouping is by real driver id**, taken directly from the filename/event table
  — not inferred from timestamp overlap like the old `add_driver_groups` fallback.
  `evaluate_models.py`'s existing `GroupShuffleSplit`/`GroupKFold` logic (see
  `src/adb_detection/modeling.py`) puts a driver's windows entirely in train or
  entirely in test.
- **No sliding-window overlap** in the model's input rows: the negative grid is
  non-overlapping, and although a driver's positive (pre-event) windows can
  overlap each other when two events are close together, that only creates
  redundant *positive* rows within one driver, which the group split already
  keeps on one side of the train/test boundary — it cannot leak across the split.
- **SMOTE stays inside the pipeline**, fit only on each training fold
  (`make_pipeline` in `src/adb_detection/modeling.py`), unchanged from the
  existing fix.
- Sleep-quality features (`ODI-3%`, `CVHRI`, `CEI`) are a per-driver trait, not an
  identifier or timestamp — they don't reintroduce the "index encodes
  participant identity" problem `leakage_audit.md` flagged.

## Regenerating

```bash
PYTHONPATH=src python scripts/build_five_minute_windows.py \
  --final-data-dir "Data example/final_data" \
  --events-xlsx "Data example/Provided DB_Final.xlsx" \
  --output data/processed/five_minute_windows.csv

PYTHONPATH=src python scripts/evaluate_models.py \
  --windows-csv data/processed/five_minute_windows.csv \
  --output reports/model_evaluation_5min.json
```

Tunable via CLI flags: `--window-minutes`, `--post-event-buffer-minutes`,
`--min-coverage`, `--negative-ratio`, `--random-state`. `--no-sleep-summary`
disables the `ODI-3%`/`CVHRI`/`CEI` merge.

## Raw-ECG-recomputed variant

`scripts/build_five_minute_windows_from_ecg.py` produces a second dataset,
`data/processed/five_minute_windows_ecg.csv`, that keeps the exact same row set
as `five_minute_windows.csv` (same `driver`/`window_start_unix`/`window_end_unix`/
`adb` per window — same labels, same negative sampling, same leakage-avoidance
properties described above) but computes each window's HRV features directly
from the raw ECG signal instead of aggregating the precomputed 30-second-sliding
stream.

**Why:** `Final_<driver>.csv` is itself a 30s-window/1s-step computation, so even
non-overlapping 5-minute *bins* over it are a statistic (mean/std/min/max) of an
already-smoothed, highly autocorrelated series — not an independent measurement.
Recomputing from `Data example/Raw_HR/No_<driver>(...)/measure/*/FilteredECG/{250,500}/*.txt`
(the filtered ECG signal, ~250Hz for most drivers, ~500Hz for two sessions of
drivers 2 and 17) gives each window one genuine R-peak-detection-based HRV
measurement instead of 92 aggregate-of-aggregate columns.

**Method** (matches `notebooks/HRV_data-multiple.ipynb`'s original approach, generalized
from a 30s/1s-step sliding loop to one call per true 5-minute window):
`biosppy.signals.ecg.ecg(signal, sampling_rate, show=False)` for R-peak detection,
RR intervals from consecutive R-peak sample indices, `hrvanalysis.preprocessing.get_nn_intervals`
(300-2000ms physiological bounds) to clean artifacts, then `hrvanalysis`'s
`get_time_domain_features`/`get_frequency_domain_features`. `hrvanalysis`'s own
output keys (`mean_nni`, `sdnn`, ..., `lf`, `hf`, `lfnu`, `hfnu`, `vlf`, ...) are
exactly this project's existing HRV column names — no renaming needed. One
deliberate deviation: the notebook passes `int(len(window)/30)` as
`get_frequency_domain_features`'s `sampling_frequency` argument, which — because
`len(window)` scales with window duration at a fixed sample rate — always
evaluates to the ECG sample rate (~250) regardless of window size. That's not a
sensible NN-interval resampling rate for Welch's method, so this script uses the
library default (4 Hz) instead.

A window is dropped (and counted in a per-driver drop-reason summary) if: no raw
ECG files exist for that driver, no file overlaps the window's time span, ECG
sample coverage is below `--min-coverage` (default 0.6, same semantics as the
aggregate build), or too few beats are detected (scaled from the notebook's
"drop if <15 beats/30s" rule, i.e. ~150 beats/5min). In practice this drops very
little: 3,803 of 3,817 windows (99.6%) recomputed successfully across all 29
drivers, all 14 drops for insufficient detected beats (no driver lost entirely).

```bash
pip install biosppy hrv-analysis   # not installed by default; see requirements.txt

PYTHONPATH=src python scripts/build_five_minute_windows_from_ecg.py \
  --windows-csv data/processed/five_minute_windows.csv \
  --raw-hr-dir "Data example/Raw_HR" \
  --output data/processed/five_minute_windows_ecg.csv

PYTHONPATH=src python scripts/evaluate_models.py \
  --windows-csv data/processed/five_minute_windows_ecg.csv \
  --output reports/model_evaluation_5min_ecg.json
```

Compare `reports/model_evaluation_5min.json` (aggregate features) against
`reports/model_evaluation_5min_ecg.json` (raw-ECG features) to see whether
recomputing from source signal actually improves generalization over just
aggregating the precomputed stream.

## Caveats

- Positive counts per driver are small (as few as 3 events for some drivers), so
  grouped CV folds vary in difficulty; read the per-fold and holdout metrics
  together, not just the mean.
- The pre-event window is fixed at exactly 5 minutes before each event; it does
  not account for how long ago the *previous* event was, so a small number of
  positive windows may still contain a fragment of an earlier event's recovery
  period if two events are within 10 minutes of each other.
- `Data example/` is a large, gitignored local folder; this pipeline expects it
  to exist at the repository root as-is (see `data/README.md`) rather than
  requiring the data to be copied into `data/raw/` first.
