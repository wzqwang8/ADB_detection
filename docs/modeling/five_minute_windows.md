# 5-Minute-Window ADB Prediction

This documents the windowed dataset built by `scripts/build_five_minute_windows.py`,
which replaces the old per-event mean tables (`Final_adb_means.csv`, `mean_adb.csv`,
and the reconstructed `final_non_adb_means.csv` fallback) as the input to
`scripts/evaluate_models.py`. See `docs/modeling/leakage_audit.md` for the earlier
leakage findings this build resolves.

## Why the old tables were still a problem

`notebooks/ADB_Intervals-Final.ipynb` computed one *mean* HRV vector per ADB event
(mean over the 5 minutes before/after each event), giving only a few hundred rows
total, and negatives came from a separately reconstructed frequency-only table that
was not time-aligned to the same drivers or timeline. That's a small, mismatched
sample with no real notion of a "5-minute window" to classify.

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
