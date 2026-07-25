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

**Result:** mixed but net positive. `gradient_boosting` and `xgboost` improve
meaningfully (xgboost ROC-AUC 0.610 → 0.685, the best result across every
dataset/model tried), while `random_forest` and `svm_rbf` do slightly worse —
plausibly because real R-peak detection on ambulatory ECG introduces some
motion-artifact noise the pre-cleaned aggregate stream didn't have, which hurts
models more sensitive to noisy individual features than boosted trees are.

## Combined variant (tried, not adopted)

`scripts/combine_window_features.py` merges the aggregate and raw-ECG feature
sets for the same windows (driver/window bounds) into one 118-feature table,
`data/processed/five_minute_windows_combined.csv`. The hypothesis was that
aggregate stats (std/min/max, capturing within-window variability) and clean
ECG-derived central values might be complementary. In practice it made the two
best models *worse* — xgboost's ROC-AUC collapsed from 0.685 to 0.495 (chance),
gradient_boosting from 0.671 to 0.529 — while logistic regression/random forest
ticked up slightly. Doubling the feature count without adding rows (still
~3,800) looks like it pushed the boosted-tree models into overfitting. Kept in
the repo as a documented negative result, not as the recommended dataset.

## Result instability at this sample size

Raising `--min-coverage` from 0.6 to 0.9 for the raw-ECG build dropped only 3 of
3,803 windows (coverage was already high almost everywhere; the bottleneck
wasn't incomplete windows). Despite the dataset being ~99.9% identical, holdout
scores moved more than that tiny data change should justify: xgboost ROC-AUC
0.685 → 0.623, random_forest 0.587 → 0.514 (now below chance), gradient_boosting
0.671 → 0.609, while logistic_regression/svm barely moved. The likely
explanation is that `GroupKFold`'s fold boundaries shift slightly when a driver
loses a handful of rows, which changes which hyperparameters `GridSearchCV`
selects, which then changes the fitted model evaluated on the (unchanged) test
drivers — i.e. this is small-sample instability in model selection, not a real
effect of the coverage threshold.

**Takeaway:** with only 29 drivers and ~350 positive windows, a single
train/test split's point estimate (e.g. "xgboost gets 0.685 ROC-AUC") should be
read as "somewhere in the 0.5-0.7 range for this model," not as a precise
number — the variance between near-identical runs is comparable to the
differences between models or between feature sets. Comparing many single-split
results (as this document does) is useful for spotting large, consistent
effects (e.g. random-split vs. grouped-split scores differing by leakage), but
not for ranking small differences. A more defensible next step would be
repeated grouped resampling (many random `GroupShuffleSplit` holdouts, or
leave-one-driver-out CV) to get a distribution rather than one number per
model/dataset.

## Leave-one-driver-out evaluation (the authoritative number)

`scripts/evaluate_models_logo.py` implements the "more defensible next step"
flagged above: it picks each model's hyperparameters once via grouped CV over
the whole dataset (`modeling.select_best_hyperparameters`), then refits each
model with those *fixed* hyperparameters once per held-out driver
(`modeling.leave_one_group_out_evaluation` — exhaustive, since there are only
29 drivers, so no arbitrary choice of "how many repeats"). This gives a real
distribution of holdout scores instead of one split's point estimate, at a
fraction of the cost of repeating the full grid search per fold.

```bash
PYTHONPATH=src python scripts/evaluate_models_logo.py \
  --windows-csv data/processed/five_minute_windows_ecg.csv \
  --output reports/model_evaluation_logo_ecg.csv
```

**Result across all 29 driver holdouts, both datasets** (mean ± std ROC-AUC):

| Model | Aggregate | Raw ECG |
|---|---|---|
| xgboost | 0.567 ± 0.153 | 0.580 ± 0.155 |
| random_forest | 0.577 ± 0.153 | 0.588 ± 0.148 |
| logistic_regression | 0.565 ± 0.139 | 0.580 ± 0.143 |
| gradient_boosting | 0.558 ± 0.151 | 0.571 ± 0.137 |
| svm_rbf | 0.585 ± 0.155 | 0.567 ± 0.148 |

**This supersedes every single-split headline above, including the "xgboost
gets 0.685 ROC-AUC" one.** Under exhaustive leave-one-driver-out evaluation:

- Every model on both datasets converges to roughly the same **~0.56-0.59 mean
  ROC-AUC**, with the per-model, per-dataset differences (≤0.02) an order of
  magnitude smaller than the std (~0.14-0.16). There is no genuinely best model
  and **no genuine improvement from raw-ECG recomputation over the aggregate
  features** — the single-split result suggesting xgboost+raw-ECG was a clear
  winner (0.685 vs 0.610) was mostly noise from a single favorable split, not a
  robust effect. The "combined features hurt xgboost/gradient_boosting"
  finding above should be read the same way: probably also within normal
  single-split noise, not a proven overfitting effect (it wasn't re-checked
  under LOGO).
- Individual driver folds vary hugely (ROC-AUC from ~0.15 to ~0.87 — see
  `reports/model_evaluation_logo_{agg,ecg}.csv` for the per-driver breakdown).
  Some drivers' physiology predicts their own ADB risk well; others don't
  generalize from the rest of the cohort at all.

**Treat this table, not any single-split number, as the project's actual
current performance ceiling**: real signal, but weak (~0.57 ROC-AUC, versus
0.5 for a coin flip) and highly driver-dependent — consistent with a small
(29-driver, ~350-event) dataset rather than a fundamental flaw in either
feature-engineering approach. Growing the driver cohort would do more for
this project's results than further feature tuning.

## Per-driver normalization (tried, marginal help)

`modeling.normalize_within_driver` (used via `--per-driver-normalize` on both
evaluation scripts) z-scores each feature against that driver's own mean/std
before modelling, turning absolute HRV levels into "deviation from this
driver's own baseline." This targets between-subject baseline variance
specifically, rather than trying yet another model family — the LOGO table
above already shows every model family converging to the same ~0.57 ROC-AUC,
which pointed at the data/features rather than model choice as the
bottleneck. It doesn't cross the group boundary `GroupKFold`/
`LeaveOneGroupOut` rely on (a driver's own rows supply their own stats, so no
other driver's data is used), though it does assume access to that driver's
own aggregate feature statistics — realistic for a wearable with a
calibration period, but slightly more information than a single incoming
reading would give you.

**Result on the aggregate dataset** (mean ROC-AUC across all 29 driver
holdouts, baseline vs. per-driver normalized):

| Model | Baseline | Normalized | Δ |
|---|---|---|---|
| svm_rbf | 0.585 | 0.618 | +0.033 |
| random_forest | 0.577 | 0.593 | +0.016 |
| xgboost | 0.584 | 0.591 | +0.007 |
| gradient_boosting | 0.569 | 0.583 | +0.014 |
| logistic_regression | 0.566 | 0.561 | −0.005 |

4 of 5 models improve on both ROC-AUC and balanced accuracy (random_forest's
balanced accuracy moves the most, +0.034), svm_rbf improves the most on
ROC-AUC. logistic_regression is the outlier, getting slightly worse —
plausibly because it already scales features globally (`StandardScaler` in
`make_pipeline`), so per-driver normalization on top is a redundant second
normalization rather than new information.

**Read this the same way as the "Result instability" section above**: every
delta here (0.01-0.03) is an order of magnitude smaller than the per-driver
std (~0.12-0.15), so this is a real, cheap, worth-keeping improvement, not a
result that overturns the ~0.55-0.6 ROC-AUC ceiling. It's consistent with
that ceiling being a real-signal/small-cohort limit rather than a fixable
modelling mistake — normalizing away between-driver baseline differences
helps a little, but there's no remaining large effect it was masking.

```bash
PYTHONPATH=src python scripts/evaluate_models_logo.py \
  --windows-csv data/processed/five_minute_windows.csv \
  --per-driver-normalize \
  --output reports/model_evaluation_logo_agg_normalized.csv
```

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
