# Interview Narrative: Revisiting the ADB Detection FYP

This documents how to talk about this project three years on — what changed,
why, and how to frame it as a strength rather than "my old results were
wrong." See `docs/modeling/leakage_audit.md` and
`docs/modeling/five_minute_windows.md` for the full technical detail behind
every claim here.

## The one-line version

*"I revisited a three-year-old final year project with the validation rigor
I'd apply professionally now, found classic ML leakage bugs in my own old
work, fixed them, and the honest result dropped from ~98% accuracy to ~0.57
ROC-AUC — then I spent real effort trying to improve that honest number and
confirmed it's a small-cohort data limit, not a fixable mistake."*

That's a story about technical maturity and self-audit, not a story about a
project that "got worse."

## Old (2023) results vs. current results

**2023 poster/presentation:** XGBoost 97.9% accuracy, KNN 92%, Random Forest
88%, SVM 79.4%, Logistic Regression and Naive Bayes "performed poorly" — on
baseline-vs-pre-event 5-minute intervals, 80:20 random split, SMOTE applied
before the split, 10-fold cross-validation, accuracy as the headline metric.

**Now (leave-one-driver-out evaluation, leakage-audited pipeline):** every
model — logistic regression, random forest, gradient boosting, SVM, XGBoost —
converges to roughly the same **~0.55-0.6 ROC-AUC** (0.5 = coin flip).
Weak, but real, signal; nothing close to 98%.

### Why the gap is almost entirely leakage, not model quality

1. **SMOTE was fit before the train/test split.** Synthetic oversampled
   points were built from neighbours that included rows that later became
   "test" rows — the model partly memorized its own test set.
2. **Splits were random by row, not grouped by driver.** HRV windows from the
   same driver/trip are highly autocorrelated; random splitting let
   near-duplicate windows from the same driver land on both sides of the
   split, so models were partly recognizing *drivers*, not detecting
   fatigue.
3. **Timestamp/index columns were left in as features** — these can encode
   collection order and participant identity, a shortcut a model will
   happily exploit.
4. **Accuracy was the headline metric on a ~91:9 imbalanced problem** — a
   model that mostly ignores the positive class already looks great on
   accuracy alone.

Fix all four (grouped splits, SMOTE inside the CV pipeline, drop identifier
columns, report balanced accuracy/ROC-AUC/recall together) and the signal
drops to "real but weak." That is expected and correct — it's evidence the
fixes were necessary, not evidence the new pipeline is worse at modelling.

## Why not linear regression (and why logistic regression alone wasn't enough either)

Two distinct points worth keeping separate when explaining this:

- **This is a classification problem, not a regression one** — the target is
  binary (ADB event vs. not), so plain linear regression is the wrong tool
  by construction. The correct "linear" baseline is logistic regression,
  which was included in both the old and new pipelines.
- **Even logistic regression consistently underperforms the tree ensembles —
  and that's expected, not a bug.** Fatigue's effect on HRV isn't a smooth
  linear function of any single feature: the autonomic nervous system
  shifts between sympathetic and parasympathetic dominance (a threshold-like
  switch, not a linear trend), and the informative signal is often an
  *interaction* — e.g. LF:HF ratio combined with prior-night sleep quality,
  or HRV combined with time-since-shift-start. A linear/logistic model can
  only combine features additively; gradient-boosted trees and random
  forests naturally capture thresholds and interactions, which is why they
  were the stronger performers in both the 2023 and current evaluations.

**Interview framing:** *"I used logistic regression as an interpretable,
low-variance baseline, but chose gradient-boosted trees as the primary model
family because the underlying physiology is non-linear and
interaction-heavy — HRV's relationship to fatigue involves autonomic-state
switching, not a smooth linear trend."*

## What was tried to push past the ~0.57 ROC-AUC ceiling

After the leakage audit, three independent, reasonable hypotheses for why
the ceiling might still be "fixable" were each tested and ruled out:

| Hypothesis | Experiment | Result |
|---|---|---|
| Between-driver baseline differences dominate | Per-driver z-score normalization | Small, inconsistent effect (+0.03 on aggregate dataset, slightly negative on raw-ECG dataset) — noise-level |
| 95 features are redundant / causing overfitting | Univariate feature selection (SelectKBest, k tuned via grid search) | Essentially no change; grid search often chose "no reduction" |
| Collapsing each window to one summary vector throws away useful history | GRU sequence model over each driver's lookback of prior monitored windows | 0.576 ± 0.178 ROC-AUC — lands inside the existing models' range, not better |

Each experiment used the same leave-one-driver-out methodology as the
headline result, so the comparisons are apples-to-apples. None moved the
needle beyond the ~0.12-0.19 per-driver standard deviation already present
in the baseline — i.e., none of these "obvious" fixes were sitting on
top of a real, hidden effect.

**What this demonstrates, and how to say it:** *"Rather than declaring
0.57 ROC-AUC the final answer, I tested three specific, falsifiable
hypotheses for why it might be improvable — feature redundancy, subject
baseline drift, and missing temporal context — using the same rigorous
validation each time. All three came back negative, which is itself a
result: it means the ceiling reflects the dataset's scale (29 drivers, ~350
events), not a gap in the modelling approach. The one lever that would
plausibly help is more drivers/events, which is a data collection problem,
not a machine learning problem."*

## Suggested interview talking points, in order

1. **The hook:** "I went back to my FYP with three more years of ML
   experience and audited my own old work."
2. **The finding:** classic leakage (SMOTE-before-split, ungrouped splits,
   identifier features, accuracy-only scoring) inflated the original scores
   from a real ~0.57 ROC-AUC to an apparent 98% accuracy.
3. **The fix:** grouped/leave-one-driver-out evaluation, SMOTE inside the CV
   pipeline, dropped identifier columns, balanced metrics.
4. **The model-choice rationale:** logistic regression as an interpretable
   baseline; gradient-boosted trees as the primary model because fatigue's
   physiological signature is non-linear and interaction-heavy.
5. **The follow-through:** three targeted experiments (normalization,
   feature selection, sequence modelling) to test whether the honest result
   was still improvable — all negative, which pins the ceiling on cohort
   size rather than modelling choices.
6. **The takeaway:** a smaller, honest, well-validated result beats a
   larger, leakage-inflated one — and knowing how to tell the difference,
   and how to keep testing a "final" number instead of accepting it, is the
   actual skill being demonstrated.
