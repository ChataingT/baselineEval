# Cross-Validation Scheme

## Design: Repeated Nested Cross-Validation

The pipeline uses **repeated nested (double) cross-validation** to obtain
unbiased performance estimates while simultaneously tuning hyperparameters.

### Outer loop: Performance estimation

- **Method**: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)`
- **Total outer evaluations**: 5 × 10 = **50 fold scores per model**
- **Stratification**: By class label (classification) or by quantile-binned
  target (regression, 5 bins).
- **Purpose**: Each outer fold provides a held-out evaluation of the best
  model found by the inner loop. The 50 scores give a robust distribution
  for statistical comparison.

### Inner loop: Hyperparameter tuning

- **Method**: `RandomizedSearchCV(cv=StratifiedKFold(3), n_iter=50)`
- **Scoring**: `roc_auc` (classification), `neg_root_mean_squared_error`
  (regression).
- **Refit**: `True` — the best configuration is refitted on the full outer
  training partition.
- **Purpose**: Finds the best hyperparameters without using the outer
  validation fold.

### Why nested CV?

Standard CV (tuning + testing on the same folds) produces optimistically
biased performance estimates because the test fold indirectly influences the
model through hyperparameter selection. Nested CV separates these concerns:

```
Outer loop: performance estimation
 └── Inner loop: hyperparameter tuning
      └── Preprocessing fitted here (no leakage)
```

### Why repeated (10 repeats)?

A single 5-fold split produces only 5 scores — too few for reliable
statistical tests. Repeating with 10 different random shuffles yields 50
scores, providing more stable estimates of mean performance and enabling
Wilcoxon signed-rank tests with adequate power.

## Preprocessing within CV

The full preprocessing pipeline (missingness filter → median imputer →
near-zero-variance filter → correlation filter → robust scaler) is
embedded as part of the scikit-learn `Pipeline` and fitted only on the
inner training data. PCA (when applied) is also fitted within the pipeline.

This prevents any form of **data leakage**:
- Imputation values are computed from the training fold only.
- Feature selection decisions are based on the training fold only.
- Scaling parameters are fit on the training fold only.
- PCA components are learned from the training fold only.

## Stratification for regression

Continuous regression targets (CSS, SA, RRB) are binned into 5 quantiles
for stratification purposes. This ensures that each fold contains a
representative distribution of severity scores, preventing situations where
all high-severity subjects end up in one fold.

## Output files

Per target × representation:

| File | Description |
|------|-------------|
| `cv_results.csv` | Per-fold per-model metrics (repeat, fold, model, all metrics, best_params) |
| `predictions_per_subject.csv` | Per-subject per-fold predictions (uuid, y_true, y_pred, y_prob) |
