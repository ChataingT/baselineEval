# Held-Out Validation

## Purpose

After identifying the best model through cross-validation, the pipeline
performs a **final independent evaluation** on the pre-defined held-out
test set (~20 subjects) that was never used during CV.

## Workflow

1. **Select best model**: From the CV results, identify the model (LightGBM
   or Lasso) with the highest mean AUC-ROC (classification) or lowest mean
   RMSE (regression).
2. **Extract best hyperparameters**: Use the most frequently selected
   hyperparameter configuration across the 50 CV folds.
3. **Retrain**: Fit the full pipeline (preprocessing → PCA → model) on the
   entire training set (~99 subjects).
4. **Evaluate**: Compute all metrics on the held-out test set (~20 subjects).
5. **Confidence intervals**: Quantify estimation uncertainty.

## Confidence intervals

### Classification: Exact binomial (Clopper-Pearson)

For accuracy on the held-out set:

$$
CI_{1-\alpha} = \left[ B^{-1}\!\left(\frac{\alpha}{2};\, k,\, n-k+1\right), \;
B^{-1}\!\left(1 - \frac{\alpha}{2};\, k+1,\, n-k\right) \right]
$$

where $B^{-1}$ is the inverse of the Beta CDF, $k$ is the number of correct
predictions, and $n$ is the total number of test subjects.

**Why exact binomial?** With only ~20 test subjects, asymptotic (normal)
approximations are unreliable. The Clopper-Pearson interval is conservative
and exact.

**Note**: CIs on ~20 subjects will be **wide** (e.g. 15/20 correct →
95% CI [0.51, 0.95]). This is expected and honest.

### Regression: Percentile bootstrap

For RMSE, MAE, and R²:

1. Resample  the test set (with replacement) 1000 times.
2. Compute the metric on each bootstrap sample.
3. Report the 2.5th and 97.5th percentiles as the 95% CI.

**Settings**: 1000 resamples, seed = 42.

## Output files

Per target:

| File | Description |
|------|-------------|
| `held_out_results.csv` | Point estimates + CIs for all metrics |
| `held_out_predictions.csv` | Per-subject predictions (uuid, y_true, y_pred, y_prob) |

## Interpretation caveats

- **Small test set**: ~20 subjects is insufficient for precise performance
  estimation. The CIs will be wide. Treat held-out results as a
  **sanity check**, not as the primary evidence.
- **Primary evidence is the CV**: The 50-fold CV scores provide much more
  stable estimates than the 20-subject held-out set.
- **No re-tuning**: Held-out evaluation is a single pass — no further
  model selection or threshold calibration on the test data.
