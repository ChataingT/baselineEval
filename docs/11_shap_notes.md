# SHAP Explainability — Notes and TODOs

## Status: PLACEHOLDER (Step 7 in the pipeline)

SHAP (SHapley Additive exPlanations) provides model-agnostic feature
importance scores that decompose each prediction into individual feature
contributions. This step is planned but not yet implemented.

## Planned implementation

### Per target × best representation:

1. Retrain the best (model, representation) pipeline on the full training
   set (N ≈ 99).
2. Choose the appropriate SHAP explainer:
   - **LightGBM**: `shap.TreeExplainer(model)` — exact, fast.
   - **Lasso**: `shap.LinearExplainer(model, X_train)` — exact for linear
     models.
3. Compute SHAP values for all training subjects.
4. Generate plots:
   - **Beeswarm plot** (global feature importance + direction of effect)
   - **Bar plot** (top-20 features by mean |SHAP|)
5. Save SHAP values as CSV for reproducibility.

### Output files (planned)

```
results/{representation}/shap/{target}/
├── shap_values.csv              # (n_subjects × n_features) SHAP matrix
├── shap_beeswarm.png + _data.csv
├── shap_bar_top20.png + _data.csv
```

## Dependencies

```bash
pip install shap
```

## Open questions

- Should SHAP be computed on the full training set or on the held-out test
  set? Full training is more informative for feature importance; test set
  shows what drove individual predictions.
- For the embedding representation, SHAP on embedding dimensions is hard
  to interpret (they have no human-readable names). Consider using only
  the kinematic representations for SHAP.
- Should per-subject SHAP waterfall plots be generated, or only global
  summaries?
