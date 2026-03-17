# Outputs and Plots

## Directory structure

```
results/
├── {representation}/            # motor_only, full_kinematic, embedding
│   ├── classification/{target}/
│   │   ├── cv_results.csv
│   │   └── predictions_per_subject.csv
│   ├── regression/{target}/
│   │   ├── cv_results.csv
│   │   └── predictions_per_subject.csv
│   ├── held_out/classification/{target}/
│   │   ├── held_out_results.csv
│   │   └── held_out_predictions.csv
│   ├── held_out/regression/{target}/
│   │   ├── held_out_results.csv
│   │   └── held_out_predictions.csv
│   ├── plots/classification/{target}/
│   │   ├── roc_curves.png            + roc_curves_data.csv
│   │   ├── confusion_matrices.png    + confusion_matrices_data.csv
│   │   ├── cv_metric_boxplots.png    + cv_metric_boxplots_data.csv
│   │   └── hyperparams_{model}.png   + hyperparameter_distributions_data.csv
│   ├── plots/regression/{target}/
│   │   ├── pred_vs_actual_{model}.png + pred_vs_actual_data.csv
│   │   ├── residuals_{model}.png      + residuals_data.csv
│   │   ├── cv_metric_boxplots.png     + cv_metric_boxplots_data.csv
│   │   └── hyperparams_{model}.png    + hyperparameter_distributions_data.csv
│   ├── preprocessing/
│   │   ├── feature_selection_report.csv
│   │   ├── pca_explained_variance.png + pca_explained_variance.csv
│   │   └── pca_explained_variance_data.csv
├── comparison/{target}/
│   ├── comparison_results.csv
│   └── ...
├── comparison/all_comparison_results.csv
├── plots/comparison/{target}/
│   ├── representation_comparison.png + representation_comparison_data.csv
├── plots/held_out/
│   ├── held_out_summary_classification.png + _data.csv
│   └── held_out_summary_regression.png     + _data.csv
```

## Plot descriptions and reproduction

### ROC curves (`roc_curves.png`)

- **Content**: Mean ROC curve ± 1 std band per model, from CV predictions.
- **Companion data**: `roc_curves_data.csv` (columns: model, fpr, mean_tpr,
  std_tpr, mean_auc, std_auc).
- **Reproduction**: Load CSV → plot mean_tpr vs fpr with fill_between for
  ± std_tpr.

### Confusion matrices (`confusion_matrices.png`)

- **Content**: Aggregated confusion matrix (all CV folds combined) per model.
- **Companion data**: `confusion_matrices_data.csv` (model, true_label,
  pred_label, count).
- **Reproduction**: Pivot table from CSV → heatmap.

### CV metric boxplots (`cv_metric_boxplots.png`)

- **Content**: Box plots of 50 outer-fold scores per model, one panel per
  metric.
- **Companion data**: `cv_metric_boxplots_data.csv` (metric, model, value).
- **Reproduction**: Group by metric and model → box plot.

### Hyperparameter distributions (`hyperparams_{model}.png`)

- **Content**: Histograms of best hyperparameter values across 50 CV folds.
- **Companion data**: `hyperparameter_distributions_data.csv` (model, param,
  value).
- **Reproduction**: Filter by model and param → histogram.

### Predicted vs actual (`pred_vs_actual_{model}.png`)

- **Content**: Scatter plot of CV predictions vs true values with 1:1 line
  and RMSE / ρ / r annotations.
- **Companion data**: `pred_vs_actual_data.csv` (model, y_true, y_pred).
- **Reproduction**: Scatter from CSV + polyfit for regression line.

### Residuals (`residuals_{model}.png`)

- **Content**: Two panels — histogram of residuals + scatter of residuals
  vs predicted.
- **Companion data**: `residuals_data.csv` (model, y_pred, residual).
- **Reproduction**: Histogram + scatter from CSV.

### Held-out summary (`held_out_summary_{task}.png`)

- **Content**: Bar chart of held-out metrics across targets with CI error
  bars (when available).
- **Companion data**: `held_out_summary_{task}_data.csv`.

### Representation comparison (`representation_comparison.png`)

- **Content**: Grouped bar chart of mean CV metric per representation with
  significance markers (* for p_corrected < 0.05).
- **Companion data**: `representation_comparison_data.csv` (representation,
  mean, std).
- **Reproduction**: Bar chart from CSV + add significance brackets from
  `comparison_results.csv`.
