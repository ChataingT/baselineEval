# Preprocessing Pipeline

## Overview

Before entering the model, the raw feature matrix passes through a
**5-step scikit-learn Pipeline** that filters unreliable features, imputes
remaining missing values, removes redundancy, and scales the data. The
pipeline is always **fitted inside CV folds** to prevent information leakage.

## Steps

### 1. Missingness filter

- **Threshold**: > 30% NaN across subjects → feature is dropped.
- **Rationale**: Features with too many missing values are unreliable for
  modelling. This threshold is generous enough to keep most metrics while
  removing those with widespread tracking failures.
- **Implementation**: `MissingnessFilter` (custom transformer in
  `preprocessing.py`).

### 2. Median imputation

- **Method**: `SimpleImputer(strategy="median")` from scikit-learn.
- **Rationale**: After removing high-missingness features, remaining NaNs
  are imputed with the median (robust to outliers, unlike mean imputation).

### 3. Near-zero-variance filter

- **Threshold**: Features where > 95% of values are identical are removed.
- **Rationale**: Near-constant features carry little predictive information
  and can cause numerical instabilities.
- **Implementation**: `NearZeroVarianceFilter` (custom transformer).

### 4. Correlation filter

- **Threshold**: When two features have |Pearson r| > 0.95, one is dropped
  (the one encountered second in column order).
- **Rationale**: Highly correlated features are redundant. Removing one
  reduces multicollinearity without loss of information.
- **Implementation**: `CorrelationFilter` (custom transformer).

### 5. Robust scaling

- **Method**: `RobustScaler()` from scikit-learn (centres by median, scales
  by IQR).
- **Rationale**: More robust to outliers than StandardScaler. Important
  for Lasso (L1-penalised models are sensitive to feature scale).

## PCA (optional post-pipeline step)

After scaling, PCA is applied to retain **95% of explained variance**.

- **Implementation**: `sklearn.decomposition.PCA(n_components=0.95,
  svd_solver="full")`.
- **Skip condition**: PCA is skipped if the number of features after
  filtering is below **64** (relevant for the embedding representation
  which may already be low-dimensional).
- **PCA is fitted per CV fold** — never on the full dataset.

### PCA explained variance plot

The function `plot_pca_explained_variance()` generates:
- A **scree plot** (individual explained variance ratio per component) +
  **cumulative variance curve** with a horizontal line at 95%.
- A companion CSV (`pca_explained_variance.csv`) with columns:
  `component`, `explained_variance`, `explained_variance_ratio`,
  `cumulative_ratio`.

This plot is produced once on the full training set (informational only;
actual PCA is refitted per CV fold).

## Feature selection report

`generate_feature_selection_report()` runs the full 5-step pipeline on the
training set and records which features were dropped and why. Output:
`feature_selection_report.csv` with columns: `feature`, `kept`,
`drop_reason`.

This is **informational only** — the actual filtering during CV is done
independently per fold.
