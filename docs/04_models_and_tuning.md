# Models and Hyperparameter Tuning

## Models

Two model families are used for each task (classification and regression).
They were chosen to balance interpretability with predictive power:

### LightGBM

- **Classification**: `LGBMClassifier(is_unbalance=True, verbosity=-1)`
- **Regression**: `LGBMRegressor(verbosity=-1)`
- **Strengths**: Handles non-linear feature interactions, robust to
  irrelevant features, fast training. The `is_unbalance` flag automatically
  adjusts class weights for the ASD/TD imbalance.

### Lasso (L1-penalised linear model)

- **Classification**: `LogisticRegression(penalty="l1", solver="saga",
  max_iter=5000)`
- **Regression**: `Lasso(max_iter=5000)`
- **Strengths**: Built-in feature selection (L1 drives coefficients to
  zero), fully interpretable coefficients, computationally lightweight.

## Hyperparameter search spaces

### LightGBM

| Parameter | Search space |
|-----------|-------------|
| `n_estimators` | {100, 300, 500} |
| `max_depth` | {3, 5, 7, −1} (−1 = unlimited) |
| `learning_rate` | {0.01, 0.05, 0.1, 0.2} |
| `num_leaves` | {15, 31, 63} |
| `subsample` | {0.6, 0.8, 1.0} |
| `colsample_bytree` | {0.6, 0.8, 1.0} |
| `min_child_samples` | {5, 10, 20} |

All parameters are prefixed with `model__` for the scikit-learn Pipeline.

### Lasso

| Parameter | Search space |
|-----------|-------------|
| `C` (classification) | 20 values log-spaced in $[10^{-3}, 10^2]$ |
| `alpha` (regression) | 20 values log-spaced in $[10^{-3}, 10^3]$ |

## Tuning strategy: RandomizedSearchCV

Exhaustive grid search over the full LightGBM space would require
$3 × 4 × 4 × 3 × 3 × 3 × 3 = 3{,}888$ configurations — too expensive for
nested CV. Instead, we use **RandomizedSearchCV** with `n_iter=50`, which
samples 50 random configurations per inner loop and typically finds
near-optimal parameters.

### Inner loop

- **CV scheme**: `StratifiedKFold(n_splits=3)` inside each outer fold.
- **Scoring**: `roc_auc` (classification) or `neg_root_mean_squared_error`
  (regression).
- **Refit**: `True` — the best configuration is refitted on the full inner
  training data.
- **n_iter**: 50 random parameter samples per search.

### Why not Bayesian optimisation?

RandomizedSearchCV is simpler, parallelisable, and introduces no additional
dependencies. With 50 iterations and 3 inner folds (= 150 model fits per
outer fold), it explores the space sufficiently for these dataset sizes.
Bayesian optimisation (e.g. Optuna) would be beneficial for larger search
spaces or longer training times.

## GPU support

LightGBM can use CUDA for training if `--use-gpu` is passed. This requires
the CUDA-enabled LightGBM build. The SLURM script requests one GPU by
default.
