"""
Model definitions and hyperparameter search spaces.

Two models for each task type:
  Classification: LGBMClassifier + LogisticRegression (L1 / Lasso)
  Regression:     LGBMRegressor  + Lasso (ElasticNet with l1_ratio=1)

Hyperparameters are tuned via ``RandomizedSearchCV(n_iter=50)`` inside
the inner loop of nested cross-validation.  See ``cross_validation.py``
and ``docs/04_models_and_tuning.md`` for the full strategy.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression

from .config import RANDOM_STATE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Classification models
# ─────────────────────────────────────────────────────────────

def build_classification_models(
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple]:
    """Return ``{model_id: (estimator, param_distributions)}`` for classification.

    Models
    ------
    lgbm
        LGBMClassifier with ``is_unbalance=True`` to handle class imbalance.
        Gradient-boosted decision trees — captures non-linear feature
        interactions and is robust to irrelevant features.

    lasso
        L1-regularised logistic regression (``penalty='l1'``).  Linear
        baseline that produces sparse solutions — useful for identifying
        which features drive classification in a high-dimensional,
        low-sample regime.
    """
    models: dict[str, tuple] = {}

    # ── LightGBM ─────────────────────────────────────────────
    try:
        from lightgbm import LGBMClassifier

        lgbm_est = LGBMClassifier(
            is_unbalance=True,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )
        lgbm_params = {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 5, 7, -1],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__num_leaves": [15, 31, 63],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_samples": [5, 10, 20],
        }
        models["lgbm"] = (lgbm_est, lgbm_params)
    except ImportError:
        logger.warning("lightgbm not available; skipping LGBM classifier")

    # ── Lasso (L1 logistic regression) ───────────────────────
    lasso_est = LogisticRegression(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    lasso_params = {
        "model__C": np.logspace(-3, 2, 20).tolist(),
    }
    models["lasso"] = (lasso_est, lasso_params)

    return models


# ─────────────────────────────────────────────────────────────
# Regression models
# ─────────────────────────────────────────────────────────────

def build_regression_models(
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple]:
    """Return ``{model_id: (estimator, param_distributions)}`` for regression.

    Models
    ------
    lgbm
        LGBMRegressor.  Same strengths as the classifier variant — captures
        non-linear relationships and feature interactions.

    lasso
        Lasso regression (L1-regularised linear regression).  Produces
        sparse solutions, selecting only the most predictive features.
    """
    models: dict[str, tuple] = {}

    # ── LightGBM ─────────────────────────────────────────────
    try:
        from lightgbm import LGBMRegressor

        lgbm_est = LGBMRegressor(
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )
        lgbm_params = {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 5, 7, -1],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__num_leaves": [15, 31, 63],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_samples": [5, 10, 20],
        }
        models["lgbm"] = (lgbm_est, lgbm_params)
    except ImportError:
        logger.warning("lightgbm not available; skipping LGBM regressor")

    # ── Lasso ────────────────────────────────────────────────
    lasso_est = Lasso(
        max_iter=5000,
        random_state=random_state,
    )
    lasso_params = {
        "model__alpha": np.logspace(-3, 3, 20).tolist(),
    }
    models["lasso"] = (lasso_est, lasso_params)

    return models
