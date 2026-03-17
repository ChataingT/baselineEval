"""
Held-out evaluation on the pre-defined test split (N ≈ 20).

Workflow
--------
1. **Model selection** — identify the best model from CV results (highest
   mean AUC-ROC for classification; lowest mean RMSE for regression).
2. **Retrain** — fit the 3-step preprocessing→model pipeline on the entire
   training set (N ≈ 102) using the best hyperparameters found in CV.
3. **Evaluate** — compute metrics on the held-out test set.
4. **Confidence intervals** — exact binomial CIs for classification accuracy;
   bootstrap CIs for regression metrics.
5. **SHAP** — for motor_only and full_kinematic representations, compute
   SHAP feature importances on the training set using the fitted model.
   Saves bar chart (mean |SHAP|) and beeswarm (per-subject values), plus
   a raw ``shap_values.csv``.  Skipped for the embedding representation.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_N_RESAMPLES,
    RANDOM_STATE,
)
from .models import build_classification_models, build_regression_models
from .preprocessing import build_preprocessing_pipeline, get_pipeline_feature_names
from .config import REPR_EMBEDDING

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Best model selector
# ─────────────────────────────────────────────────────────────

def select_best_model_from_cv(
    df_cv: pd.DataFrame,
    task_type: str,
) -> tuple[str, dict]:
    """Pick the best model from CV results.

    Parameters
    ----------
    df_cv : DataFrame
        Produced by ``run_repeated_nested_cv_{classification,regression}``.
    task_type : ``"classification"`` or ``"regression"``.

    Returns
    -------
    (model_id, summary_dict)
        *model_id* is e.g. ``"lgbm"`` or ``"lasso"``.
        *summary_dict* contains aggregated metric means.
    """
    if task_type == "classification":
        summary = (
            df_cv.groupby("model")["auc_roc"]
            .agg(["mean", "std"])
            .reset_index()
        )
        best_row = summary.loc[summary["mean"].idxmax()]
        best_model = best_row["model"]
        logger.info(
            "Best classification model: %s (AUC=%.4f ± %.4f)",
            best_model, best_row["mean"], best_row["std"],
        )
    else:
        summary = (
            df_cv.groupby("model")["rmse"]
            .agg(["mean", "std"])
            .reset_index()
        )
        best_row = summary.loc[summary["mean"].idxmin()]
        best_model = best_row["model"]
        logger.info(
            "Best regression model: %s (RMSE=%.4f ± %.4f)",
            best_model, best_row["mean"], best_row["std"],
        )
    return best_model, best_row.to_dict()


def _parse_best_params(df_cv: pd.DataFrame, model_id: str) -> dict:
    """Extract most frequent best_params dict from CV runs for *model_id*."""
    import ast

    sub = df_cv.loc[df_cv["model"] == model_id, "best_params"]
    # Find the most common parameter set
    most_common = sub.value_counts().idxmax()
    return ast.literal_eval(most_common)


# ─────────────────────────────────────────────────────────────
# Confidence intervals
# ─────────────────────────────────────────────────────────────

def _binomial_ci(n_correct: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval."""
    from scipy.stats import beta as beta_dist

    if n_total == 0:
        return (0.0, 1.0)
    lo = beta_dist.ppf(alpha / 2, n_correct, n_total - n_correct + 1) if n_correct > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, n_correct + 1, n_total - n_correct) if n_correct < n_total else 1.0
    return (float(lo), float(hi))


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci: float = BOOTSTRAP_CI,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a regression metric.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(random_state)
    point = float(metric_fn(y_true, y_pred))
    scores = []
    n = len(y_true)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
    lo = float(np.percentile(scores, (1 - ci) / 2 * 100))
    hi = float(np.percentile(scores, (1 + ci) / 2 * 100))
    return point, lo, hi


# ─────────────────────────────────────────────────────────────
# SHAP helper
# ─────────────────────────────────────────────────────────────

def _compute_shap(
    pipe: Pipeline,
    X_train: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    target_name: str,
    task_type: str,
    model_id: str = "",
) -> None:
    """Compute and save SHAP values for the fitted held-out model.

    Parameters
    ----------
    pipe : fitted sklearn Pipeline (preprocessing + model).
    X_train : raw training features (N × F) — preprocessing is applied inside.
    feature_names : original feature column names (length F).
    output_dir : directory for this target (a ``shap/`` sub-folder is created).
    target_name : used in log messages only.
    task_type : ``"classification"`` or ``"regression"``.
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        logger.warning(
            "shap not installed — skipping SHAP for %s. Install with: pip install shap",
            target_name,
        )
        return

    import shap  # type: ignore[import]

    shap_dir = output_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    # Transform training data through all steps except the final model
    feat_names_out = get_pipeline_feature_names(pipe[:-1], feature_names)
    X_preproc = pipe[:-1].transform(X_train)

    model = pipe.named_steps["model"]
    model_name = type(model).__name__.lower()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "lgbm" in model_name or "lightgbm" in model_name:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_preproc)
        else:
            explainer = shap.LinearExplainer(model, X_preproc)
            sv = explainer.shap_values(X_preproc)

    # For binary classification shap_values may be a list (one array per class)
    if isinstance(sv, list):
        sv = sv[1]  # positive class

    # Save raw SHAP matrix
    pd.DataFrame(sv, columns=feat_names_out).to_csv(
        shap_dir / "shap_values.csv", index=False
    )

    # Plots
    from .plotting import plot_shap_summary
    shap_source = f"Train data — {model_id}" if model_id else ""
    plot_shap_summary(sv, X_preproc, feat_names_out, shap_dir, data_source=shap_source)
    logger.info("SHAP saved → %s", shap_dir)


# ─────────────────────────────────────────────────────────────
# Held-out evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_held_out_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_meta_test: pd.DataFrame,
    df_cv: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    target_name: str,
    representation: str = "unknown",
    random_state: int = RANDOM_STATE,
) -> dict:
    """Retrain best model from CV on full training set, evaluate on held-out.

    Pipeline: MissingnessFilter → SimpleImputer → RobustScaler → Model.
    SHAP feature importances are computed on the training set when
    ``representation`` is not ``"embedding"``.

    Returns
    -------
    dict with point metrics + CIs.
    """
    out = output_dir / "held_out" / "classification" / target_name
    out.mkdir(parents=True, exist_ok=True)

    best_model_id, _ = select_best_model_from_cv(df_cv, "classification")
    best_params = _parse_best_params(df_cv, best_model_id)

    model_defs = build_classification_models(random_state=random_state)
    estimator, _ = model_defs[best_model_id]

    # Apply best params
    for k, v in best_params.items():
        # param keys are prefixed with "model__"
        if k.startswith("model__"):
            setattr(estimator, k.replace("model__", ""), v)

    preproc = build_preprocessing_pipeline()
    steps = list(preproc.steps) + [("model", estimator)]
    pipe = Pipeline(steps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

    # Metrics
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = np.nan
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan

    # Binomial CI on accuracy
    n_correct = int(tp + tn)
    n_total = len(y_test)
    acc_lo, acc_hi = _binomial_ci(n_correct, n_total)

    results = {
        "representation": representation,
        "model": best_model_id,
        "target": target_name,
        "data_source": "test_held_out",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "auc_roc": auc,
        "balanced_acc": bacc,
        "f1_macro": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "accuracy_ci_lo": acc_lo,
        "accuracy_ci_hi": acc_hi,
        "best_params": str(best_params),
    }

    # Save
    pd.DataFrame([results]).to_csv(out / "held_out_results.csv", index=False)

    # Predictions
    uuids = df_meta_test["uuid"].values if "uuid" in df_meta_test.columns else np.arange(len(y_test))
    pred_df = pd.DataFrame({
        "uuid": uuids,
        "model": best_model_id,
        "data_source": "test_held_out",
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob_pos": y_prob,
    })
    pred_df.to_csv(out / "held_out_predictions.csv", index=False)

    logger.info(
        "Held-out classification (%s | %s): model=%s AUC=%.3f BAcc=%.3f Acc=%.3f [%.3f, %.3f]",
        representation, target_name, best_model_id, auc, bacc, accuracy, acc_lo, acc_hi,
    )

    # SHAP (skipped for embedding representation)
    if representation != REPR_EMBEDDING:
        _compute_shap(pipe, X_train, feature_names, out, target_name, "classification",
                      model_id=best_model_id)

    return results


def evaluate_held_out_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_meta_test: pd.DataFrame,
    df_cv: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    target_name: str,
    representation: str = "unknown",
    random_state: int = RANDOM_STATE,
) -> dict:
    """Retrain best model from CV on full training set, evaluate on held-out.

    Pipeline: MissingnessFilter → SimpleImputer → RobustScaler → Model.
    SHAP feature importances are computed on the training set when
    ``representation`` is not ``"embedding"``.

    Returns
    -------
    dict with point metrics + bootstrap CIs.
    """
    out = output_dir / "held_out" / "regression" / target_name
    out.mkdir(parents=True, exist_ok=True)

    best_model_id, _ = select_best_model_from_cv(df_cv, "regression")
    best_params = _parse_best_params(df_cv, best_model_id)

    model_defs = build_regression_models(random_state=random_state)
    estimator, _ = model_defs[best_model_id]

    for k, v in best_params.items():
        if k.startswith("model__"):
            setattr(estimator, k.replace("model__", ""), v)

    preproc = build_preprocessing_pipeline()
    steps = list(preproc.steps) + [("model", estimator)]
    pipe = Pipeline(steps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

    # Metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    sp_r, _ = spearmanr(y_test, y_pred)
    pe_r, _ = pearsonr(y_test, y_pred)

    # Bootstrap CIs
    rmse_pt, rmse_lo, rmse_hi = _bootstrap_ci(
        y_test, y_pred, lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
    )
    mae_pt, mae_lo, mae_hi = _bootstrap_ci(
        y_test, y_pred, mean_absolute_error,
    )
    r2_pt, r2_lo, r2_hi = _bootstrap_ci(
        y_test, y_pred, r2_score,
    )

    results = {
        "representation": representation,
        "model": best_model_id,
        "target": target_name,
        "data_source": "test_held_out",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "rmse": rmse,
        "rmse_ci_lo": rmse_lo,
        "rmse_ci_hi": rmse_hi,
        "mae": mae,
        "mae_ci_lo": mae_lo,
        "mae_ci_hi": mae_hi,
        "r2": r2,
        "r2_ci_lo": r2_lo,
        "r2_ci_hi": r2_hi,
        "spearman_r": float(sp_r),
        "pearson_r": float(pe_r),
        "best_params": str(best_params),
    }

    pd.DataFrame([results]).to_csv(out / "held_out_results.csv", index=False)

    uuids = df_meta_test["uuid"].values if "uuid" in df_meta_test.columns else np.arange(len(y_test))
    pred_df = pd.DataFrame({
        "uuid": uuids,
        "model": best_model_id,
        "data_source": "test_held_out",
        "y_true": y_test,
        "y_pred": y_pred,
        "residual": y_pred - y_test,
    })
    pred_df.to_csv(out / "held_out_predictions.csv", index=False)

    logger.info(
        "Held-out regression (%s | %s): model=%s RMSE=%.3f [%.3f, %.3f] R²=%.3f [%.3f, %.3f] ρ=%.3f",
        representation, target_name, best_model_id, rmse, rmse_lo, rmse_hi, r2, r2_lo, r2_hi, sp_r,
    )

    # SHAP (skipped for embedding representation)
    if representation != REPR_EMBEDDING:
        _compute_shap(pipe, X_train, feature_names, out, target_name, "regression",
                      model_id=best_model_id)

    return results
