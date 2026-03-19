"""
Repeated nested cross-validation (5 outer folds × 10 repeats).

Outer loop: ``RepeatedStratifiedKFold(n_splits=5, n_repeats=10)``
  — produces 50 outer evaluations for robust statistical comparison.

Inner loop: ``StratifiedKFold(3)`` + ``RandomizedSearchCV(n_iter=50)``
  — tunes hyperparameters on the training partition of each outer fold.

The preprocessing pipeline (missingness → impute → scale) is fitted inside
the inner loop only, preventing any form of data leakage.

Classification metrics per fold: AUC-ROC, balanced accuracy, F1-macro,
    sensitivity, specificity
Regression metrics per fold: RMSE, MAE, R², Spearman ρ, Pearson r
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from .config import (
    N_INNER_FOLDS,
    N_ITER,
    N_OUTER_FOLDS,
    N_REPEATS,
    RANDOM_STATE,
    REG_STRAT_BINS,
)
from .models import build_classification_models, build_regression_models
from .preprocessing import build_preprocessing_pipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Stratification helpers
# ─────────────────────────────────────────────────────────────

def _build_regression_strat(y: np.ndarray, n_bins: int = REG_STRAT_BINS) -> np.ndarray:
    """Bin continuous targets into quantiles for stratified splitting."""
    y_series = pd.Series(y)
    try:
        bins = pd.qcut(y_series, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(y_series, bins=n_bins, labels=False)
    return bins.fillna(-1).astype(int).values


# ─────────────────────────────────────────────────────────────
# Single outer fold
# ─────────────────────────────────────────────────────────────

def _run_one_fold_classification(
    fold_idx: int,
    repeat_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_defs: dict,
    n_inner: int,
    n_iter: int,
    random_state: int,
) -> list[dict]:
    """Run all classification models for one outer fold."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    n_classes = len(np.unique(y_train))
    multiclass = n_classes > 2

    inner_cv = StratifiedKFold(
        n_splits=n_inner, shuffle=True, random_state=random_state,
    )
    fold_results = []

    for model_id, (estimator, param_dist) in model_defs.items():
        est = copy.deepcopy(estimator)
        preproc = build_preprocessing_pipeline()
        steps = list(preproc.steps) + [("model", est)]
        pipe = Pipeline(steps)

        scoring = "f1_macro" if multiclass else "roc_auc"
        search = RandomizedSearchCV(
            pipe, param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
            error_score=np.nan,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        best = search.best_estimator_

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_prob = best.predict_proba(X_test)
            y_pred = best.predict(X_test)

        if multiclass:
            y_prob_pos = np.full(len(y_test), np.nan, dtype=float)
        else:
            y_prob_pos = y_prob[:, 1]

        if multiclass:
            auc = np.nan
        else:
            try:
                auc = roc_auc_score(y_test, y_prob_pos)
            except Exception:
                auc = np.nan

        bacc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        if multiclass:
            labels = sorted(np.unique(y_test))
        else:
            labels = [0, 1]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        else:
            sensitivity = np.nan
            specificity = np.nan

        fold_results.append({
            "repeat": repeat_idx,
            "fold": fold_idx,
            "model": model_id,
            "auc_roc": auc,
            "balanced_acc": bacc,
            "f1_macro": f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "best_params": str(search.best_params_),
            "y_prob": y_prob_pos.tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "test_indices": test_idx.tolist(),
        })
        logger.debug(
            "  Repeat %d Fold %d | %s: AUC=%.3f BAcc=%.3f",
            repeat_idx, fold_idx, model_id, auc, bacc,
        )

    return fold_results


def _run_one_fold_regression(
    fold_idx: int,
    repeat_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_defs: dict,
    n_inner: int,
    n_iter: int,
    random_state: int,
) -> list[dict]:
    """Run all regression models for one outer fold."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    inner_cv = StratifiedKFold(
        n_splits=n_inner, shuffle=True, random_state=random_state,
    )
    # Stratify inner CV by binned target for regression
    y_train_strat = _build_regression_strat(y_train)

    fold_results = []

    for model_id, (estimator, param_dist) in model_defs.items():
        est = copy.deepcopy(estimator)
        preproc = build_preprocessing_pipeline()
        steps = list(preproc.steps) + [("model", est)]
        pipe = Pipeline(steps)

        # Inner CV uses binned target for stratification
        class _StratifiedFromBins:
            """Adapter: StratifiedKFold that uses externally provided bins."""
            def __init__(self, cv, bins):
                self._cv = cv
                self._bins = bins
            def split(self, X, y=None, groups=None):
                return self._cv.split(X, self._bins, groups)
            def get_n_splits(self, X=None, y=None, groups=None):
                return self._cv.get_n_splits()

        inner_cv_adapter = _StratifiedFromBins(inner_cv, y_train_strat)

        search = RandomizedSearchCV(
            pipe, param_dist,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv_adapter,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
            error_score=np.nan,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        best = search.best_estimator_

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = best.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp_r, sp_p = spearmanr(y_test, y_pred)
            pe_r, pe_p = pearsonr(y_test, y_pred)

        fold_results.append({
            "repeat": repeat_idx,
            "fold": fold_idx,
            "model": model_id,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman_r": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
            "best_params": str(search.best_params_),
            "y_pred": y_pred.tolist(),
            "y_test": y_test.tolist(),
            "test_indices": test_idx.tolist(),
        })
        logger.debug(
            "  Repeat %d Fold %d | %s: RMSE=%.3f R²=%.3f ρ=%.3f",
            repeat_idx, fold_idx, model_id, rmse, r2, sp_r,
        )

    return fold_results


# ─────────────────────────────────────────────────────────────
# Main CV runners
# ─────────────────────────────────────────────────────────────

def run_repeated_nested_cv_classification(
    X: np.ndarray,
    y: np.ndarray,
    df_meta: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    target_name: str,
    n_outer: int = N_OUTER_FOLDS,
    n_repeats: int = N_REPEATS,
    n_inner: int = N_INNER_FOLDS,
    n_iter: int = N_ITER,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """Run 5×10 repeated nested CV for binary classification.

    Parameters
    ----------
    X : array, shape (n_subjects, n_features)
        Raw (un-preprocessed) feature matrix.
    y : array, shape (n_subjects,)
        Integer-encoded labels (0 = TD, 1 = ASD).
    df_meta : DataFrame
        Clinical metadata (same row order as X).
    feature_names : list[str]
        Feature column names.
    output_dir : Path
        Output directory for CSV results.
    target_name : str
        Name of the target variable.

    Returns
    -------
    DataFrame of per-fold per-model scores.
    """
    out = output_dir / "classification" / target_name
    out.mkdir(parents=True, exist_ok=True)

    model_defs = build_classification_models(random_state=random_state)
    logger.info(
        "Classification CV: %d models, %d×%d nested CV (%d total outer folds)",
        len(model_defs), n_outer, n_repeats, n_outer * n_repeats,
    )

    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_outer, n_repeats=n_repeats, random_state=random_state,
    )
    splits = list(outer_cv.split(X, y))

    # Compute repeat and fold indices
    repeat_fold_indices = [
        (i // n_outer, i % n_outer) for i in range(len(splits))
    ]

    all_results = Parallel(n_jobs=min(n_jobs, len(splits)), backend="loky")(
        delayed(_run_one_fold_classification)(
            fold_idx=fold_idx,
            repeat_idx=repeat_idx,
            train_idx=train,
            test_idx=test,
            X=X,
            y=y,
            model_defs=model_defs,
            n_inner=n_inner,
            n_iter=n_iter,
            random_state=random_state,
        )
        for (train, test), (repeat_idx, fold_idx) in zip(splits, repeat_fold_indices)
    )

    # Flatten
    rows_full = []
    for fold_res in all_results:
        rows_full.extend(fold_res)

    # ── Per-subject predictions ──────────────────────────────
    uuids = df_meta["uuid"].values if "uuid" in df_meta.columns else np.arange(len(df_meta))
    pred_rows = []
    for r in rows_full:
        test_idx = np.array(r["test_indices"])
        y_test_fold = np.array(r["y_test"])
        y_pred_fold = np.array(r["y_pred"])
        y_prob_fold = np.array(r["y_prob"])
        for idx, yt, yp, ypr in zip(test_idx, y_test_fold, y_pred_fold, y_prob_fold):
            pred_rows.append({
                "repeat": r["repeat"],
                "fold": r["fold"],
                "model": r["model"],
                "uuid": uuids[idx],
                "y_true": int(yt),
                "y_pred": int(yp),
                "y_prob_pos": float(ypr),
                "data_source": "train_cv",
            })
    pd.DataFrame(pred_rows).to_csv(out / "predictions_per_subject.csv", index=False)
    logger.info("  Saved: predictions_per_subject.csv")

    # ── Summary (drop raw arrays) ────────────────────────────
    rows_summary = [
        {k: v for k, v in r.items() if k not in ("y_test", "y_prob", "y_pred", "test_indices")}
        for r in rows_full
    ]
    for row in rows_summary:
        row["data_source"] = "train_cv"
    df_cv = pd.DataFrame(rows_summary)
    df_cv.to_csv(out / "cv_results.csv", index=False)
    logger.info("  CV results: %d rows → %s", len(df_cv), out / "cv_results.csv")

    # Log summary
    if df_cv["auc_roc"].notna().any():
        metric = "auc_roc"
        label = "AUC-ROC"
    else:
        metric = "f1_macro"
        label = "F1-macro"
    summary = df_cv.groupby("model")[metric].agg(["mean", "std"]).reset_index()
    summary.columns = ["model", f"{metric}_mean", f"{metric}_std"]
    logger.info(
        "\n── Classification %s summary (%s) ──\n%s",
        label, target_name, summary.to_string(index=False),
    )
    return df_cv


def run_repeated_nested_cv_regression(
    X: np.ndarray,
    y: np.ndarray,
    df_meta: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    target_name: str,
    n_outer: int = N_OUTER_FOLDS,
    n_repeats: int = N_REPEATS,
    n_inner: int = N_INNER_FOLDS,
    n_iter: int = N_ITER,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """Run 5×10 repeated nested CV for regression.

    Parameters
    ----------
    X, y, df_meta, feature_names, output_dir, target_name
        Same semantics as the classification variant, except *y* is continuous.

    Returns
    -------
    DataFrame of per-fold per-model scores.
    """
    out = output_dir / "regression" / target_name
    out.mkdir(parents=True, exist_ok=True)

    model_defs = build_regression_models(random_state=random_state)
    logger.info(
        "Regression CV: %d models, %d×%d nested CV (%d total outer folds)",
        len(model_defs), n_outer, n_repeats, n_outer * n_repeats,
    )

    # Stratify outer loop by binned target
    y_strat = _build_regression_strat(y)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_outer, n_repeats=n_repeats, random_state=random_state,
    )
    splits = list(outer_cv.split(X, y_strat))
    repeat_fold_indices = [
        (i // n_outer, i % n_outer) for i in range(len(splits))
    ]

    all_results = Parallel(n_jobs=min(n_jobs, len(splits)), backend="loky")(
        delayed(_run_one_fold_regression)(
            fold_idx=fold_idx,
            repeat_idx=repeat_idx,
            train_idx=train,
            test_idx=test,
            X=X,
            y=y,
            model_defs=model_defs,
            n_inner=n_inner,
            n_iter=n_iter,
            random_state=random_state,
        )
        for (train, test), (repeat_idx, fold_idx) in zip(splits, repeat_fold_indices)
    )

    # Flatten
    rows_full = []
    for fold_res in all_results:
        rows_full.extend(fold_res)

    # ── Per-subject predictions ──────────────────────────────
    uuids = df_meta["uuid"].values if "uuid" in df_meta.columns else np.arange(len(df_meta))
    pred_rows = []
    for r in rows_full:
        test_idx = np.array(r["test_indices"])
        y_test_fold = np.array(r["y_test"])
        y_pred_fold = np.array(r["y_pred"])
        for idx, yt, yp in zip(test_idx, y_test_fold, y_pred_fold):
            pred_rows.append({
                "repeat": r["repeat"],
                "fold": r["fold"],
                "model": r["model"],
                "uuid": uuids[idx],
                "y_true": float(yt),
                "y_pred": float(yp),
                "residual": float(yp - yt),
                "data_source": "train_cv",
            })
    pd.DataFrame(pred_rows).to_csv(out / "predictions_per_subject.csv", index=False)
    logger.info("  Saved: predictions_per_subject.csv")

    # ── Summary ──────────────────────────────────────────────
    rows_summary = [
        {k: v for k, v in r.items() if k not in ("y_pred", "y_test", "test_indices")}
        for r in rows_full
    ]
    for row in rows_summary:
        row["data_source"] = "train_cv"
    df_cv = pd.DataFrame(rows_summary)
    df_cv.to_csv(out / "cv_results.csv", index=False)
    logger.info("  CV results: %d rows → %s", len(df_cv), out / "cv_results.csv")

    summary = df_cv.groupby("model")[["rmse", "r2", "spearman_r"]].mean().reset_index()
    logger.info(
        "\n── Regression summary (%s) ──\n%s",
        target_name, summary.to_string(index=False),
    )
    return df_cv
