"""
Preprocessing pipeline and PCA dimensionality reduction.

Three-step sklearn Pipeline (fitted on training fold only — no leakage):
  1. MissingnessFilter  — drop features with > 30 % NaN
  2. SimpleImputer       — median imputation
  3. RobustScaler        — IQR-based scaling (robust to outliers)

The NearZeroVarianceFilter and CorrelationFilter have been removed from the
prediction pipeline: LightGBM handles collinear/low-variance features natively
via its split criterion, and Lasso's L1 penalty drives redundant coefficients
to zero.  Removing these filters preserves the full feature set for SHAP
feature-importance analysis.

PCA is retained as an **exploratory diagnostic only** (see
``plot_pca_explained_variance``).  It is *not* applied during prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import (
    CORR_THRESHOLD,
    DPI,
    MAX_MISSING_FRAC,
    NEAR_ZERO_FRAC,
    PCA_MIN_FEATURES,
    PCA_VARIANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Custom sklearn transformers
# ─────────────────────────────────────────────────────────────

class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Drop columns with more than ``max_missing_frac`` NaN on the training set."""

    def __init__(self, max_missing_frac: float = MAX_MISSING_FRAC):
        self.max_missing_frac = max_missing_frac

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        missing_frac = np.mean(np.isnan(X), axis=0)
        self.keep_mask_ = missing_frac <= self.max_missing_frac
        self.n_dropped_ = int((~self.keep_mask_).sum())
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.keep_mask_))])[self.keep_mask_]
        return np.asarray(input_features)[self.keep_mask_]


class NearZeroVarianceFilter(BaseEstimator, TransformerMixin):
    """Drop features where more than ``frac`` of samples share the same value."""

    def __init__(self, frac: float = NEAR_ZERO_FRAC):
        self.frac = frac

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        keep = []
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                keep.append(False)
                continue
            _, counts = np.unique(valid, return_counts=True)
            max_frac = counts.max() / len(valid)
            keep.append(max_frac <= self.frac)
        self.keep_mask_ = np.array(keep, dtype=bool)
        self.n_dropped_ = int((~self.keep_mask_).sum())
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.keep_mask_))])[self.keep_mask_]
        return np.asarray(input_features)[self.keep_mask_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from each highly correlated pair (|r| > threshold).

    Greedy algorithm ordered by descending variance: the kept feature is
    always the one with higher variance.
    """

    def __init__(self, threshold: float = CORR_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_feat = X.shape[1]
        df_tmp = pd.DataFrame(X)
        corr_mat = df_tmp.corr(method="pearson").values
        variances = np.nanvar(X, axis=0)
        order = np.argsort(-variances)

        to_drop: set[int] = set()
        dropped_pairs: list[tuple[int, int]] = []

        for i_pos, i in enumerate(order):
            if i in to_drop:
                continue
            for j in order[i_pos + 1:]:
                if j in to_drop:
                    continue
                r = corr_mat[i, j]
                if np.isnan(r):
                    continue
                if abs(r) > self.threshold:
                    to_drop.add(j)
                    dropped_pairs.append((int(i), int(j)))

        self.keep_mask_ = np.array(
            [i not in to_drop for i in range(n_feat)], dtype=bool
        )
        self.dropped_pairs_ = dropped_pairs
        self.n_dropped_ = len(to_drop)
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.keep_mask_))])[self.keep_mask_]
        return np.asarray(input_features)[self.keep_mask_]


# ─────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    max_missing_frac: float = MAX_MISSING_FRAC,
) -> Pipeline:
    """Build the 3-step feature preprocessing Pipeline.

    Steps
    -----
    1. ``MissingnessFilter`` — drop columns with > ``max_missing_frac`` NaN
    2. ``SimpleImputer``     — median imputation of remaining NaN
    3. ``RobustScaler``      — IQR-based scaling (robust to outliers)

    This pipeline must be fitted inside each CV fold on the training
    partition only to prevent data leakage.
    """
    return Pipeline([
        ("missingness_filter", MissingnessFilter(max_missing_frac=max_missing_frac)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])


def get_pipeline_feature_names(
    pipeline: Pipeline,
    input_names: list[str],
) -> list[str]:
    """Trace feature names through a fitted preprocessing pipeline.

    ``MissingnessFilter`` (which has ``keep_mask_``) is the only step that
    changes the feature set; ``SimpleImputer`` and ``RobustScaler`` are
    value-only transforms that leave column count unchanged.
    """
    names = np.asarray(input_names)
    for _step_name, step in pipeline.steps:
        if hasattr(step, "keep_mask_"):
            names = names[step.keep_mask_]
    return list(names)


# ─────────────────────────────────────────────────────────────
# PCA with explained-variance analysis
# ─────────────────────────────────────────────────────────────

def fit_pca(
    X_scaled: np.ndarray,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
) -> PCA | None:
    """Fit PCA retaining ``variance_threshold`` of explained variance.

    Returns None if ``X_scaled`` has fewer than PCA_MIN_FEATURES columns
    (e.g. for low-dimensional embeddings where PCA is unnecessary).
    """
    if X_scaled.shape[1] < PCA_MIN_FEATURES:
        logger.info(
            "PCA skipped: only %d features (< %d threshold)",
            X_scaled.shape[1], PCA_MIN_FEATURES,
        )
        return None

    pca = PCA(n_components=variance_threshold, svd_solver="full")
    pca.fit(X_scaled)
    logger.info(
        "PCA: %d → %d components (%.1f %% variance retained)",
        X_scaled.shape[1],
        pca.n_components_,
        100.0 * np.sum(pca.explained_variance_ratio_),
    )
    return pca


def plot_pca_explained_variance(
    X_scaled: np.ndarray,
    output_dir: Path,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
    tag: str = "",
) -> Path | None:
    """Fit full PCA and plot explained-variance scree + cumulative curve.

    Saves both the plot and the data used to create it.

    Parameters
    ----------
    X_scaled : np.ndarray
        Preprocessed (scaled) feature matrix.
    output_dir : Path
        Directory to save the plot and CSV.
    variance_threshold : float
        The 95 % threshold line drawn on the cumulative plot.
    tag : str
        Optional label appended to filenames (e.g. "motor_only").

    Returns
    -------
    Path to saved CSV, or None if PCA was skipped.
    """
    if X_scaled.shape[1] < PCA_MIN_FEATURES:
        logger.info("PCA explained-variance plot skipped (too few features)")
        return None

    pca_full = PCA(svd_solver="full")
    pca_full.fit(X_scaled)

    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_at_threshold = int(np.searchsorted(cumulative, variance_threshold) + 1)

    # Save data
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    csv_path = output_dir / f"pca_explained_variance{suffix}.csv"
    df_pca = pd.DataFrame({
        "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "explained_variance": pca_full.explained_variance_,
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative_ratio": cumulative,
    })
    df_pca.to_csv(csv_path, index=False)
    logger.info("PCA explained-variance data → %s", csv_path)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scree plot
    ax = axes[0]
    n_show = min(50, len(pca_full.explained_variance_ratio_))
    ax.bar(
        range(1, n_show + 1),
        pca_full.explained_variance_ratio_[:n_show],
        color="#3498DB", alpha=0.8, edgecolor="white",
    )
    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Explained Variance Ratio", fontsize=11)
    ax.set_title("Scree Plot (first 50 components)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    # Cumulative plot
    ax = axes[1]
    ax.plot(range(1, len(cumulative) + 1), cumulative, "o-", color="#E74C3C",
            markersize=2, lw=1.5)
    ax.axhline(variance_threshold, color="grey", linestyle="--", lw=1,
               label=f"{variance_threshold:.0%} threshold")
    ax.axvline(n_components_at_threshold, color="grey", linestyle=":", lw=1,
               label=f"{n_components_at_threshold} components")
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=11)
    ax.set_title("Cumulative Explained Variance", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"PCA Explained Variance{f' — {tag}' if tag else ''}", fontsize=13)
    fig.tight_layout()
    plot_path = output_dir / f"pca_explained_variance{suffix}.png"
    fig.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("PCA explained-variance plot → %s", plot_path)

    return csv_path


# ─────────────────────────────────────────────────────────────
# Feature selection report (informational, outside CV)
# ─────────────────────────────────────────────────────────────

def generate_feature_selection_report(
    X: pd.DataFrame,
    output_dir: Path,
    corr_threshold: float = CORR_THRESHOLD,
    near_zero_frac: float = NEAR_ZERO_FRAC,
    max_missing_frac: float = MAX_MISSING_FRAC,
) -> pd.DataFrame:
    """Fit the preprocessing pipeline on the full feature matrix and produce
    a CSV report listing every dropped feature and the reason.

    This is run once for reporting purposes; actual CV uses the Pipeline
    fitted independently on each training fold.
    """
    feature_names = np.array(X.columns.tolist())
    X_arr = X.values.astype(float)
    n_orig = X_arr.shape[1]
    rows = []

    # Step 1: Missingness
    mf = MissingnessFilter(max_missing_frac=max_missing_frac)
    mf.fit(X_arr)
    dropped_miss = feature_names[~mf.keep_mask_]
    for feat in dropped_miss:
        miss_frac = np.mean(np.isnan(X_arr[:, list(feature_names).index(feat)]))
        rows.append({"feature": feat, "reason": f"high_missingness (frac={miss_frac:.2f})", "correlated_with": ""})
    X_arr2 = mf.transform(X_arr)
    names2 = feature_names[mf.keep_mask_]

    # Step 2: Imputation (no features dropped)
    imp = SimpleImputer(strategy="median")
    X_arr3 = imp.fit_transform(X_arr2)

    # Step 3: Near-zero variance
    nzv = NearZeroVarianceFilter(frac=near_zero_frac)
    nzv.fit(X_arr3)
    dropped_nzv = names2[~nzv.keep_mask_]
    for feat in dropped_nzv:
        rows.append({"feature": feat, "reason": "near_zero_variance", "correlated_with": ""})
    X_arr4 = nzv.transform(X_arr3)
    names4 = names2[nzv.keep_mask_]

    # Step 4: Correlation filter
    cf = CorrelationFilter(threshold=corr_threshold)
    cf.fit(X_arr4)
    dropped_corr_mask = ~cf.keep_mask_
    kept_by_dropped = {}
    for kept_idx, dropped_idx in cf.dropped_pairs_:
        kept_by_dropped[dropped_idx] = kept_idx
    for j, feat in enumerate(names4):
        if dropped_corr_mask[j]:
            kept_feat = names4[kept_by_dropped.get(j, 0)]
            rows.append({
                "feature": feat,
                "reason": f"high_correlation (|r|>{corr_threshold})",
                "correlated_with": kept_feat,
            })

    report = pd.DataFrame(rows, columns=["feature", "reason", "correlated_with"])
    n_kept = n_orig - len(report)
    logger.info(
        "Feature selection (full dataset): %d features → %d kept "
        "(%d missingness, %d near-zero-var, %d correlated)",
        n_orig, n_kept, len(dropped_miss), len(dropped_nzv),
        int(dropped_corr_mask.sum()),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "feature_selection_report.csv"
    report.to_csv(out_path, index=False)
    logger.info("Feature selection report → %s", out_path)
    return report
