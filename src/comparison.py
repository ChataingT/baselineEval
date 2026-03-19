"""
Statistical comparison across feature representations.

For each target, compare the 50 outer-fold scores (from the 5×10 repeated CV)
between every pair of representations using a Wilcoxon signed-rank test.

Corrections
-----------
- **Bonferroni** within each target: 3 pairwise comparisons →
  adjusted α = 0.05 / 3 ≈ 0.0167.

Effect size
-----------
- **Rank-biserial** *r* = 1 − 2W / (n(n+1)/2), where *W* is the Wilcoxon
  statistic and *n* is the number of non-zero differences.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


def _pick_classification_metric(cv_results: dict[str, pd.DataFrame]) -> str:
    """Pick the best available metric for classification comparisons.

    Prefer AUC-ROC when it has any non-NaN values; otherwise fall back to
    macro F1 for multiclass or degenerate cases.
    """
    for df_cv in cv_results.values():
        if "auc_roc" in df_cv.columns and df_cv["auc_roc"].notna().any():
            return "auc_roc"
    for df_cv in cv_results.values():
        if "f1_macro" in df_cv.columns and df_cv["f1_macro"].notna().any():
            return "f1_macro"
    if any("auc_roc" in df_cv.columns for df_cv in cv_results.values()):
        return "auc_roc"
    return "f1_macro"


def _rank_biserial(stat: float, n: int) -> float:
    """Rank-biserial correlation from the Wilcoxon statistic."""
    if n < 1:
        return np.nan
    return 1.0 - (2.0 * stat) / (n * (n + 1) / 2)


def compare_representations(
    cv_results: dict[str, pd.DataFrame],
    task_type: str,
    target_name: str,
    output_dir: Path,
    best_model: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests between representations.

    Parameters
    ----------
    cv_results : dict
        ``{representation_name: cv_results_df}`` produced by
        ``run_repeated_nested_cv_*``.
    task_type : ``"classification"`` or ``"regression"``.
    target_name : str
        Target column name (used for logging and file naming).
    output_dir : Path
        Where to save the comparison CSV.
    best_model : str or None
        If given, restrict comparison to this model only (e.g. ``"lgbm"``).
        Otherwise, use the best model per representation.
    alpha : float
        Nominal significance level (default 0.05).

    Returns
    -------
    DataFrame with columns: rep_a, rep_b, metric, stat, p_value,
    p_corrected, effect_size_r, significant, n_pairs.
    """
    out = output_dir / "comparison" / target_name
    out.mkdir(parents=True, exist_ok=True)

    if task_type == "classification":
        metric = _pick_classification_metric(cv_results)
        higher_better = True
    else:
        metric = "rmse"
        higher_better = False

    # Determine the per-representation best model (or use the forced one)
    scores = {}
    for rep_name, df_cv in cv_results.items():
        if df_cv.empty or metric not in df_cv.columns:
            logger.info("  Skipping %s for %s: missing %s", rep_name, target_name, metric)
            continue
        if best_model is not None:
            df_sub = df_cv.loc[df_cv["model"] == best_model]
        else:
            # Pick model with best mean on the comparison metric
            agg = df_cv.groupby("model")[metric].mean().dropna()
            if agg.empty:
                logger.info("  Skipping %s for %s: no valid %s values", rep_name, target_name, metric)
                continue
            if higher_better:
                chosen = agg.idxmax()
            else:
                chosen = agg.idxmin()
            df_sub = df_cv.loc[df_cv["model"] == chosen]
        if df_sub.empty or metric not in df_sub.columns:
            logger.info("  Skipping %s for %s: no rows for %s", rep_name, target_name, metric)
            continue
        df_sub = df_sub.loc[df_sub[metric].notna()]
        if df_sub.empty:
            logger.info("  Skipping %s for %s: %s is all NaN", rep_name, target_name, metric)
            continue
        # Sort to align folds
        df_sub = df_sub.sort_values(["repeat", "fold"]).reset_index(drop=True)
        scores[rep_name] = df_sub[metric].values

    # All pairwise comparisons
    rep_names = sorted(scores.keys())
    pairs = list(itertools.combinations(rep_names, 2))
    n_comparisons = len(pairs)
    bonferroni_alpha = alpha / max(n_comparisons, 1)

    rows = []
    for rep_a, rep_b in pairs:
        a = scores[rep_a]
        b = scores[rep_b]

        n_pairs = min(len(a), len(b))
        a = a[:n_pairs]
        b = b[:n_pairs]

        diff = a - b
        non_zero = np.count_nonzero(diff)

        if non_zero < 1:
            rows.append({
                "rep_a": rep_a,
                "rep_b": rep_b,
                "metric": metric,
                "stat": np.nan,
                "p_value": 1.0,
                "p_corrected": 1.0,
                "effect_size_r": 0.0,
                "significant": False,
                "n_pairs": n_pairs,
                "n_nonzero": 0,
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "mean_diff": 0.0,
            })
            continue

        stat, p = wilcoxon(a, b, alternative="two-sided")
        p_corr = min(p * n_comparisons, 1.0)
        r = _rank_biserial(stat, non_zero)

        rows.append({
            "rep_a": rep_a,
            "rep_b": rep_b,
            "metric": metric,
            "stat": float(stat),
            "p_value": float(p),
            "p_corrected": float(p_corr),
            "effect_size_r": float(r),
            "significant": p_corr < alpha,
            "n_pairs": n_pairs,
            "n_nonzero": int(non_zero),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(diff)),
        })
        logger.info(
            "  %s vs %s (%s, %s): W=%.1f, p=%.4f (p_corr=%.4f), r=%.3f %s",
            rep_a, rep_b, target_name, metric,
            stat, p, p_corr, r,
            "***" if p_corr < alpha else "",
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "comparison_results.csv", index=False)
    logger.info("  Saved: %s", out / "comparison_results.csv")
    return df


def compare_all_targets(
    all_cv_results: dict[str, dict[str, pd.DataFrame]],
    targets_tasks: dict[str, str],
    output_dir: Path,
    best_model: str | None = None,
) -> pd.DataFrame:
    """Run comparisons for every target and concatenate results.

    Parameters
    ----------
    all_cv_results : dict
        ``{target_name: {representation_name: cv_results_df}}``.
    targets_tasks : dict
        ``{target_name: "classification" | "regression"}``.
    output_dir : Path
        Root output directory.
    best_model : str or None
        See :func:`compare_representations`.

    Returns
    -------
    Concatenated DataFrame of all comparison results.
    """
    frames = []
    for target_name, rep_results in all_cv_results.items():
        task_type = targets_tasks[target_name]
        df = compare_representations(
            cv_results=rep_results,
            task_type=task_type,
            target_name=target_name,
            output_dir=output_dir,
            best_model=best_model,
        )
        df["target"] = target_name
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_path = output_dir / "comparison" / "all_comparison_results.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(combined_path, index=False)
    logger.info("All comparisons → %s", combined_path)
    return df_all


def build_heatmap_tables(
    all_cv_results: dict[str, dict[str, pd.DataFrame]],
    targets_tasks: dict[str, str],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build per-target summary tables for heatmap plotting.

    Returns a dict with keys: "all", "classification", "regression".
    """
    rows_all = []
    rows_best = {"classification": [], "regression": []}

    for target_name, rep_results in all_cv_results.items():
        task_type = targets_tasks[target_name]
        if task_type == "classification":
            metric = _pick_classification_metric(rep_results)
            higher_better = True
        else:
            metric = "rmse"
            higher_better = False

        for rep_name, df_cv in rep_results.items():
            if df_cv.empty or metric not in df_cv.columns:
                continue

            agg = df_cv.groupby("model")[metric].agg(["mean", "std"]).reset_index()
            agg = agg.loc[agg["mean"].notna()]
            for _, row in agg.iterrows():
                rows_all.append({
                    "target": target_name,
                    "task_type": task_type,
                    "representation": rep_name,
                    "model": row["model"],
                    "metric": metric,
                    "metric_mean": float(row["mean"]),
                    "metric_std": float(row["std"]),
                })

            if agg.empty:
                continue

            if higher_better:
                best_row = agg.loc[agg["mean"].idxmax()]
            else:
                best_row = agg.loc[agg["mean"].idxmin()]

            rows_best[task_type].append({
                "target": target_name,
                "task_type": task_type,
                "representation": rep_name,
                "model": best_row["model"],
                "metric": metric,
                "metric_mean": float(best_row["mean"]),
                "metric_std": float(best_row["std"]),
            })

    out_dir = output_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.DataFrame(rows_all)
    df_all.to_csv(out_dir / "heatmap_source_all.csv", index=False)
    logger.info("Heatmap source (all models) → %s", out_dir / "heatmap_source_all.csv")

    results = {"all": df_all}
    for task_type, rows in rows_best.items():
        df_best = pd.DataFrame(rows)
        out_path = out_dir / f"heatmap_{task_type}_data.csv"
        df_best.to_csv(out_path, index=False)
        logger.info("Heatmap source (%s) → %s", task_type, out_path)
        results[task_type] = df_best

    return results
