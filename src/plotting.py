"""
Plotting functions for the ccnEvaluation pipeline.

Every public function saves **both** a ``.png`` image and a companion ``.csv``
containing the data used to produce the plot, so that any figure can be
reproduced or re-styled independently.

All figures use ``matplotlib`` with a consistent style:
- ``dpi=150``
- top/right spines removed
- model colours from :data:`config.PALETTE_MODELS`
- representation colours from :data:`config.PALETTE_REPRESENTATIONS`
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster jobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .config import DPI, PALETTE_MODELS, PALETTE_REPRESENTATIONS

logger = logging.getLogger(__name__)


def _clean_ax(ax):  # noqa: ANN001
    """Remove top/right spines."""
    ax.spines[["top", "right"]].set_visible(False)


# ─────────────────────────────────────────────────────────────
# ROC curves (classification)
# ─────────────────────────────────────────────────────────────

def plot_roc_curves(
    df_preds: pd.DataFrame,
    output_dir: Path,
    target_name: str,
    data_source: str = "",
) -> None:
    """Mean ROC curve ± 1 std per model from CV predictions.

    Parameters
    ----------
    df_preds : DataFrame
        Must contain columns: model, y_true, y_prob_pos, repeat, fold.
    """
    out = output_dir / "plots" / "classification" / target_name
    out.mkdir(parents=True, exist_ok=True)

    mean_fpr = np.linspace(0, 1, 200)
    csv_rows = []

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_id in sorted(df_preds["model"].unique()):
        sub = df_preds.loc[df_preds["model"] == model_id]
        tprs = []
        aucs = []
        for (rep, fold), grp in sub.groupby(["repeat", "fold"]):
            fpr, tpr, _ = roc_curve(grp["y_true"], grp["y_prob_pos"])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        color = PALETTE_MODELS.get(model_id, "#555555")
        ax.plot(
            mean_fpr, mean_tpr,
            color=color, lw=2,
            label=f"{model_id} (AUC={mean_auc:.3f}±{std_auc:.3f})",
        )
        ax.fill_between(
            mean_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            color=color, alpha=0.15,
        )

        for i, fpr_val in enumerate(mean_fpr):
            csv_rows.append({
                "model": model_id,
                "fpr": fpr_val,
                "mean_tpr": mean_tpr[i],
                "std_tpr": std_tpr[i],
                "mean_auc": mean_auc,
                "std_auc": std_auc,
            })

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"ROC — {target_name}" + (f"\n{data_source}" if data_source else ""))
    ax.legend(loc="lower right", fontsize=9)
    _clean_ax(ax)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(out / "roc_curves_data.csv", index=False)
    logger.info("  Saved: roc_curves.png + data CSV")


# ─────────────────────────────────────────────────────────────
# Confusion matrices (classification)
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    df_preds: pd.DataFrame,
    output_dir: Path,
    target_name: str,
    label_names: list[str] | None = None,
    data_source: str = "",
) -> None:
    """Aggregated confusion matrix per model across all CV folds."""
    out = output_dir / "plots" / "classification" / target_name
    out.mkdir(parents=True, exist_ok=True)

    if label_names is None:
        label_names = ["TD", "ASD"]

    model_ids = sorted(df_preds["model"].unique())
    ncols = min(len(model_ids), 3)
    nrows = (len(model_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    csv_rows = []

    for idx, model_id in enumerate(model_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        sub = df_preds.loc[df_preds["model"] == model_id]
        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1])

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set(
            xticks=[0, 1], xticklabels=label_names,
            yticks=[0, 1], yticklabels=label_names,
            xlabel="Predicted", ylabel="True",
            title=model_id,
        )

        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14,
                )
                csv_rows.append({
                    "model": model_id,
                    "true_label": label_names[i],
                    "pred_label": label_names[j],
                    "count": int(cm[i, j]),
                })

    # Hide unused axes
    for idx in range(len(model_ids), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Confusion Matrices — {target_name}" + (f"\n{data_source}" if data_source else ""),
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(out / "confusion_matrices_data.csv", index=False)
    logger.info("  Saved: confusion_matrices.png + data CSV")


# ─────────────────────────────────────────────────────────────
# CV metric box / violin plots
# ─────────────────────────────────────────────────────────────

def plot_cv_metric_boxplots(
    df_cv: pd.DataFrame,
    output_dir: Path,
    target_name: str,
    task_type: str,
    data_source: str = "",
) -> None:
    """Box plots of per-fold CV metrics, one panel per metric."""
    out = output_dir / "plots" / task_type / target_name
    out.mkdir(parents=True, exist_ok=True)

    if task_type == "classification":
        metrics = ["auc_roc", "balanced_acc", "f1_macro", "sensitivity", "specificity"]
    else:
        metrics = ["rmse", "mae", "r2", "spearman_r", "pearson_r"]

    model_ids = sorted(df_cv["model"].unique())
    ncols = min(len(metrics), 3)
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    csv_rows = []
    for idx, metric in enumerate(metrics):
        if metric not in df_cv.columns:
            continue
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        data = [df_cv.loc[df_cv["model"] == m, metric].dropna().values for m in model_ids]
        colors = [PALETTE_MODELS.get(m, "#999999") for m in model_ids]

        bplot = ax.boxplot(
            data, labels=model_ids, patch_artist=True,
            widths=0.5, showfliers=True,
        )
        for patch, c in zip(bplot["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        ax.set_title(metric)
        _clean_ax(ax)

        for m, vals in zip(model_ids, data):
            for v in vals:
                csv_rows.append({"metric": metric, "model": m, "value": float(v)})

    for idx in range(len(metrics), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"CV Metrics — {target_name}" + (f"\n{data_source}" if data_source else ""),
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "cv_metric_boxplots.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(out / "cv_metric_boxplots_data.csv", index=False)
    logger.info("  Saved: cv_metric_boxplots.png + data CSV")


# ─────────────────────────────────────────────────────────────
# Hyperparameter distributions
# ─────────────────────────────────────────────────────────────

def plot_hyperparameter_distributions(
    df_cv: pd.DataFrame,
    output_dir: Path,
    target_name: str,
    task_type: str,
    data_source: str = "",
) -> None:
    """Histogram of selected best hyperparameters across CV folds."""
    import ast

    out = output_dir / "plots" / task_type / target_name
    out.mkdir(parents=True, exist_ok=True)

    model_ids = sorted(df_cv["model"].unique())
    all_param_rows = []

    for model_id in model_ids:
        sub = df_cv.loc[df_cv["model"] == model_id, "best_params"]
        for param_str in sub:
            try:
                params = ast.literal_eval(param_str)
            except (ValueError, SyntaxError):
                continue
            for k, v in params.items():
                all_param_rows.append({"model": model_id, "param": k, "value": v})

    if not all_param_rows:
        return

    df_params = pd.DataFrame(all_param_rows)
    df_params.to_csv(out / "hyperparameter_distributions_data.csv", index=False)

    for model_id in model_ids:
        sub = df_params.loc[df_params["model"] == model_id]
        params = sorted(sub["param"].unique())
        if not params:
            continue

        ncols = min(len(params), 4)
        nrows = (len(params) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

        for idx, param in enumerate(params):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            vals = sub.loc[sub["param"] == param, "value"]

            try:
                vals_numeric = pd.to_numeric(vals, errors="raise")
                ax.hist(vals_numeric, bins=15, color=PALETTE_MODELS.get(model_id, "#999"), alpha=0.7)
            except (ValueError, TypeError):
                vc = vals.value_counts()
                ax.bar(range(len(vc)), vc.values, tick_label=[str(x) for x in vc.index],
                       color=PALETTE_MODELS.get(model_id, "#999"), alpha=0.7)

            ax.set_title(param.replace("model__", ""), fontsize=9)
            _clean_ax(ax)

        for idx in range(len(params), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            f"Hyperparams — {model_id} — {target_name}" + (f"\n{data_source}" if data_source else ""),
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out / f"hyperparams_{model_id}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    logger.info("  Saved: hyperparameter distribution plots + data CSV")


# ─────────────────────────────────────────────────────────────
# Predicted vs actual (regression)
# ─────────────────────────────────────────────────────────────

def plot_pred_vs_actual(
    df_preds: pd.DataFrame,
    output_dir: Path,
    target_name: str,
    data_source: str = "",
) -> None:
    """Scatter of predicted vs actual for regression, one plot per model."""
    out = output_dir / "plots" / "regression" / target_name
    out.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    for model_id in sorted(df_preds["model"].unique()):
        sub = df_preds.loc[df_preds["model"] == model_id]
        y_true = sub["y_true"].values
        y_pred = sub["y_pred"].values

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        sp_r, _ = spearmanr(y_true, y_pred)
        pe_r, _ = pearsonr(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        color = PALETTE_MODELS.get(model_id, "#555555")
        ax.scatter(y_true, y_pred, alpha=0.3, s=20, color=color)

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", lw=1, alpha=0.5)

        ax.set(
            xlabel="Actual", ylabel="Predicted",
            title=f"{model_id} — {target_name}\nRMSE={rmse:.3f}  ρ={sp_r:.3f}  r={pe_r:.3f}"
                  + (f"\n{data_source}" if data_source else ""),
        )
        _clean_ax(ax)
        fig.tight_layout()
        fig.savefig(out / f"pred_vs_actual_{model_id}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        for yt, yp in zip(y_true, y_pred):
            csv_rows.append({
                "model": model_id, "y_true": float(yt), "y_pred": float(yp),
            })

    pd.DataFrame(csv_rows).to_csv(out / "pred_vs_actual_data.csv", index=False)
    logger.info("  Saved: pred_vs_actual plots + data CSV")


# ─────────────────────────────────────────────────────────────
# Residual analysis (regression)
# ─────────────────────────────────────────────────────────────

def plot_residuals(
    df_preds: pd.DataFrame,
    output_dir: Path,
    target_name: str,
    data_source: str = "",
) -> None:
    """Histogram + scatter of residuals per model."""
    out = output_dir / "plots" / "regression" / target_name
    out.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    for model_id in sorted(df_preds["model"].unique()):
        sub = df_preds.loc[df_preds["model"] == model_id]
        residuals = sub["residual"].values if "residual" in sub.columns else sub["y_pred"].values - sub["y_true"].values
        y_pred = sub["y_pred"].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        color = PALETTE_MODELS.get(model_id, "#555555")

        ax1.hist(residuals, bins=20, color=color, alpha=0.7, edgecolor="white")
        ax1.axvline(0, color="black", ls="--", lw=1)
        ax1.set(xlabel="Residual", ylabel="Count", title="Distribution")
        _clean_ax(ax1)

        ax2.scatter(y_pred, residuals, alpha=0.3, s=20, color=color)
        ax2.axhline(0, color="black", ls="--", lw=1)
        ax2.set(xlabel="Predicted", ylabel="Residual", title="Residuals vs Predicted")
        _clean_ax(ax2)

        fig.suptitle(
            f"Residuals — {model_id} — {target_name}" + (f"\n{data_source}" if data_source else ""),
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out / f"residuals_{model_id}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        for yp, res in zip(y_pred, residuals):
            csv_rows.append({"model": model_id, "y_pred": float(yp), "residual": float(res)})

    pd.DataFrame(csv_rows).to_csv(out / "residuals_data.csv", index=False)
    logger.info("  Saved: residual plots + data CSV")


# ─────────────────────────────────────────────────────────────
# Held-out summary bar chart
# ─────────────────────────────────────────────────────────────

def plot_held_out_summary(
    results: list[dict],
    output_dir: Path,
    task_type: str,
    data_source: str = "",
) -> None:
    """Grouped bar chart comparing held-out metrics across targets and representations."""
    out = output_dir / "plots" / "held_out"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        return

    df.to_csv(out / f"held_out_summary_{task_type}_data.csv", index=False)

    if task_type == "classification":
        metrics = ["auc_roc", "balanced_acc", "f1_macro"]
    else:
        metrics = ["rmse", "r2", "spearman_r"]

    metrics = [m for m in metrics if m in df.columns]

    # Determine unique targets and representations for grouped bars
    targets = df["target"].unique().tolist()
    reps = df["representation"].unique().tolist() if "representation" in df.columns else [None]
    n_reps = len(reps)
    bar_width = 0.8 / n_reps

    ncols = len(metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(max(5, 3 * len(targets)) * ncols, 4), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[0][idx]

        for r_idx, rep in enumerate(reps):
            if rep is not None:
                sub = df[df["representation"] == rep]
            else:
                sub = df
            vals, x_pos, ci_lo_vals, ci_hi_vals = [], [], [], []
            for t_idx, target in enumerate(targets):
                row = sub[sub["target"] == target]
                if row.empty:
                    continue
                v = float(row[metric].iloc[0])
                vals.append(v)
                x_pos.append(t_idx + r_idx * bar_width - (n_reps - 1) * bar_width / 2)
                ci_lo_col, ci_hi_col = f"{metric}_ci_lo", f"{metric}_ci_hi"
                if ci_lo_col in row.columns and ci_hi_col in row.columns:
                    ci_lo_vals.append(v - float(row[ci_lo_col].iloc[0]))
                    ci_hi_vals.append(float(row[ci_hi_col].iloc[0]) - v)

            color = PALETTE_REPRESENTATIONS.get(rep, "#3498DB") if rep else "#3498DB"
            label = rep if rep else metric
            ax.bar(x_pos, vals, width=bar_width, color=color, alpha=0.8, label=label)
            if ci_lo_vals:
                ax.errorbar(x_pos, vals, yerr=[ci_lo_vals, ci_hi_vals],
                            fmt="none", ecolor="black", capsize=3)

        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric)
        if n_reps > 1:
            ax.legend(fontsize=7)
        _clean_ax(ax)

    fig.suptitle(
        f"Held-out {task_type} summary" + (f"\n{data_source}" if data_source else ""),
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / f"held_out_summary_{task_type}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: held_out_summary_%s.png + data CSV", task_type)


# ─────────────────────────────────────────────────────────────
# Representation comparison bar charts
# ─────────────────────────────────────────────────────────────

def plot_representation_comparison(
    df_comparison: pd.DataFrame,
    all_cv_results: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    target_name: str,
    task_type: str,
    data_source: str = "",
) -> None:
    """Grouped bar chart: mean CV metric per representation with significance markers."""
    out = output_dir / "plots" / "comparison" / target_name
    out.mkdir(parents=True, exist_ok=True)

    if task_type == "classification":
        metric = "auc_roc"
        higher_better = True
    else:
        metric = "rmse"
        higher_better = False

    # Gather per-representation mean and std
    rep_stats = {}
    for rep_name, df_cv in all_cv_results.get(target_name, {}).items():
        # Use best model per rep
        agg = df_cv.groupby("model")[metric].mean()
        best_model = agg.idxmax() if higher_better else agg.idxmin()
        vals = df_cv.loc[df_cv["model"] == best_model, metric]
        rep_stats[rep_name] = {"mean": float(vals.mean()), "std": float(vals.std())}

    if not rep_stats:
        return

    csv_rows = [{"representation": k, "mean": v["mean"], "std": v["std"]} for k, v in rep_stats.items()]
    pd.DataFrame(csv_rows).to_csv(out / "representation_comparison_data.csv", index=False)

    rep_names = sorted(rep_stats.keys())
    means = [rep_stats[r]["mean"] for r in rep_names]
    stds = [rep_stats[r]["std"] for r in rep_names]
    colors = [PALETTE_REPRESENTATIONS.get(r, "#999999") for r in rep_names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(rep_names)), means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(rep_names)))
    ax.set_xticklabels(rep_names, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(
        f"Representation comparison — {target_name}"
        + (f"\n{data_source}" if data_source else ""),
    )
    _clean_ax(ax)

    # Significance markers from comparison table
    if df_comparison is not None and not df_comparison.empty:
        comp_sub = df_comparison.loc[df_comparison.get("target", pd.Series()) == target_name]
        if comp_sub.empty and "target" not in df_comparison.columns:
            comp_sub = df_comparison
        y_max = max(means) + max(stds) * 1.5
        for _, row in comp_sub.iterrows():
            if row.get("significant", False):
                a_idx = rep_names.index(row["rep_a"]) if row["rep_a"] in rep_names else None
                b_idx = rep_names.index(row["rep_b"]) if row["rep_b"] in rep_names else None
                if a_idx is not None and b_idx is not None:
                    y_max += max(stds) * 0.3
                    ax.plot([a_idx, a_idx, b_idx, b_idx], [y_max - max(stds) * 0.1, y_max, y_max, y_max - max(stds) * 0.1],
                            color="black", lw=1)
                    ax.text((a_idx + b_idx) / 2, y_max, "*", ha="center", va="bottom", fontsize=14)

    fig.tight_layout()
    fig.savefig(out / "representation_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: representation_comparison.png + data CSV")


# ─────────────────────────────────────────────────────────────
# SHAP feature importance
# ─────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 20,
    data_source: str = "",
) -> None:
    """Bar chart (mean |SHAP|) and beeswarm plot for the top-N features.

    Parameters
    ----------
    shap_values : array of shape (n_subjects, n_features).
    X : preprocessed feature matrix of shape (n_subjects, n_features).
        Used for colour-coding the beeswarm by feature value.
    feature_names : list of feature names (length n_features).
    output_dir : directory where PNG files and companion CSVs are saved.
    top_n : number of top features to display (default 20).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = shap_values.shape[1]
    top_n = min(top_n, n_features)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    # ── Bar chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(range(top_n), top_vals[::-1], color="#3498DB", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        f"Top {top_n} features by mean |SHAP|"
        + (f"\n{data_source}" if data_source else ""),
        fontsize=12,
    )
    _clean_ax(ax)
    fig.tight_layout()
    bar_path = output_dir / f"shap_bar_top{top_n}.png"
    fig.savefig(bar_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"feature": top_names, "mean_abs_shap": top_vals}).to_csv(
        output_dir / f"shap_bar_top{top_n}_data.csv", index=False
    )

    # ── Beeswarm ──────────────────────────────────────────────
    sv_top = shap_values[:, top_idx]   # (n_subjects, top_n)
    X_top = X[:, top_idx]              # (n_subjects, top_n)
    X_std = (X_top - X_top.mean(axis=0)) / (X_top.std(axis=0) + 1e-9)

    cmap = plt.cm.RdBu_r
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    for rank in range(top_n):
        fi = top_n - 1 - rank  # most important feature → highest y position
        y_pos = np.full(len(sv_top), rank)
        y_jit = y_pos + rng.uniform(-0.3, 0.3, len(y_pos))
        sc = ax.scatter(
            sv_top[:, fi], y_jit,
            c=X_std[:, fi], cmap=cmap,
            vmin=-2, vmax=2, s=12, alpha=0.7, linewidths=0,
        )

    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("SHAP value", fontsize=11)
    ax.set_title(
        f"SHAP beeswarm — top {top_n} features"
        + (f"\n{data_source}" if data_source else ""),
        fontsize=12,
    )
    plt.colorbar(sc, ax=ax, label="Feature value (standardised)", shrink=0.6)
    _clean_ax(ax)
    fig.tight_layout()
    bee_path = output_dir / f"shap_beeswarm_top{top_n}.png"
    fig.savefig(bee_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("  Saved: %s and %s", bar_path.name, bee_path.name)
