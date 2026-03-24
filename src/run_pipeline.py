#!/usr/bin/env python3
"""
Main entry point for the ccnEvaluation prediction pipeline.

Usage
-----
    python -m ccnEvaluation.src.run_pipeline \\
        --csv dataset/info/child_for_humanlisbet_paper_with_paths_020326.csv \\
        --pose-ready-train dataset/pose_ready/train \\
        --pose-ready-test dataset/pose_ready/test \\
        --output-dir ccnEvaluation/results \\
        [--n-jobs 16] \\
        [--use-gpu] \\
        [--log-level INFO] \\
        [--skip-embedding] \\
        [--debug-n 5]

Steps
-----
  1. Feature extraction  (kinematic: motor-only & full; embedding: loaded from pre-computed dirs)
  2. PCA explained-variance diagnostic (exploratory only — not used in prediction)
  3. Repeated nested CV  (5×10 × RandomizedSearch inner loop)
     Pipeline per fold: MissingnessFilter → SimpleImputer → RobustScaler → Model
  4. Held-out evaluation  (retrain best on N≈102, evaluate on N≈20)
     SHAP feature importances computed here for motor_only and full_kinematic.
  5. Statistical comparison across representations (Wilcoxon + Bonferroni)
  6. Plotting (ROC, CM, boxplots, hyperparams, pred-vs-actual, residuals,
     held-out summary, comparison charts, SHAP bar + beeswarm)
  7. SHAP — computed inside held-out evaluation (Step 4)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "ccnEvaluation.src"

from .config import (
    ALL_REPRESENTATIONS,
    ALL_TARGETS,
    CLASSIFICATION_TARGETS,
    DEFAULT_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_READY_TEST,
    DEFAULT_POSE_READY_TRAIN,
    MIN_SAMPLES_PER_TARGET,
    RANDOM_STATE,
    REGRESSION_TARGETS,
    REPR_COMBINED,
    REPR_COMBINED_TEMPORAL,
    REPR_EMBEDDING,
    REPR_EMBEDDING_TEMPORAL,
    REPR_FULL_KINEMATIC,
    REPR_FULL_KINEMATIC_TEMPORAL,
    REPR_MOTOR_ONLY,
    REPR_MOTOR_ONLY_TEMPORAL,
    TARGET_DIAGNOSIS,
    TEMPORAL_REPRESENTATIONS,
    resolve_stat_selection,
)
from .loading import (
    filter_feature_columns,
    get_embedding_feature_columns,
    get_embedding_temporal_feature_columns,
    get_feature_columns,
    get_kinematic_temporal_feature_columns,
    load_embedding_features,
    load_embedding_features_temporal,
    load_kinematic_features,
    load_kinematic_features_temporal,
    merge_kinematic_embedding,
    prepare_Xy,
)
from .preprocessing import (
    generate_feature_selection_report,
    plot_pca_explained_variance,
)
from .cross_validation import (
    run_repeated_nested_cv_classification,
    run_repeated_nested_cv_regression,
)
from .held_out import (
    evaluate_held_out_classification,
    evaluate_held_out_regression,
)
from .comparison import build_heatmap_tables, compare_all_targets
from .plotting import (
    plot_confusion_matrices,
    plot_cv_metric_boxplots,
    plot_comparison_heatmap,
    plot_held_out_summary,
    plot_hyperparameter_distributions,
    plot_pred_vs_actual,
    plot_representation_comparison,
    plot_residuals,
    plot_roc_curves,
)

logger = logging.getLogger("ccnEvaluation")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ccnEvaluation prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="Path to subject CSV with metadata.")
    p.add_argument("--pose-ready-train", type=Path, default=DEFAULT_POSE_READY_TRAIN,
                   help="Train split directory (pose_ready).")
    p.add_argument("--pose-ready-test", type=Path, default=DEFAULT_POSE_READY_TEST,
                   help="Test split directory (pose_ready).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Root output directory for results.")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="Number of parallel workers.")
    p.add_argument("--use-gpu", action="store_true",
                   help="Enable GPU for LightGBM (requires CUDA build).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity.")
    p.add_argument("--skip-embedding", action="store_true",
                   help="Skip embedding representation even if dirs are provided.")
    p.add_argument("--embedding-dir-train", type=Path, default=None,
                   help="Train embedding directory (sub-folders with features_lisbet_embedding.csv).")
    p.add_argument("--embedding-dir-test", type=Path, default=None,
                   help="Test embedding directory (sub-folders with features_lisbet_embedding.csv).")
    p.add_argument("--debug-n", type=int, default=None,
                   help="Limit to first N subjects for quick debugging.")
    p.add_argument("--stats-kin", type=str, default=None,
                   help="Stats for kinematic features: comma-separated names or "
                        "preset ('basic', 'moments'). Default: all 11 statistics.")
    p.add_argument("--stats-emb", type=str, default=None,
                   help="Stats for embedding features: comma-separated names or "
                        "preset ('basic', 'moments'). Default: all 11 statistics.")
    p.add_argument("--save-raw-kinematic-features", action="store_true",
                   help="Save per-frame kinematic/social metrics before aggregation.")
    p.add_argument("--raw-kinematic-dir", type=Path, default=None,
                   help="Root directory to save raw per-frame kinematic features. "
                        "Defaults to <output_dir>/raw_kinematic_frames/{train|test}.")
    p.add_argument("--temporal-pooling", action="store_true",
                   help="Add temporal max-pooling representations alongside standard ones. "
                        "Each recording is split into N_TEMPORAL_SEGMENTS non-overlapping "
                        "windows; features are max-pooled within each window, then "
                        "aggregated across windows with mean / max / std.")
    p.add_argument("--skip-lgbm", action="store_true",
                   help="Skip LightGBM and run only Lasso / logistic regression. "
                        "Useful for fast exploratory runs or CPU-only environments.")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────

def _step1_load_features(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Step 1: Compute kinematic + social features and load embeddings."""
    logger.info("═══ Step 1: Feature extraction ═══")

    selected_stats_kin = resolve_stat_selection(args.stats_kin)
    selected_stats_emb = resolve_stat_selection(args.stats_emb)
    if selected_stats_kin is not None:
        logger.info("Selected statistics (kinematic): %s", selected_stats_kin)
    if selected_stats_emb is not None:
        logger.info("Selected statistics (embedding): %s", selected_stats_emb)

    raw_kin_base = None
    if args.save_raw_kinematic_features:
        raw_kin_base = args.raw_kinematic_dir or (args.output_dir / "raw_kinematic_frames")
        logger.info("Saving raw kinematic frames to %s", raw_kin_base)

    df_train = load_kinematic_features(
        csv_path=args.csv,
        pose_ready_dir=args.pose_ready_train,
        n_jobs=args.n_jobs,
        debug_n=args.debug_n,
        selected_stats=selected_stats_kin,
        raw_output_dir=(raw_kin_base / "train") if raw_kin_base is not None else None,
    )
    df_test = load_kinematic_features(
        csv_path=args.csv,
        pose_ready_dir=args.pose_ready_test,
        n_jobs=args.n_jobs,
        selected_stats=selected_stats_kin,
        raw_output_dir=(raw_kin_base / "test") if raw_kin_base is not None else None,
    )

    logger.info("  Train (kinematic): %d subjects × %d total columns", df_train.shape[0], df_train.shape[1])
    logger.info("  Test  (kinematic): %d subjects × %d total columns", df_test.shape[0], df_test.shape[1])

    result: dict = {"train": df_train, "test": df_test}

    embedding_dirs_provided = (
        args.embedding_dir_train is not None
        and args.embedding_dir_test is not None
    )
    if embedding_dirs_provided and not args.skip_embedding:
        logger.info("═══ Step 1b: Embedding feature extraction ═══")
        df_emb_train = load_embedding_features(
            csv_path=args.csv,
            embedding_dir=args.embedding_dir_train,
            debug_n=args.debug_n,
            selected_stats=selected_stats_emb,
        )
        df_emb_test = load_embedding_features(
            csv_path=args.csv,
            embedding_dir=args.embedding_dir_test,
            selected_stats=selected_stats_emb,
        )
        logger.info("  Train (embedding): %d subjects × %d total columns", df_emb_train.shape[0], df_emb_train.shape[1])
        logger.info("  Test  (embedding): %d subjects × %d total columns", df_emb_test.shape[0], df_emb_test.shape[1])
        result["emb_train"] = df_emb_train
        result["emb_test"] = df_emb_test

    return result


def _step1_load_features_temporal(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Step 1 (temporal): Load kinematic + embedding features with temporal max-pooling."""
    logger.info("═══ Step 1 (temporal): Temporal max-pooling feature extraction ═══")

    df_train = load_kinematic_features_temporal(
        csv_path=args.csv,
        pose_ready_dir=args.pose_ready_train,
        n_jobs=args.n_jobs,
        debug_n=args.debug_n,
    )
    df_test = load_kinematic_features_temporal(
        csv_path=args.csv,
        pose_ready_dir=args.pose_ready_test,
        n_jobs=args.n_jobs,
    )
    logger.info(
        "  Train (temporal kinematic): %d subjects × %d total columns",
        df_train.shape[0], df_train.shape[1],
    )
    logger.info(
        "  Test  (temporal kinematic): %d subjects × %d total columns",
        df_test.shape[0], df_test.shape[1],
    )

    result: dict[str, pd.DataFrame] = {
        "train_temporal": df_train,
        "test_temporal":  df_test,
    }

    embedding_dirs_provided = (
        args.embedding_dir_train is not None
        and args.embedding_dir_test is not None
    )
    if embedding_dirs_provided and not args.skip_embedding:
        logger.info("═══ Step 1b (temporal): Temporal embedding feature extraction ═══")
        df_emb_train = load_embedding_features_temporal(
            csv_path=args.csv,
            embedding_dir=args.embedding_dir_train,
            debug_n=args.debug_n,
        )
        df_emb_test = load_embedding_features_temporal(
            csv_path=args.csv,
            embedding_dir=args.embedding_dir_test,
        )
        logger.info(
            "  Train (temporal embedding): %d subjects × %d total columns",
            df_emb_train.shape[0], df_emb_train.shape[1],
        )
        logger.info(
            "  Test  (temporal embedding): %d subjects × %d total columns",
            df_emb_test.shape[0], df_emb_test.shape[1],
        )
        result["emb_train_temporal"] = df_emb_train
        result["emb_test_temporal"]  = df_emb_test

    return result


def _step2_preprocessing_report(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    """Step 2: Generate feature selection report and PCA variance plot."""
    logger.info("═══ Step 2: Preprocessing report ═══")
    X_df = df_train[feature_cols]
    generate_feature_selection_report(X_df, output_dir)

    # PCA needs finite values: impute NaN then scale (matching CV pipeline)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    X_arr = SimpleImputer(strategy="median").fit_transform(X_df.values)
    X_arr = RobustScaler().fit_transform(X_arr)
    plot_pca_explained_variance(X_arr, output_dir)


def _step3_cross_validation(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    target_name: str,
    task_type: str,
    n_jobs: int,
    skip_lgbm: bool = False,
) -> pd.DataFrame | None:
    """Step 3: Repeated nested CV."""
    logger.info("── CV for %s (%s) ──", target_name, task_type)

    X, y, df_meta, feat_names = prepare_Xy(df_train, feature_cols, target_name)
    logger.info("  X: %s, y: %s (classes: %s)", X.shape, y.shape, np.unique(y))

    if y.shape[0] < MIN_SAMPLES_PER_TARGET:
        logger.warning(
            "Skipping target '%s': %d samples after filtering (< %d minimum).",
            target_name, y.shape[0], MIN_SAMPLES_PER_TARGET,
        )
        return None

    if task_type == "classification" and len(np.unique(y)) < 2:
        logger.warning(
            "Skipping target '%s': only one class present after filtering.",
            target_name,
        )
        return None

    if task_type == "classification":
        return run_repeated_nested_cv_classification(
            X, y, df_meta, feat_names, output_dir, target_name,
            n_jobs=n_jobs, skip_lgbm=skip_lgbm,
        )
    else:
        return run_repeated_nested_cv_regression(
            X, y, df_meta, feat_names, output_dir, target_name,
            n_jobs=n_jobs, skip_lgbm=skip_lgbm,
        )


def _step4_held_out(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    df_cv: pd.DataFrame,
    output_dir: Path,
    target_name: str,
    task_type: str,
    representation: str = "unknown",
) -> dict:
    """Step 4: Held-out evaluation."""
    logger.info("── Held-out for %s (%s | %s) ──", target_name, task_type, representation)

    X_train, y_train, _, feat_names = prepare_Xy(df_train, feature_cols, target_name)
    X_test, y_test, df_meta_test, _ = prepare_Xy(df_test, feature_cols, target_name)

    if task_type == "classification":
        return evaluate_held_out_classification(
            X_train, y_train, X_test, y_test,
            df_meta_test, df_cv, feat_names, output_dir, target_name,
            representation=representation,
        )
    else:
        return evaluate_held_out_regression(
            X_train, y_train, X_test, y_test,
            df_meta_test, df_cv, feat_names, output_dir, target_name,
            representation=representation,
        )


def _step5_comparison(
    all_cv_results: dict[str, dict[str, pd.DataFrame]],
    targets_tasks: dict[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    """Step 5: Statistical comparison across representations."""
    logger.info("═══ Step 5: Statistical comparison ═══")
    return compare_all_targets(all_cv_results, targets_tasks, output_dir)


def _filter_targets(
    df_train: pd.DataFrame,
    targets_tasks: dict[str, str],
) -> dict[str, str]:
    """Filter out targets with too few non-null samples and log why."""
    kept: dict[str, str] = {}
    for target_name, task_type in targets_tasks.items():
        if target_name not in df_train.columns:
            logger.warning("Skipping target '%s': column not found in training data.", target_name)
            continue

        non_null = int(df_train[target_name].notna().sum())
        if non_null < MIN_SAMPLES_PER_TARGET:
            logger.warning(
                "Skipping target '%s': %d non-null samples (< %d minimum).",
                target_name, non_null, MIN_SAMPLES_PER_TARGET,
            )
            continue

        kept[target_name] = task_type

    if not kept:
        raise ValueError("No targets meet the minimum sample requirement.")

    return kept


def _step6_plotting(
    all_cv_results: dict[str, dict[str, pd.DataFrame]],
    all_pred_csvs: dict[str, dict[str, Path]],
    held_out_results: dict[str, list[dict]],
    df_comparison: pd.DataFrame,
    heatmap_tables: dict[str, pd.DataFrame],
    targets_tasks: dict[str, str],
    output_dir: Path,
) -> None:
    """Step 6: Generate all plots."""
    logger.info("═══ Step 6: Plotting ═══")

    for target_name, rep_results in all_cv_results.items():
        task_type = targets_tasks[target_name]

        for rep_name, df_cv in rep_results.items():
            rep_out = output_dir / rep_name
            cv_source = f"Train (CV) — {rep_name}"

            # CV metric boxplots
            plot_cv_metric_boxplots(df_cv, rep_out, target_name, task_type,
                                    data_source=cv_source)

            # Hyperparameter distributions
            plot_hyperparameter_distributions(df_cv, rep_out, target_name, task_type,
                                              data_source=cv_source)

            # Per-subject predictions for detailed plots
            pred_dir = rep_out / task_type / target_name
            pred_csv = pred_dir / "predictions_per_subject.csv"
            if pred_csv.exists():
                df_preds = pd.read_csv(pred_csv)

                if task_type == "classification":
                    plot_roc_curves(df_preds, rep_out, target_name,
                                    data_source=cv_source)
                    plot_confusion_matrices(df_preds, rep_out, target_name,
                                            data_source=cv_source)
                else:
                    plot_pred_vs_actual(df_preds, rep_out, target_name,
                                        data_source=cv_source)
                    plot_residuals(df_preds, rep_out, target_name,
                                   data_source=cv_source)

        # Representation comparison (across reps for this target)
        plot_representation_comparison(
            df_comparison, all_cv_results, output_dir, target_name, task_type,
            data_source="Train (CV, best model per rep)",
        )

    # Held-out summary
    for task_type in ("classification", "regression"):
        results = []
        for target_name, ht in held_out_results.items():
            if targets_tasks.get(target_name) == task_type:
                results.extend(ht if isinstance(ht, list) else [ht])
        if results:
            plot_held_out_summary(results, output_dir, task_type,
                                  data_source="Test (held-out)")

    # Comparison heatmaps (best model per representation)
    for task_type in ("classification", "regression"):
        df_heatmap = heatmap_tables.get(task_type)
        if df_heatmap is not None and not df_heatmap.empty:
            plot_comparison_heatmap(
                df_heatmap,
                output_dir,
                task_type,
                data_source="Train (CV, best model per rep)",
            )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("ccnEvaluation prediction pipeline started")
    logger.info("  CSV:           %s", args.csv)
    logger.info("  Pose (train):  %s", args.pose_ready_train)
    logger.info("  Pose (test):   %s", args.pose_ready_test)
    logger.info("  Output:        %s", args.output_dir)
    t0 = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load features ────────────────────────────────
    data = _step1_load_features(args)
    df_train = data["train"]
    df_test = data["test"]

    # Determine which representations to evaluate
    representations = [REPR_MOTOR_ONLY, REPR_FULL_KINEMATIC]
    embedding_loaded = "emb_train" in data and "emb_test" in data
    if embedding_loaded:
        representations.append(REPR_EMBEDDING)
        representations.append(REPR_COMBINED)
    elif args.skip_embedding:
        logger.info("Skipping embedding representation (--skip-embedding)")
    else:
        logger.info(
            "Skipping embedding representation "
            "(--embedding-dir-train / --embedding-dir-test not provided)."
        )

    # ── Step 1 (temporal): Load temporal max-pooling features ─
    temporal_data: dict[str, pd.DataFrame] = {}
    if args.temporal_pooling:
        temporal_data = _step1_load_features_temporal(args)
        representations += [REPR_MOTOR_ONLY_TEMPORAL, REPR_FULL_KINEMATIC_TEMPORAL]
        if "emb_train_temporal" in temporal_data:
            representations += [REPR_EMBEDDING_TEMPORAL, REPR_COMBINED_TEMPORAL]
        logger.info(
            "Temporal pooling enabled — added %d temporal representations.",
            len([r for r in representations if r in TEMPORAL_REPRESENTATIONS]),
        )

    # Build target → task mapping
    targets_tasks = {}
    for t in CLASSIFICATION_TARGETS:
        targets_tasks[t] = "classification"
    for t in REGRESSION_TARGETS:
        targets_tasks[t] = "regression"

    targets_tasks = _filter_targets(df_train, targets_tasks)

    # ── Steps 2–4: Per representation × target ───────────────
    all_cv_results: dict[str, dict[str, pd.DataFrame]] = {}  # {target: {rep: df_cv}}
    all_pred_csvs: dict[str, dict[str, Path]] = {}
    held_out_results: dict[str, list[dict]] = {}

    for rep_name in representations:
        logger.info("═══ Representation: %s ═══", rep_name)
        rep_out = args.output_dir / rep_name

        # Select the right DataFrames and feature columns for this representation.
        if rep_name == REPR_EMBEDDING:
            df_rep_train = data["emb_train"]
            df_rep_test  = data["emb_test"]
            feature_cols = get_embedding_feature_columns(df_rep_train)
        elif rep_name == REPR_COMBINED:
            df_rep_train = merge_kinematic_embedding(df_train, data["emb_train"])
            df_rep_test  = merge_kinematic_embedding(df_test, data["emb_test"])
            kin_cols = filter_feature_columns(
                get_feature_columns(df_rep_train), REPR_FULL_KINEMATIC,
            )
            emb_cols = get_embedding_feature_columns(df_rep_train)
            feature_cols = kin_cols + emb_cols
        # ── Temporal representations ──────────────────────────
        elif rep_name == REPR_EMBEDDING_TEMPORAL:
            df_rep_train = temporal_data["emb_train_temporal"]
            df_rep_test  = temporal_data["emb_test_temporal"]
            feature_cols = get_embedding_temporal_feature_columns(df_rep_train)
        elif rep_name == REPR_COMBINED_TEMPORAL:
            df_rep_train = merge_kinematic_embedding(
                temporal_data["train_temporal"], temporal_data["emb_train_temporal"]
            )
            df_rep_test  = merge_kinematic_embedding(
                temporal_data["test_temporal"], temporal_data["emb_test_temporal"]
            )
            kin_cols = filter_feature_columns(
                get_kinematic_temporal_feature_columns(df_rep_train),
                REPR_FULL_KINEMATIC_TEMPORAL,
            )
            emb_cols = get_embedding_temporal_feature_columns(df_rep_train)
            feature_cols = kin_cols + emb_cols
        elif rep_name in (REPR_MOTOR_ONLY_TEMPORAL, REPR_FULL_KINEMATIC_TEMPORAL):
            df_rep_train = temporal_data["train_temporal"]
            df_rep_test  = temporal_data["test_temporal"]
            all_feature_cols = get_kinematic_temporal_feature_columns(df_rep_train)
            feature_cols = filter_feature_columns(all_feature_cols, rep_name)
        else:
            df_rep_train = df_train
            df_rep_test  = df_test
            all_feature_cols = get_feature_columns(df_rep_train)
            feature_cols = filter_feature_columns(all_feature_cols, rep_name)

        logger.info("  Features for %s: %d columns", rep_name, len(feature_cols))

        # Step 2: Preprocessing report (once per representation)
        _step2_preprocessing_report(df_rep_train, feature_cols, rep_out)

        for target_name, task_type in targets_tasks.items():

            # Step 3: Cross-validation
            df_cv = _step3_cross_validation(
                df_rep_train, feature_cols, rep_out, target_name, task_type,
                n_jobs=args.n_jobs, skip_lgbm=args.skip_lgbm,
            )

            if df_cv is None:
                continue

            all_cv_results.setdefault(target_name, {})[rep_name] = df_cv

            # Step 4: Held-out evaluation
            ho_result = _step4_held_out(
                df_rep_train, df_rep_test, feature_cols, df_cv,
                rep_out, target_name, task_type,
                representation=rep_name,
            )
            held_out_results.setdefault(target_name, []).append(ho_result)

    # ── Step 5: Statistical comparison ───────────────────────
    df_comparison = _step5_comparison(all_cv_results, targets_tasks, args.output_dir)

    # ── Step 5b: Heatmap summary tables ──────────────────────
    heatmap_tables = build_heatmap_tables(all_cv_results, targets_tasks, args.output_dir)

    # ── Step 6: Plotting ─────────────────────────────────────
    _step6_plotting(
        all_cv_results, all_pred_csvs, held_out_results,
        df_comparison, heatmap_tables, targets_tasks, args.output_dir,
    )

    # ── Step 7: SHAP ─────────────────────────────────────────
    # SHAP feature importances are computed inside evaluate_held_out_classification
    # and evaluate_held_out_regression (Step 4) for motor_only and full_kinematic.
    # Results are saved under <output_dir>/held_out/<task>/<target>/shap/.
    logger.info("═══ Step 7: SHAP — computed inside held-out evaluation ═══")

    elapsed = time.time() - t0
    logger.info("Pipeline completed in %.1f s (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
