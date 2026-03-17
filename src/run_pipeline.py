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
    RANDOM_STATE,
    REGRESSION_TARGETS,
    REPR_COMBINED,
    REPR_EMBEDDING,
    REPR_FULL_KINEMATIC,
    REPR_MOTOR_ONLY,
    TARGET_DIAGNOSIS,
    resolve_stat_selection,
)
from .loading import (
    filter_feature_columns,
    get_embedding_feature_columns,
    get_feature_columns,
    load_embedding_features,
    load_kinematic_features,
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
from .comparison import compare_all_targets
from .plotting import (
    plot_confusion_matrices,
    plot_cv_metric_boxplots,
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

    df_train = load_kinematic_features(
        csv_path=args.csv,
        pose_ready_dir=args.pose_ready_train,
        n_jobs=args.n_jobs,
        debug_n=args.debug_n,
        selected_stats=selected_stats_kin,
    )
    df_test = load_kinematic_features(
        csv_path=args.csv,
        pose_ready_dir=args.pose_ready_test,
        n_jobs=args.n_jobs,
        selected_stats=selected_stats_kin,
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
) -> pd.DataFrame:
    """Step 3: Repeated nested CV."""
    logger.info("── CV for %s (%s) ──", target_name, task_type)

    X, y, df_meta, feat_names = prepare_Xy(df_train, feature_cols, target_name)
    logger.info("  X: %s, y: %s (classes: %s)", X.shape, y.shape, np.unique(y))

    if task_type == "classification":
        return run_repeated_nested_cv_classification(
            X, y, df_meta, feat_names, output_dir, target_name, n_jobs=n_jobs,
        )
    else:
        return run_repeated_nested_cv_regression(
            X, y, df_meta, feat_names, output_dir, target_name, n_jobs=n_jobs,
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


def _step6_plotting(
    all_cv_results: dict[str, dict[str, pd.DataFrame]],
    all_pred_csvs: dict[str, dict[str, Path]],
    held_out_results: dict[str, list[dict]],
    df_comparison: pd.DataFrame,
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

    # Build target → task mapping
    targets_tasks = {}
    for t in CLASSIFICATION_TARGETS:
        targets_tasks[t] = "classification"
    for t in REGRESSION_TARGETS:
        targets_tasks[t] = "regression"

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
        else:
            df_rep_train = df_train
            df_rep_test  = df_test
            all_feature_cols = get_feature_columns(df_rep_train)
            feature_cols = filter_feature_columns(all_feature_cols, rep_name)

        logger.info("  Features for %s: %d columns", rep_name, len(feature_cols))

        # Step 2: Preprocessing report (once per representation)
        _step2_preprocessing_report(df_rep_train, feature_cols, rep_out)

        for target_name in ALL_TARGETS:
            task_type = targets_tasks[target_name]

            # Step 3: Cross-validation
            df_cv = _step3_cross_validation(
                df_rep_train, feature_cols, rep_out, target_name, task_type,
                n_jobs=args.n_jobs,
            )

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

    # ── Step 6: Plotting ─────────────────────────────────────
    _step6_plotting(
        all_cv_results, all_pred_csvs, held_out_results,
        df_comparison, targets_tasks, args.output_dir,
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
