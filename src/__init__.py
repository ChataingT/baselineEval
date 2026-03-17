"""
ccnEvaluation.src — Pose vs Embedding prediction pipeline.

Compares three feature representations for ASD classification and ADOS
severity prediction:
  1. Motor-only kinematics  (individual speed, acceleration, velocity, …)
  2. Full kinematics        (motor + interpersonal distance, facingness, …)
  3. HumanLISBET embeddings (learned ViT representations — placeholder)

Pipeline steps:
  - Feature extraction (11 Tukey-clipped statistics per normalised metric)
  - Preprocessing (missingness → impute → near-zero-var → corr-filter → scale)
  - PCA dimensionality reduction (95 % explained variance)
  - 5×10 repeated nested cross-validation (LightGBM + Lasso)
  - Held-out evaluation (retrain best on N=99, evaluate on N=20)
  - Statistical comparison across representations (Wilcoxon signed-rank)
  - Publication-quality figures (each saved with companion data CSV)

Usage:
    python -m ccnEvaluation.src.run_pipeline \\
        --csv      dataset/info/child_for_humanlisbet_paper_with_paths_020326.csv \\
        --pose-records dataset/pose_records \\
        --output-dir ccnEvaluation/results
"""
