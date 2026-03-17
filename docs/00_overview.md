# ccnEvaluation — Pipeline Overview

## Purpose

This module implements a machine-learning prediction pipeline that compares
three feature representations for predicting Autism Spectrum Disorder (ASD)
diagnosis and ADOS-2 severity scores from pose-estimation data:

1. **Motor-only kinematics** — individual movement features (speed,
   acceleration, velocity, kinetic energy, distance, keypoint set changes)
   for both child and clinician.
2. **Full kinematics** — motor + social (interpersonal distance, approach,
   facingness, congruent motion, agitation).
3. **HumanLISBET embeddings** — frame-level deep-learning embeddings from the
   HumanLISBET encoder (*placeholder, see `10_embedding_notes.md`*).

## Scientific rationale

Only **trunk-height-normalised** features are used. Normalisation by the
participant's trunk height removes depth artefacts introduced by the
monocular 2-D projection (participants closer to the camera appear larger),
making cross-session comparisons scientifically valid.

## Prediction targets

| Target | Type | Column |
|--------|------|--------|
| ASD diagnosis (ASD vs TD) | Binary classification | `diagnosis` |
| ADOS-2 Calibrated Severity Score (CSS) | Regression | `ADOS_G_ADOS_2_TOTAL_score_de_severite` |
| Social Affect severity (SA) | Regression | `ADOS_2_ADOS_G_REVISED_SA_SEVERITY_SCORE` |
| Restricted & Repetitive Behaviours (RRB) | Regression | `ADOS_2_ADOS_G_REVISED_RRB_SEVERITY_SCORE_new` |

## Pipeline data flow

```
Step 1  Feature extraction
          ├── Motor-only kinematics (N subjects × M_motor features)
          ├── Full kinematics      (N subjects × M_full features)
          └── Embeddings           (placeholder)

Step 2  Preprocessing report (informational — feature selection stats)

Step 3  Repeated nested cross-validation (5 outer × 10 repeats)
          ├── Inner loop: 3-fold RandomizedSearchCV (n_iter=50)
          ├── Preprocessing fitted inside inner loop (no leakage)
          └── Metrics per fold: AUC-ROC / RMSE etc.

Step 4  Held-out evaluation
          ├── Retrain best model on full training set (N ≈ 99)
          └── Evaluate on pre-defined test set (N ≈ 20) + CIs

Step 5  Statistical comparison across representations
          └── Wilcoxon signed-rank + Bonferroni (3 pairwise tests)

Step 6  Plotting (every plot saves companion CSV)

Step 7  SHAP explainability (TODO — placeholder)
```

## Dataset split

The dataset uses a **pre-existing train/test split** stored in
`dataset/pose_ready/train/` and `dataset/pose_ready/test/`. The split was
created before this pipeline and is fixed to prevent information leakage.

- Training: ~99 subjects (used for CV and final retraining)
- Test: ~20 subjects (held-out, touched only in Step 4)

## Running the pipeline

```bash
# From humanLISBET-paper/ directory:
python -m ccnEvaluation.src.run_pipeline \
    --csv dataset/info/child_for_humanlisbet_paper_with_paths_020326.csv \
    --pose-records dataset/pose_records \
    --output-dir ccnEvaluation/results \
    --n-jobs 16 --use-gpu --skip-embedding

# Or via SLURM:
sbatch ccnEvaluation/scripts/run_pipeline.sh
```

## Module map

| File | Purpose |
|------|---------|
| `src/config.py` | All constants, paths, targets, feature prefixes |
| `src/loading.py` | Feature extraction (kinematic + embedding placeholder) |
| `src/preprocessing.py` | 5-step pipeline + PCA |
| `src/models.py` | LightGBM + Lasso definitions |
| `src/cross_validation.py` | 5×10 repeated nested CV |
| `src/held_out.py` | Final retrain + test evaluation + CIs |
| `src/comparison.py` | Wilcoxon signed-rank + effect sizes |
| `src/plotting.py` | All visualisation functions |
| `src/run_pipeline.py` | CLI entry point |
| `scripts/run_pipeline.sh` | SLURM submission script |
