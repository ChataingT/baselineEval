#!/bin/bash
#SBATCH --job-name=ccn_stats_sweep
#SBATCH --output=logs/sweep_stats_%A_%a.out
#SBATCH --error=logs/sweep_stats_%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=shared-gpu
#SBATCH --constraint="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9"
#SBATCH --array=0-0

# ── Environment ──────────────────────────────────────────────
module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0

source /home/shares/schaerm/schaer2/thibaut/humanlisbet/lisbet_venv/bin/activate

# ── Working directory ────────────────────────────────────────
cd /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper

# ── Paths ────────────────────────────────────────────────────
CSV=/srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper/dataset/info/child_for_humanlisbet_paper_with_paths_020326.csv
POSE_READY_TRAIN=/srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper/dataset/pose_ready/train
POSE_READY_TEST=/srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper/dataset/pose_ready/test
EMB_ROOT=/home/shares/schaerm/schaer2/thibaut/humanlisbet/output/full_train/embeddings/models/hlis-gr-w1200-o600-L8-H8-E128-FF512-auggeom-embedder/embeddings
OUTPUT_BASE=/srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper/ccnEvaluation/results_stats

# ── Sweep configurations ────────────────────────────────────
KIN_CONFIGS=("mean,std,skewness,kurtosis")
EMB_CONFIGS=("mean,std")
LABELS=("kin-moments_emb-basic")

STATS_KIN="${KIN_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
STATS_EMB="${EMB_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${OUTPUT_BASE}_${LABEL}_2"

echo "Array task $SLURM_ARRAY_TASK_ID: --stats-kin $STATS_KIN --stats-emb $STATS_EMB → $OUTPUT_DIR"

# ── Run ──────────────────────────────────────────────────────
python -m ccnEvaluation.src.run_pipeline \
    --csv              "$CSV" \
    --pose-ready-train "$POSE_READY_TRAIN" \
    --pose-ready-test  "$POSE_READY_TEST" \
    --output-dir       "$OUTPUT_DIR" \
    --embedding-dir-train "$EMB_ROOT/train" \
    --embedding-dir-test  "$EMB_ROOT/test" \
    --n-jobs         16 \
    --use-gpu \
    --log-level      INFO \
    --stats-kin      "$STATS_KIN" \
    --stats-emb      "$STATS_EMB"

echo "SLURM job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished with exit code $?"
