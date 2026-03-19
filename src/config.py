"""
Central configuration for the ccnEvaluation prediction pipeline.

All constants, paths, target definitions, feature set definitions, and
cross-validation parameters are defined here so that every other module
imports from a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths (relative to the humanLISBET-paper/ working directory)
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # humanLISBET-paper/

DEFAULT_CSV = PROJECT_ROOT / "dataset" / "info" / "child_for_humanlisbet_paper_with_paths_020326.csv"
DEFAULT_POSE_READY_TRAIN = PROJECT_ROOT / "dataset" / "pose_ready" / "train"
DEFAULT_POSE_READY_TEST = PROJECT_ROOT / "dataset" / "pose_ready" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ccnEvaluation" / "results"


# ─────────────────────────────────────────────────────────────
# Prediction targets
# ─────────────────────────────────────────────────────────────

# Primary outcomes (binary classification)
TARGET_DIAGNOSIS = "diagnosis"
TARGET_ADOS_RRB_LEVEL = "ADOS_2_ADOS_G_revised_RRB_level_of_symptoms"
TARGET_ADOS_SA_LEVEL = "ADOS_2_ADOS_G_REVISED_SA_LEVEL_OF_SYMPTOMS"

# Secondary outcomes (regression)
TARGET_CSS = "ADOS_G_ADOS_2_TOTAL_score_de_severite"       # Calibrated Severity Score
TARGET_SA = "ADOS_2_ADOS_G_REVISED_SA_SEVERITY_SCORE"      # Social Affect
TARGET_RRB = "ADOS_2_ADOS_G_REVISED_RRB_SEVERITY_SCORE_new"  # Restricted & Repetitive Behaviours

VLDII_TARGETS = [
    "VLDII_AdSS",
    "VLDII_MotorSS",
    "VLDII_gmsVS",
    "VLDII_fmsVS",
    "VLDII_SocSS",
    "VLDII_intVS",
    "VLDII_plaVS",
    "VLDII_copVS",
    "VLDII_DaiSS",
    "VLDII_perVS",
    "VLDII_comVS",
    "VLDII_domVS",
    "VLDII_ComSS",
    "VLDII_expVS",
    "VLDII_recVS",
]

MSEL_TARGETS = [
    "MSEL_TOTAL_DQ",
    "MSEL_FM_DQ",
    "MSEL_VR_DQ",
    "MSEL_LR_DQ",
    "MSEL_LE_DQ",
    "MSEL_NV_DQ",
    "MSEL_V_DQ",
    "MSEL_GM_DQ",
]

CLASSIFICATION_TARGETS = [TARGET_DIAGNOSIS, TARGET_ADOS_RRB_LEVEL, TARGET_ADOS_SA_LEVEL]
REGRESSION_TARGETS = [TARGET_CSS, TARGET_SA, TARGET_RRB] + VLDII_TARGETS + MSEL_TARGETS
ALL_TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS

MIN_SAMPLES_PER_TARGET = 90

# Clinical / meta columns needed from the CSV (never used as features)
META_COLS = [
    "uuid", "code", "sujet_id", "diagnosis", "gender", "Ados_2_Age",
    "Ados_2_Module", "ADOS_2_TOTAL",
    TARGET_ADOS_RRB_LEVEL, TARGET_ADOS_SA_LEVEL,
    TARGET_CSS, TARGET_SA, TARGET_RRB,
] + VLDII_TARGETS + MSEL_TARGETS


# ─────────────────────────────────────────────────────────────
# Feature representations
# ─────────────────────────────────────────────────────────────

REPR_MOTOR_ONLY = "motor_only"
REPR_FULL_KINEMATIC = "full_kinematic"
REPR_EMBEDDING = "embedding"
REPR_COMBINED = "full_kinematic_embedding"
ALL_REPRESENTATIONS = [REPR_MOTOR_ONLY, REPR_FULL_KINEMATIC, REPR_EMBEDDING, REPR_COMBINED]

# ── Motor (individual) metric prefixes ───────────────────────
# These prefixes identify individual-level kinematics for child and clinician.
MOTOR_PREFIXES = (
    "child_speed_",
    "child_acceleration_",
    "child_velocity_",
    "child_kinetic_energy",
    "child_total_distance_",
    "child_kp_set_changed",
    "clinician_speed_",
    "clinician_acceleration_",
    "clinician_velocity_",
    "clinician_kinetic_energy",
    "clinician_total_distance_",
    "clinician_kp_set_changed",
)

# ── Social (interpersonal) metric prefixes ───────────────────
# These prefixes identify dyadic / interpersonal metrics.
SOCIAL_PREFIXES = (
    "interpersonal_distance_",
    "interpersonal_approach",
    "facingness",
    "congruent_motion",
    "agitation_global_ke",
)

# ── Excluded suffixes ────────────────────────────────────────
# For metrics that have both centroid and trunk variants (speed, total_distance,
# interpersonal_distance), keep only the centroid variant. The centroid uses
# all visible keypoints and is more robust; trunk-height normalisation already
# accounts for depth.
EXCLUDE_TRUNK_COLUMNS = True   # if True, drop columns with '_trunk' suffix


def is_motor_feature(col: str) -> bool:
    """Return True if *col* is a motor (individual) feature column."""
    base = col.split("__")[0]  # strip __norm__stat suffix
    return any(base.startswith(p) or base == p.rstrip("_") for p in MOTOR_PREFIXES)


def is_social_feature(col: str) -> bool:
    """Return True if *col* is a social (interpersonal) feature column."""
    base = col.split("__")[0]
    return any(base.startswith(p) or base == p.rstrip("_") for p in SOCIAL_PREFIXES)


# ─────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────

# Only trunk-height-normalised features are used (scientifically motivated:
# normalisation removes depth artefacts from the 2-D projection).
FEATURE_VARIANT = "norm"   # only __norm__ columns

# Minimum valid (non-NaN) frames to compute summary statistics
MIN_VALID_FRAMES = 50

# 11 summary statistics computed per metric
FLOAT_STAT_TYPES = [
    "mean", "std", "q10", "q25", "median", "q75", "q90",
    "iqr", "cv", "skewness", "kurtosis",
]
BOOL_STAT_TYPES = ["mean"]

# Named presets for --stats CLI argument
STAT_PRESETS: dict[str, list[str]] = {
    "basic":    ["mean", "std"],
    "moments":  ["mean", "std", "skewness", "kurtosis"],
}


def resolve_stat_selection(raw: str | None) -> list[str] | None:
    """Resolve a ``--stats`` CLI value into a validated list of stat names.

    Accepts ``None`` (all stats), a preset name (``'basic'``,
    ``'moments'``), or a comma-separated string of stat names
    (``'mean,std,kurtosis'``).

    Returns ``None`` when all 11 stats should be kept.
    Raises ``ValueError`` for unknown stat names or preset names.
    """
    if raw is None:
        return None

    raw = raw.strip()
    if raw in STAT_PRESETS:
        return list(STAT_PRESETS[raw])

    names = [s.strip() for s in raw.split(",") if s.strip()]
    invalid = [n for n in names if n not in FLOAT_STAT_TYPES]
    if invalid:
        raise ValueError(
            f"Unknown stat names: {invalid}. "
            f"Valid: {FLOAT_STAT_TYPES}. Presets: {list(STAT_PRESETS)}"
        )
    if not names:
        return None
    return names


# Columns to skip in the frame-level CSVs
SKIP_COLS = {"segment_id"}

# Tukey fence multiplier for IQR-based outlier clipping
TUKEY_K = 4.0


# ─────────────────────────────────────────────────────────────
# Metric computation (pose → frame-level features)
# ─────────────────────────────────────────────────────────────

# Individual roles in the dyad (used as column prefixes)
INDIVIDUAL_CHILD = "child"
INDIVIDUAL_CLINICIAN = "clinician"
DYADIC_INDIVIDUALS = [INDIVIDUAL_CHILD, INDIVIDUAL_CLINICIAN]

# Minimum number of shared visible keypoints to compute valid frame displacement
MIN_INTERSECTION_KP: int = 3

# COCO trunk keypoints used for trunk height / facingness computation
TRUNK_KPS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

# Rolling window for congruent motion (frames)
DEFAULT_CONGRUENT_WINDOW: int = 60

# Rolling median window for trunk-height smoothing (frames)
TRUNK_SMOOTH_WINDOW: int = 25


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────

MAX_MISSING_FRAC = 0.30    # drop features with > 30 % NaN
NEAR_ZERO_FRAC = 0.95      # drop features where > 95 % of values are identical
CORR_THRESHOLD = 0.95       # |Pearson r| above which one feature is dropped


# ─────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────

PCA_VARIANCE_THRESHOLD = 0.95   # retain 95 % of explained variance
PCA_MIN_FEATURES = 64           # skip PCA if n_features below this threshold


# ─────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────

N_OUTER_FOLDS = 5
N_REPEATS = 10
N_INNER_FOLDS = 3
N_ITER = 50               # RandomizedSearchCV iterations per inner search
RANDOM_STATE = 42
REG_STRAT_BINS = 5         # number of bins for stratifying continuous targets


# ─────────────────────────────────────────────────────────────
# Held-out evaluation
# ─────────────────────────────────────────────────────────────

BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_CI = 0.95


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

DPI = 150

PALETTE_MODELS = {
    "lgbm":  "#F39C12",
    "lasso": "#3498DB",
}

PALETTE_REPRESENTATIONS = {
    REPR_MOTOR_ONLY:     "#27AE60",
    REPR_FULL_KINEMATIC: "#E74C3C",
    REPR_EMBEDDING:      "#9B59B6",
    REPR_COMBINED:       "#E67E22",
}
