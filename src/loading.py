"""
Data loading for kinematic features and embedding placeholder.

Provides three loaders:
  load_kinematic_features  — load raw pose data (xarray ``tracking.nc``),
                             compute all kinematic and social metrics,
                             normalise, then extract 11 summary statistics.
  load_embedding_features  — placeholder (TODO) for HumanLISBET ViT embeddings.
  load_all_representations — convenience wrapper returning all three.

All metric computation is self-contained — adapted from
``poseToRecord/metrics.py`` with no imports from that module.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats as scipy_stats
from tqdm import tqdm

from .config import (
    BOOL_STAT_TYPES,
    CLASSIFICATION_TARGETS,
    DEFAULT_CONGRUENT_WINDOW,
    DYADIC_INDIVIDUALS,
    EXCLUDE_TRUNK_COLUMNS,
    FEATURE_VARIANT,
    FLOAT_STAT_TYPES,
    INDIVIDUAL_CHILD,
    INDIVIDUAL_CLINICIAN,
    META_COLS,
    MIN_INTERSECTION_KP,
    MIN_VALID_FRAMES,
    REPR_EMBEDDING,
    REPR_FULL_KINEMATIC,
    REPR_MOTOR_ONLY,
    TRUNK_KPS,
    TRUNK_SMOOTH_WINDOW,
    TUKEY_K,
    is_motor_feature,
    is_social_feature,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Per-array summary statistics (Tukey-clipped where appropriate)
# ─────────────────────────────────────────────────────────────

def _stats_float(
    arr: np.ndarray,
    selected: list[str] | None = None,
    track_clipping: bool = False,
) -> dict[str, float]:
    """Compute summary statistics for a numeric metric column.

    All 11 statistics are always computed internally (some depend on
    each other via Tukey clipping).  Only those in *selected* are
    returned.  When *selected* is ``None``, all 11 are returned.

    Statistics on *original* (unclipped) data:
        mean, q10, q25, median, q75, q90, iqr

    Statistics on *Tukey-fence-clipped* data (k=3×IQR):
        std, cv, skewness, kurtosis

    When *track_clipping* is True, two extra keys are added:
    ``_clip_n`` (count) and ``_clip_pct`` (percentage of clipped values).

    Returns NaN for all statistics if fewer than MIN_VALID_FRAMES valid frames.
    """
    stat_keys = selected if selected is not None else FLOAT_STAT_TYPES
    valid = arr[~np.isnan(arr)]
    if len(valid) < MIN_VALID_FRAMES:
        out = {s: np.nan for s in stat_keys}
        if track_clipping:
            out["_clip_n"] = 0.0
            out["_clip_pct"] = 0.0
        return out

    q10, q25, median, q75, q90 = np.percentile(valid, [10, 25, 50, 75, 90])
    mean = float(np.mean(valid))
    iqr = float(q75 - q25)

    # Tukey fence clipping (k = TUKEY_K)
    if iqr > 0:
        lo = q25 - TUKEY_K * iqr
        hi = q75 + TUKEY_K * iqr
    else:
        lo, hi = q25, q75
    valid_wins = np.clip(valid, lo, hi)

    n_clipped = int(np.sum(valid != valid_wins))
    pct_clipped = n_clipped / len(valid) * 100
    if pct_clipped > 0:
        logger.debug(
            "Tukey clip: %.2f%% of %d values clipped (k=%.0f)",
            pct_clipped, len(valid), TUKEY_K,
        )

    std = float(np.std(valid_wins, ddof=1))
    mean_wins = float(np.mean(valid_wins))
    cv = std / mean_wins if abs(mean_wins) > 0.01 * std else np.nan

    with np.errstate(all="ignore"):
        skewness = float(scipy_stats.skew(valid_wins, bias=False))
        kurtosis = float(scipy_stats.kurtosis(valid_wins, bias=False))

    all_stats = {
        "mean": mean,
        "std": std,
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
        "iqr": iqr,
        "cv": cv,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
    out = {k: all_stats[k] for k in stat_keys}
    if track_clipping:
        out["_clip_n"] = float(n_clipped)
        out["_clip_pct"] = pct_clipped
    return out


def _stats_bool(arr: np.ndarray) -> dict[str, float]:
    """Compute mean (proportion True) for a boolean metric column."""
    valid = arr[~np.isnan(arr.astype(float))]
    if len(valid) < MIN_VALID_FRAMES:
        return {"mean": np.nan}
    return {"mean": float(np.mean(valid.astype(float)))}


# ═══════════════════════════════════════════════════════════════
# Frame-level metric computation  (adapted from poseToRecord/metrics.py)
# ═══════════════════════════════════════════════════════════════

def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply a NaN-aware rolling median to a 1-D array."""
    result = np.full_like(arr, np.nan)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        vals = arr[lo:hi]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            result[i] = float(np.median(valid))
    return result


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division; NaN where denominator is NaN or zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (denominator == 0) | np.isnan(denominator),
            np.nan,
            numerator / denominator,
        )
    return result


def _compute_trunk_height(pos: np.ndarray, kp_names: list[str]) -> np.ndarray:
    """Compute trunk height per frame, smoothed with rolling median.

    trunk_height = AVG(d(left_shoulder, left_hip), d(right_shoulder, right_hip))
    Falls back to whichever side pair is visible if only one is.
    """
    n_frames = pos.shape[0]
    trunk_h = np.full(n_frames, np.nan)

    def _idx(name: str) -> int | None:
        return kp_names.index(name) if name in kp_names else None

    ls_i = _idx("left_shoulder")
    rs_i = _idx("right_shoulder")
    lh_i = _idx("left_hip")
    rh_i = _idx("right_hip")

    visible = np.abs(pos).sum(axis=-1) > 0  # (T, K)

    for t in range(n_frames):
        distances = []
        if ls_i is not None and lh_i is not None:
            if visible[t, ls_i] and visible[t, lh_i]:
                distances.append(float(np.linalg.norm(pos[t, ls_i] - pos[t, lh_i])))
        if rs_i is not None and rh_i is not None:
            if visible[t, rs_i] and visible[t, rh_i]:
                distances.append(float(np.linalg.norm(pos[t, rs_i] - pos[t, rh_i])))
        if distances:
            trunk_h[t] = float(np.mean(distances))

    return _rolling_median(trunk_h, TRUNK_SMOOTH_WINDOW)


def _centroid_speed_velocity(
    pos: np.ndarray, visible: np.ndarray, subset_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centroid speed/velocity using intersection of visible keypoints.

    Returns speed, vx, vy, kp_set_changed — all shape (T,).
    """
    n_frames = pos.shape[0]
    speed = np.full(n_frames, np.nan)
    vx = np.full(n_frames, np.nan)
    vy = np.full(n_frames, np.nan)
    kp_set_changed = np.zeros(n_frames, dtype=bool)

    for t in range(1, n_frames):
        shared = visible[t] & visible[t - 1]
        n_shared = int(shared.sum())
        if n_shared < MIN_INTERSECTION_KP:
            continue

        c_curr = pos[t, shared, :].mean(axis=0)
        c_prev = pos[t - 1, shared, :].mean(axis=0)
        disp = c_curr - c_prev
        speed[t] = float(np.linalg.norm(disp))
        vx[t] = float(disp[0])
        vy[t] = float(disp[1])

        if not np.array_equal(visible[t], visible[t - 1]):
            kp_set_changed[t] = True

    return speed, vx, vy, kp_set_changed


def _acceleration(speed: np.ndarray) -> np.ndarray:
    """Absolute frame-to-frame change in speed."""
    acc = np.full_like(speed, np.nan)
    for t in range(1, len(speed)):
        if not np.isnan(speed[t]) and not np.isnan(speed[t - 1]):
            acc[t] = abs(speed[t] - speed[t - 1])
    return acc


def _kinetic_energy(pos: np.ndarray) -> np.ndarray:
    """Per-frame kinetic energy: sum of squared displacements over shared kps."""
    n_frames = pos.shape[0]
    ke = np.full(n_frames, np.nan)
    visible = np.abs(pos).sum(axis=-1) > 0

    for t in range(1, n_frames):
        shared = visible[t] & visible[t - 1]
        if not shared.any():
            continue
        disp = pos[t, shared, :] - pos[t - 1, shared, :]
        ke[t] = float((disp ** 2).sum())
    return ke


def _total_distance_over_duration(speed: np.ndarray, fps: float) -> float:
    """Total path length divided by segment duration in seconds."""
    n = len(speed)
    if n == 0 or fps <= 0:
        return float(np.nan)
    if not (~np.isnan(speed)).any():
        return float(np.nan)
    return float(np.nansum(speed) / (n / fps))


def _individual_kinematics(
    pos: np.ndarray, kp_names: list[str], prefix: str,
) -> pd.DataFrame:
    """Compute speed, velocity, acceleration, KE for one individual."""
    n_frames, n_kp, _ = pos.shape
    visible = np.abs(pos).sum(axis=-1) > 0  # (T, K)

    # Centroid speed / velocity (all keypoints)
    speed_c, vx_c, vy_c, kp_set_changed = _centroid_speed_velocity(
        pos, visible, "all"
    )
    acc_c = _acceleration(speed_c)

    # Trunk centroid speed / velocity
    trunk_kp_idx = [kp_names.index(k) for k in TRUNK_KPS if k in kp_names]
    if trunk_kp_idx:
        pos_trunk = pos[:, trunk_kp_idx, :]
        vis_trunk = visible[:, trunk_kp_idx]
        speed_t, vx_t, vy_t, _ = _centroid_speed_velocity(
            pos_trunk, vis_trunk, "trunk"
        )
        acc_t = _acceleration(speed_t)
    else:
        speed_t = vx_t = vy_t = acc_t = np.full(n_frames, np.nan)

    # Kinetic energy
    ke = _kinetic_energy(pos)

    # Per-keypoint speed
    per_kp_speed: dict[str, np.ndarray] = {}
    for k, kp in enumerate(kp_names):
        spd_kp = np.full(n_frames, np.nan)
        for t in range(1, n_frames):
            if visible[t, k] and visible[t - 1, k]:
                spd_kp[t] = float(np.linalg.norm(pos[t, k] - pos[t - 1, k]))
        per_kp_speed[kp] = spd_kp

    # Assemble DataFrame
    df = pd.DataFrame(index=np.arange(n_frames))
    df[f"{prefix}_speed_centroid"] = speed_c
    df[f"{prefix}_speed_trunk"] = speed_t
    df[f"{prefix}_velocity_centroid_x"] = vx_c
    df[f"{prefix}_velocity_centroid_y"] = vy_c
    df[f"{prefix}_velocity_trunk_x"] = vx_t
    df[f"{prefix}_velocity_trunk_y"] = vy_t
    df[f"{prefix}_acceleration_centroid"] = acc_c
    df[f"{prefix}_acceleration_trunk"] = acc_t
    df[f"{prefix}_kinetic_energy"] = ke
    df[f"{prefix}_kp_set_changed"] = kp_set_changed
    for kp, spd in per_kp_speed.items():
        df[f"{prefix}_speed_kp_{kp}"] = spd

    return df


def _interpersonal_distances(
    pos_a: np.ndarray, pos_b: np.ndarray, kp_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Centroid-based and trunk-based interpersonal distance per frame."""
    n_frames = pos_a.shape[0]
    vis_a = np.abs(pos_a).sum(axis=-1) > 0
    vis_b = np.abs(pos_b).sum(axis=-1) > 0

    dist_c = np.full(n_frames, np.nan)
    dist_t = np.full(n_frames, np.nan)

    trunk_idx = [kp_names.index(k) for k in TRUNK_KPS if k in kp_names]

    for t in range(n_frames):
        if vis_a[t].sum() >= 3 and vis_b[t].sum() >= 3:
            ca = pos_a[t, vis_a[t], :].mean(axis=0)
            cb = pos_b[t, vis_b[t], :].mean(axis=0)
            dist_c[t] = float(np.linalg.norm(ca - cb))

        if trunk_idx:
            vis_t_a = vis_a[t][trunk_idx]
            vis_t_b = vis_b[t][trunk_idx]
            if vis_t_a.sum() >= 2 and vis_t_b.sum() >= 2:
                ta = pos_a[t][trunk_idx][vis_t_a].mean(axis=0)
                tb = pos_b[t][trunk_idx][vis_t_b].mean(axis=0)
                dist_t[t] = float(np.linalg.norm(ta - tb))

    return dist_c, dist_t


def _head_facing_direction(
    pos_t: np.ndarray, vis_t: np.ndarray,
    nose_i: int, le_i: int, re_i: int, ley_i: int, rey_i: int,
) -> np.ndarray | None:
    """Return 2-D head facing direction for one person at one frame.

    Strategy (in priority order):
    1. ``nose − mid(left_ear, right_ear)``  — best lateral resolution
    2. ``nose − visible_ear``               — single-ear fallback
    3. ``nose − mid(left_eye, right_eye)``  — eye fallback (less lateral range)

    Returns None when the nose or all reference keypoints are missing.
    """
    if not vis_t[nose_i]:
        return None
    nose = pos_t[nose_i]

    # Try ears first
    ear_ok_l = vis_t[le_i]
    ear_ok_r = vis_t[re_i]
    if ear_ok_l and ear_ok_r:
        ref = (pos_t[le_i] + pos_t[re_i]) / 2.0
    elif ear_ok_l:
        ref = pos_t[le_i]
    elif ear_ok_r:
        ref = pos_t[re_i]
    else:
        # Fall back to eyes
        eye_ok_l = vis_t[ley_i]
        eye_ok_r = vis_t[rey_i]
        if eye_ok_l and eye_ok_r:
            ref = (pos_t[ley_i] + pos_t[rey_i]) / 2.0
        elif eye_ok_l:
            ref = pos_t[ley_i]
        elif eye_ok_r:
            ref = pos_t[rey_i]
        else:
            return None

    vec = nose - ref
    return vec


def _facingness(
    pos_a: np.ndarray, pos_b: np.ndarray, kp_names: list[str],
) -> np.ndarray:
    """Head-orientation facingness per frame.

    For each person the head facing direction is estimated from
    ``nose − mid(ears)`` (with single-ear and eye fallbacks).  The metric
    measures whether each person's head is oriented toward the other by
    computing::

        cos_A = cos(heading_A, centroid_B − centroid_A)
        cos_B = cos(heading_B, centroid_A − centroid_B)
        facingness = (cos_A + cos_B) / 2

    Range [-1, +1]:
      * +1 → both facing each other
      * −1 → both facing away
      *  0 → perpendicular / mixed
    """
    n_frames = pos_a.shape[0]
    result = np.full(n_frames, np.nan)

    def _idx(name: str) -> int | None:
        return kp_names.index(name) if name in kp_names else None

    nose_i = _idx("nose")
    le_i = _idx("left_ear")
    re_i = _idx("right_ear")
    ley_i = _idx("left_eye")
    rey_i = _idx("right_eye")

    if nose_i is None:
        return result
    # Need at least one pair of reference keypoints to ever succeed
    if all(i is None for i in [le_i, re_i, ley_i, rey_i]):
        return result
    # Replace None indices with -1 (will always fail the visibility check)
    le_i = le_i if le_i is not None else -1
    re_i = re_i if re_i is not None else -1
    ley_i = ley_i if ley_i is not None else -1
    rey_i = rey_i if rey_i is not None else -1

    vis_a = np.abs(pos_a).sum(axis=-1) > 0  # (T, K)
    vis_b = np.abs(pos_b).sum(axis=-1) > 0

    for t in range(n_frames):
        # Head facing directions
        h_a = _head_facing_direction(pos_a[t], vis_a[t], nose_i, le_i, re_i, ley_i, rey_i)
        h_b = _head_facing_direction(pos_b[t], vis_b[t], nose_i, le_i, re_i, ley_i, rey_i)
        if h_a is None or h_b is None:
            continue

        norm_ha = np.linalg.norm(h_a)
        norm_hb = np.linalg.norm(h_b)
        if norm_ha < 1e-6 or norm_hb < 1e-6:
            continue

        # Centroids from all visible keypoints
        vis_kps_a = pos_a[t, vis_a[t]]
        vis_kps_b = pos_b[t, vis_b[t]]
        if len(vis_kps_a) == 0 or len(vis_kps_b) == 0:
            continue

        cent_a = vis_kps_a.mean(axis=0)
        cent_b = vis_kps_b.mean(axis=0)
        toward_b = cent_b - cent_a
        toward_a = cent_a - cent_b

        norm_tb = np.linalg.norm(toward_b)
        norm_ta = np.linalg.norm(toward_a)
        if norm_tb < 1e-6 or norm_ta < 1e-6:
            continue

        cos_a = float(np.dot(h_a, toward_b) / (norm_ha * norm_tb))
        cos_b = float(np.dot(h_b, toward_a) / (norm_hb * norm_ta))
        result[t] = (cos_a + cos_b) / 2.0

    return result


def _congruent_motion(
    speed_a: np.ndarray, speed_b: np.ndarray, window: int,
) -> np.ndarray:
    """Rolling Pearson correlation of two speed time series."""
    n_frames = len(speed_a)
    result = np.full(n_frames, np.nan)

    for t in range(window - 1, n_frames):
        wa = speed_a[t - window + 1 : t + 1]
        wb = speed_b[t - window + 1 : t + 1]

        both_valid = ~(np.isnan(wa) | np.isnan(wb))
        if both_valid.mean() < 0.70:
            continue

        wa_v = wa[both_valid]
        wb_v = wb[both_valid]
        if len(wa_v) < 3 or wa_v.std() < 1e-9 or wb_v.std() < 1e-9:
            continue

        result[t] = float(np.corrcoef(wa_v, wb_v)[0, 1])

    return result


def _dyadic_metrics(
    pos_a: np.ndarray, pos_b: np.ndarray, kp_names: list[str],
    ind_a: str, ind_b: str, congruent_window: int,
) -> pd.DataFrame:
    """Compute all dyadic metrics between two individuals."""
    n_frames = pos_a.shape[0]
    df = pd.DataFrame(index=np.arange(n_frames))

    dist_c, dist_t = _interpersonal_distances(pos_a, pos_b, kp_names)
    df["interpersonal_distance_centroid"] = dist_c
    df["interpersonal_distance_trunk"] = dist_t

    approach = np.full(n_frames, np.nan)
    for t in range(1, n_frames):
        if not np.isnan(dist_c[t]) and not np.isnan(dist_c[t - 1]):
            approach[t] = dist_c[t] - dist_c[t - 1]
    df["interpersonal_approach"] = approach

    df["facingness"] = _facingness(pos_a, pos_b, kp_names)

    vis_a = np.abs(pos_a).sum(axis=-1) > 0
    vis_b = np.abs(pos_b).sum(axis=-1) > 0
    speed_a, _, _, _ = _centroid_speed_velocity(pos_a, vis_a, ind_a)
    speed_b, _, _, _ = _centroid_speed_velocity(pos_b, vis_b, ind_b)
    df["congruent_motion"] = _congruent_motion(speed_a, speed_b, congruent_window)

    return df


def _compute_normalised_metrics(ds: xr.Dataset) -> pd.DataFrame | None:
    """Compute all metrics from an xarray pose Dataset, return normalised DF.

    Parameters
    ----------
    ds : xarray.Dataset
        Contains ``position`` (T, I, K, 2) and ``confidence`` (T, I, K).
        Coordinates: time, individuals, keypoints, space.

    Returns
    -------
    Normalised frame-level DataFrame, or None on failure.
    """
    ind_names = ds.coords["individuals"].values.tolist()

    # Require exactly 2 individuals (child, clinician)
    if len(ind_names) < 2:
        logger.debug("Segment has %d individuals, need 2 — skipping.", len(ind_names))
        return None

    ind_a, ind_b = ind_names[0], ind_names[1]
    fps = float(ds.attrs.get("fps", 25.0))
    kp_names = ds.coords["keypoints"].values.tolist()
    pos = ds["position"].values  # (T, I, K, 2)
    n_frames = pos.shape[0]

    if n_frames < 2:
        return None

    pos_a = pos[:, 0, :, :]  # (T, K, 2)
    pos_b = pos[:, 1, :, :]

    # Trunk heights
    trunk_h_a = _compute_trunk_height(pos_a, kp_names)
    trunk_h_b = _compute_trunk_height(pos_b, kp_names)

    # Individual kinematics  (use canonical names: child, clinician)
    df_a = _individual_kinematics(pos_a, kp_names, prefix=INDIVIDUAL_CHILD)
    df_b = _individual_kinematics(pos_b, kp_names, prefix=INDIVIDUAL_CLINICIAN)

    # Total distance / duration
    for ind, df_ind in ((INDIVIDUAL_CHILD, df_a), (INDIVIDUAL_CLINICIAN, df_b)):
        df_ind[f"{ind}_total_distance_centroid"] = _total_distance_over_duration(
            df_ind[f"{ind}_speed_centroid"].values, fps,
        )
        df_ind[f"{ind}_total_distance_trunk"] = _total_distance_over_duration(
            df_ind[f"{ind}_speed_trunk"].values, fps,
        )

    # Dyadic metrics
    df_dyadic = _dyadic_metrics(
        pos_a, pos_b, kp_names,
        INDIVIDUAL_CHILD, INDIVIDUAL_CLINICIAN,
        DEFAULT_CONGRUENT_WINDOW,
    )

    # Assemble raw DataFrame
    raw = pd.concat([df_a, df_b, df_dyadic], axis=1)
    raw["agitation_global_ke"] = raw[
        [f"{INDIVIDUAL_CHILD}_kinetic_energy", f"{INDIVIDUAL_CLINICIAN}_kinetic_energy"]
    ].mean(axis=1)

    # ── Normalisation ──
    norm = raw.copy()

    # Individual metrics ÷ own trunk height
    for ind, trunk_h in ((INDIVIDUAL_CHILD, trunk_h_a), (INDIVIDUAL_CLINICIAN, trunk_h_b)):
        cols = [c for c in df_a.columns if c.startswith(f"{ind}_")]
        cols = [c for c in cols if any(k in c for k in ("speed", "acc", "ke", "velocity"))]
        # Don't normalise kp_set_changed (bool) or total_distance (separate)
        cols = [c for c in cols if "total_distance" not in c and "kp_set_changed" not in c]
        for col in cols:
            if col in norm.columns:
                norm[col] = _safe_divide(norm[col].values, trunk_h)

    # Dyadic distances ÷ mean trunk height
    mean_trunk = np.where(
        np.isnan(trunk_h_a) | np.isnan(trunk_h_b), np.nan,
        (trunk_h_a + trunk_h_b) / 2.0,
    )
    for col in [c for c in df_dyadic.columns if "distance" in c or "approach" in c]:
        if col in norm.columns:
            norm[col] = _safe_divide(norm[col].values, mean_trunk)

    # Agitation ÷ mean trunk height
    norm["agitation_global_ke"] = _safe_divide(
        norm["agitation_global_ke"].values, mean_trunk,
    )

    # Total distance ÷ median trunk height (per individual)
    for ind, trunk_h in ((INDIVIDUAL_CHILD, trunk_h_a), (INDIVIDUAL_CLINICIAN, trunk_h_b)):
        med_th = float(np.nanmedian(trunk_h)) if not np.all(np.isnan(trunk_h)) else np.nan
        for suffix in ("centroid", "trunk"):
            col = f"{ind}_total_distance_{suffix}"
            if col in norm.columns:
                if np.isnan(med_th) or med_th == 0:
                    norm[col] = np.nan
                else:
                    norm[col] = norm[col] / med_th

    # Facingness and congruent_motion are angles/correlations — not normalised

    return norm


# ─────────────────────────────────────────────────────────────
# Segment / subject discovery and matching
# ─────────────────────────────────────────────────────────────

_SEG_SUFFIX_RE = re.compile(r"_seg_\d+$")


def _video_id_from_segment(segment_name: str) -> str:
    """Strip ``_seg_NNN`` suffix to get video/subject ID."""
    return _SEG_SUFFIX_RE.sub("", segment_name)


def _match_video_to_csv(
    video_id: str, meta: pd.DataFrame,
) -> pd.Series | None:
    """Map video_id to CSV row.

    V-codes (V001, V012, …) match the ``code`` column.
    Numeric IDs (8042_T2a_ADOS, …) match first 4 chars to ``sujet_id``.
    """
    if video_id.startswith("V"):
        code = video_id  # e.g. "V012"
        mask = meta["code"].astype(str).str.strip() == code
        matches = meta[mask]
        if len(matches) == 1:
            return matches.iloc[0]
        if len(matches) > 1:
            logger.warning("Multiple CSV rows for code=%s, using first.", code)
            return matches.iloc[0]
        return None

    sujet_id = video_id[:4]  # first 4 chars = numeric subject ID
    mask = meta["sujet_id"].astype(str).str.strip() == sujet_id
    matches = meta[mask]
    if len(matches) == 1:
        return matches.iloc[0]
    if len(matches) > 1:
        logger.warning("Multiple CSV rows for sujet_id=%s, using first.", sujet_id)
        return matches.iloc[0]
    return None


def _load_one_segment(segment_dir: Path) -> pd.DataFrame | None:
    """Load tracking.nc from a segment directory and compute normalised metrics."""
    nc_path = segment_dir / "tracking.nc"
    if not nc_path.exists():
        return None
    try:
        ds = xr.open_dataset(nc_path)
        result = _compute_normalised_metrics(ds)
        ds.close()
        return result
    except Exception as exc:
        logger.debug("Failed to load %s: %s", nc_path, exc)
        return None


def _process_one_subject(
    video_id: str,
    segment_dirs: list[Path],
    meta_row: pd.Series,
    selected_stats: list[str] | None = None,
    raw_output_dir: Path | None = None,
) -> dict | None:
    """Load all segments for one subject, compute summary statistics."""
    segment_dfs: list[pd.DataFrame] = []
    for seg_dir in sorted(segment_dirs):
        df_seg = _load_one_segment(seg_dir)
        if df_seg is not None:
            segment_dfs.append(df_seg)

    if not segment_dfs:
        logger.debug("No valid segments for %s", video_id)
        return None

    # Concatenate all segment frames
    df_all = pd.concat(segment_dfs, ignore_index=True)

    if raw_output_dir is not None:
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_output_dir / f"{video_id}_frame_metrics.csv.gz"
        df_all.to_csv(raw_path, index=False, compression="gzip")

    # Drop trunk variant columns before computing summary stats
    if EXCLUDE_TRUNK_COLUMNS:
        trunk_cols = [c for c in df_all.columns if c.endswith("_trunk")]
        # Also velocity_trunk_x/y
        trunk_cols += [c for c in df_all.columns if "_trunk_" in c]
        trunk_cols = list(set(trunk_cols))
        df_all = df_all.drop(columns=trunk_cols, errors="ignore")

    # Compute summary statistics across all frames
    records: dict[str, float] = {}
    for col in df_all.columns:
        arr = df_all[col].values
        dtype = df_all[col].dtype

        if dtype == bool or str(dtype) == "bool":
            for stat_name, val in _stats_bool(arr).items():
                records[f"{col}__{FEATURE_VARIANT}__{stat_name}"] = val
        else:
            arr_float = arr.astype(float)
            for stat_name, val in _stats_float(
                arr_float,
                selected=selected_stats,
                track_clipping=True,
            ).items():
                records[f"{col}__{FEATURE_VARIANT}__{stat_name}"] = val

    # Merge clinical metadata
    clinical_cols = [c for c in META_COLS if c in meta_row.index]
    result = {k: meta_row[k] for k in clinical_cols}
    result.update(records)
    return result


# ─────────────────────────────────────────────────────────────
# Full-dataset loader (kinematic)
# ─────────────────────────────────────────────────────────────

def load_kinematic_features(
    csv_path: Path,
    pose_ready_dir: Path,
    n_jobs: int = 4,
    debug_n: int | None = None,
    selected_stats: list[str] | None = None,
    raw_output_dir: Path | None = None,
) -> pd.DataFrame:
    """Load raw pose data, compute kinematic + social metrics, return features.

    Parameters
    ----------
    csv_path : Path
        Path to the clinical metadata CSV.
    pose_ready_dir : Path
        Directory containing per-segment pose data (``{video_id}_seg_NNN/tracking.nc``).
    n_jobs : int
        Parallel workers for subject processing.
    debug_n : int or None
        Limit to first N subjects for quick testing.
    raw_output_dir : Path or None
        If provided, save per-frame kinematic/social metrics before aggregation
        as compressed CSV files in this directory.

    Returns
    -------
    DataFrame with one row per subject.  Columns include clinical meta
    (``META_COLS``) and feature columns named ``{metric}__norm__{stat_type}``.
    """
    # Load clinical metadata
    meta = pd.read_csv(csv_path)
    meta = meta[meta["diagnosis"].isin(["ASD", "TD"])].reset_index(drop=True)
    logger.info("Clinical CSV: %d subjects with ASD/TD diagnosis", len(meta))

    # Discover segments and group by video_id
    all_segment_dirs = sorted(
        p for p in pose_ready_dir.iterdir() if p.is_dir()
    )
    logger.info("Found %d segment directories in %s", len(all_segment_dirs), pose_ready_dir)

    groups: dict[str, list[Path]] = defaultdict(list)
    for seg_dir in all_segment_dirs:
        vid = _video_id_from_segment(seg_dir.name)
        groups[vid].append(seg_dir)

    logger.info("Grouped into %d video/subject IDs:", len(groups))
    for vid, seg_dirs in sorted(groups.items()):
        logger.info("  %s: %d segment(s)", vid, len(seg_dirs))

    # Build tasks: match video_id → clinical CSV row
    tasks: list[tuple[str, list[Path], pd.Series]] = []
    unmatched: list[str] = []

    for vid, seg_dirs in sorted(groups.items()):
        meta_row = _match_video_to_csv(vid, meta)
        if meta_row is None:
            unmatched.append(vid)
            continue
        tasks.append((vid, seg_dirs, meta_row))

    if unmatched:
        logger.info(
            "Skipped %d video IDs (no CSV match or not ASD/TD): %s%s",
            len(unmatched),
            ", ".join(unmatched[:5]),
            "…" if len(unmatched) > 5 else "",
        )

    if debug_n is not None:
        tasks = tasks[:debug_n]
        logger.info("Debug mode: limited to first %d subjects", debug_n)

    logger.info(
        "Computing features for %d subjects (n_jobs=%d) …",
        len(tasks), n_jobs,
    )

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_process_one_subject)(
            vid,
            seg_dirs,
            meta_row,
            selected_stats,
            raw_output_dir,
        )
        for vid, seg_dirs, meta_row in tqdm(tasks, desc="Computing features")
    )

    rows = [r for r in results if r is not None]
    n_skipped = len(results) - len(rows)
    if n_skipped:
        logger.warning("Skipped %d subjects (no valid segments)", n_skipped)

    df = pd.DataFrame(rows)

    # Summarise and drop Tukey clipping tracker columns
    clip_cols = [c for c in df.columns if c.endswith("___clip_pct")]
    if clip_cols:
        mean_clip = df[clip_cols].mean().mean()
        max_clip = df[clip_cols].max().max()
        logger.info(
            "Tukey clipping summary (kinematic): mean=%.2f%%, max=%.2f%% across %d metrics",
            mean_clip, max_clip, len(clip_cols),
        )
    clip_all = [c for c in df.columns if "___clip_" in c]
    if clip_all:
        df = df.drop(columns=clip_all)
    n_feat = len(get_feature_columns(df))
    n_meta = len(df.columns) - n_feat
    logger.info(
        "Feature matrix: %d subjects × %d feature columns (+%d meta = %d total)",
        len(df), n_feat, n_meta, len(df.columns),
    )
    return df


# ─────────────────────────────────────────────────────────────
# Feature column helpers
# ─────────────────────────────────────────────────────────────

def _is_trunk_variant(col: str) -> bool:
    """Return True if the column is a trunk variant (not centroid).

    Trunk variants end in ``_trunk__norm__<stat>`` while centroid variants
    end in ``_centroid__norm__<stat>``.  Metrics that have both (speed,
    total_distance, interpersonal_distance) should keep only the centroid.
    """
    base = col.split("__")[0]
    return base.endswith("_trunk")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all normalised feature column names from the DataFrame.

    When ``EXCLUDE_TRUNK_COLUMNS`` is True, columns whose base metric ends
    in ``_trunk`` are dropped (e.g. ``child_speed_trunk``,
    ``interpersonal_distance_trunk``).  Only the centroid variant is kept.
    """
    cols = [c for c in df.columns if f"__{FEATURE_VARIANT}__" in c]
    if EXCLUDE_TRUNK_COLUMNS:
        cols = [c for c in cols if not _is_trunk_variant(c)]
    return cols


def filter_feature_columns(
    feature_cols: list[str],
    representation: str,
) -> list[str]:
    """Filter feature columns by representation type.

    Parameters
    ----------
    feature_cols : list[str]
        Full list of normalised feature columns (trunk variants already
        excluded by ``get_feature_columns`` when ``EXCLUDE_TRUNK_COLUMNS``
        is True).
    representation : str
        One of REPR_MOTOR_ONLY, REPR_FULL_KINEMATIC, REPR_EMBEDDING.

    Returns
    -------
    Filtered list of feature column names.
    """
    # Safety: also apply trunk filter here in case called on raw column list
    if EXCLUDE_TRUNK_COLUMNS:
        feature_cols = [c for c in feature_cols if not _is_trunk_variant(c)]

    if representation == REPR_FULL_KINEMATIC:
        return feature_cols  # motor + social = all

    if representation == REPR_MOTOR_ONLY:
        return [c for c in feature_cols if is_motor_feature(c)]

    raise ValueError(
        f"Cannot filter kinematic columns for representation={representation!r}"
    )


def prepare_Xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Extract X, y, metadata from the feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame from ``load_kinematic_features``.
    feature_cols : list[str]
        Feature columns to include.
    target_col : str
        Target column name.

    Returns
    -------
    X : np.ndarray, shape (n_subjects, n_features)
    y : np.ndarray
    df_meta : pd.DataFrame (same rows as X, clinical columns only)
    feature_names : list[str]
    """
    valid_mask = df[target_col].notna()
    if target_col == "diagnosis":
        valid_mask &= df[target_col].isin(["ASD", "TD"])
    elif target_col in CLASSIFICATION_TARGETS:
        valid_mask &= df[target_col].astype(str).str.strip().ne("")
    else:
        valid_mask &= pd.to_numeric(df[target_col], errors="coerce").notna()

    df_valid = df[valid_mask].reset_index(drop=True)

    X = df_valid[feature_cols].values.astype(float)
    feature_names = list(feature_cols)

    if target_col in CLASSIFICATION_TARGETS:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df_valid[target_col].astype(str).values)
    else:
        y = pd.to_numeric(df_valid[target_col], errors="coerce").values.astype(float)

    meta_cols = [c for c in META_COLS if c in df_valid.columns]
    df_meta = df_valid[meta_cols].reset_index(drop=True)

    return X, y, df_meta, feature_names


# ─────────────────────────────────────────────────────────────
# Embedding placeholder
# ─────────────────────────────────────────────────────────────

def load_embedding_features(
    csv_path: Path,
    embedding_dir: Path,
    embedding_dim: int = 128,
    debug_n: int | None = None,
    selected_stats: list[str] | None = None,
) -> pd.DataFrame:
    """Load HumanLISBET embedding features for each subject.

    For each subject in the metadata CSV, this function finds all session
    segment folders inside ``embedding_dir`` whose name prefix (first 4
    characters) matches the subject's ``sujet_id``.  All segment CSVs are
    concatenated and then aggregated using the same 11 summary statistics as
    the kinematic loader (``_stats_float``), yielding
    ``embedding_dim × 11`` feature columns per subject.

    Folder naming mirrors the pose_ready directory convention:

    * Numeric subjects: ``<sujet_id>_<session>_seg_<NNN>`` (e.g. ``7797_T1a_ADOS1_seg_001``).
      Matched via first 4 characters of ``sujet_id``.
    * V-code subjects: ``<code>_<session>_seg_<NNN>`` (e.g. ``V001_T2a_ADOS_seg_001``).
      Matched via the ``code`` column value (e.g. ``"V001"``).

    File layout expected inside ``embedding_dir``::

        <embedding_dir>/
            7797_T1a_ADOS1_seg_001/
                features_lisbet_embedding.csv   # rows=frames, cols=0..127
            7797_T1a_ADOS1_seg_002/
                features_lisbet_embedding.csv
            V001_T2a_ADOS_seg_001/
                features_lisbet_embedding.csv
            ...

    The first column of each CSV is the frame index (used as the row index);
    the remaining columns are the embedding dimensions labelled ``0``..``d-1``.

    Notes
    -----
    The encoder was trained on all N≈99 training subjects.  Frame-level
    embeddings for cross-validation fold subjects were therefore "seen" during
    self-supervised pre-training (without labels).  This is a known source of
    optimistic bias — see ``docs/12_limitations.md``.

    Parameters
    ----------
    csv_path : Path
        Clinical metadata CSV (same as kinematic loader).
    embedding_dir : Path
        Directory whose sub-folders contain ``features_lisbet_embedding.csv``
        files (one sub-folder per segment, e.g. ``7797_T1a_ADOS1_seg_001``).
    embedding_dim : int
        Expected embedding dimension *d* of the encoder (used only for
        column-name generation; actual dimension is inferred from the files).
    debug_n : int or None
        Limit to first N subjects.

    Returns
    -------
    DataFrame with one row per subject.  Feature columns are named
    ``emb_{dim}_{stat}`` where dim ∈ 0..d-1 and stat ∈ ``FLOAT_STAT_TYPES``.
    """
    embedding_dir = Path(embedding_dir)

    meta = pd.read_csv(csv_path)
    meta = meta.dropna(subset=["results_path"])
    meta = meta[meta["diagnosis"].isin(["ASD", "TD"])].reset_index(drop=True)

    if debug_n is not None:
        meta = meta.head(debug_n)

    # Index available segment folders by their 4-char prefix.
    # V-code folders (e.g. V001_T2a_ADOS_seg_001) are keyed by "V001";
    # numeric folders (e.g. 7797_T1a_ADOS1_seg_001) by "7797".
    # This mirrors the naming convention used in the pose_ready directories.
    all_seg_dirs = sorted(
        d for d in embedding_dir.iterdir() if d.is_dir()
    )
    prefix_to_dirs: dict[str, list[Path]] = {}
    for seg_dir in all_seg_dirs:
        prefix = seg_dir.name[:4]
        prefix_to_dirs.setdefault(prefix, []).append(seg_dir)

    clinical_cols = [c for c in META_COLS if c in meta.columns]
    rows = []
    n_no_csv = 0

    # Iterate over what is actually present in the embedding directory.
    # This avoids spurious warnings for subjects that simply belong to the
    # other train/test split (their folders are absent by design).
    for prefix, seg_dirs in sorted(prefix_to_dirs.items()):
        # Re-use _match_video_to_csv logic: V-code prefix ("V001") matches the
        # ``code`` column; numeric prefix ("7797") matches ``sujet_id[:4]``.
        meta_row = _match_video_to_csv(prefix, meta)
        if meta_row is None:
            logger.debug(
                "No CSV row for embedding prefix=%s — skipping.", prefix
            )
            n_no_csv += 1
            continue

        sujet_id = str(meta_row["sujet_id"]).strip()

        # Load and concatenate all segments for this subject.
        seg_dfs: list[pd.DataFrame] = []
        for seg_dir in seg_dirs:
            csv_file = seg_dir / "features_lisbet_embedding.csv"
            if not csv_file.exists():
                logger.warning(
                    "Missing embedding file: %s — skipping segment.", csv_file
                )
                continue
            seg_dfs.append(pd.read_csv(csv_file, index_col=0))

        if not seg_dfs:
            logger.warning(
                "All segment files missing for sujet_id=%s (prefix=%s) — skipping.",
                sujet_id, prefix,
            )
            continue

        emb_df = pd.concat(seg_dfs, ignore_index=True)  # shape (T_total, d)
        emb_df.columns = emb_df.columns.astype(str)

        # Build feature record for this subject.
        record = {k: meta_row[k] for k in clinical_cols if k in meta_row.index}
        for col in emb_df.columns:
            stats = _stats_float(
                emb_df[col].values.astype(float),
                selected=selected_stats,
                track_clipping=True,
            )
            for stat_name, stat_val in stats.items():
                record[f"emb_{col}_{stat_name}"] = stat_val

        rows.append(record)

    if n_no_csv:
        logger.debug(
            "%d embedding prefixes had no matching CSV row (likely other split).",
            n_no_csv,
        )

    df = pd.DataFrame(rows)

    # Summarise and drop Tukey clipping tracker columns
    clip_cols = [c for c in df.columns if c.endswith("_clip_pct")]
    if clip_cols:
        mean_clip = df[clip_cols].mean().mean()
        max_clip = df[clip_cols].max().max()
        logger.info(
            "Tukey clipping summary (embedding): mean=%.2f%%, max=%.2f%% across %d dims",
            mean_clip, max_clip, len(clip_cols),
        )
    clip_all = [c for c in df.columns if "_clip_n" in c or "_clip_pct" in c]
    if clip_all:
        df = df.drop(columns=clip_all)

    n_emb_cols = len([c for c in df.columns if c.startswith("emb_")])
    n_meta = len(df.columns) - n_emb_cols
    logger.info(
        "Embedding DataFrame: %d subjects × %d embedding features (+%d meta = %d total)",
        len(df), n_emb_cols, n_meta, len(df.columns),
    )
    return df


def get_embedding_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return embedding feature column names."""
    return [c for c in df.columns if c.startswith("emb_")]


def merge_kinematic_embedding(
    df_kin: pd.DataFrame,
    df_emb: pd.DataFrame,
) -> pd.DataFrame:
    """Merge kinematic and embedding DataFrames on ``sujet_id``.

    Returns a single DataFrame with META_COLS (taken from *df_kin*),
    all kinematic feature columns, and all embedding feature columns.
    Only subjects present in **both** DataFrames are kept (inner join).
    """
    emb_feature_cols = get_embedding_feature_columns(df_emb)
    df_emb_slim = df_emb[["sujet_id"] + emb_feature_cols].copy()
    df_emb_slim["sujet_id"] = df_emb_slim["sujet_id"].astype(str).str.strip()

    df_kin = df_kin.copy()
    df_kin["sujet_id"] = df_kin["sujet_id"].astype(str).str.strip()

    merged = df_kin.merge(df_emb_slim, on="sujet_id", how="inner")
    logger.info(
        "Merged kinematic (%d) + embedding (%d) → %d subjects",
        len(df_kin), len(df_emb), len(merged),
    )
    return merged


def prepare_Xy_embedding(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Extract X, y, metadata from the embedding DataFrame."""
    feature_cols = get_embedding_feature_columns(df)
    return prepare_Xy(df, feature_cols, target_col)


# ─────────────────────────────────────────────────────────────
# Convenience: load all representations
# ─────────────────────────────────────────────────────────────

def load_all_representations(
    csv_path: Path,
    pose_ready_dir: Path,
    n_jobs: int = 4,
    debug_n: int | None = None,
    embedding_dir: Path | None = None,
    raw_output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load data for all feature representations.

    Returns
    -------
    dict mapping representation name → DataFrame.
    Motor-only and full-kinematic share the same DataFrame (column filtering
    happens downstream via ``filter_feature_columns``).
    If ``embedding_dir`` is None the embedding representation is omitted.
    When ``raw_output_dir`` is provided, per-frame kinematic metrics are saved
    before aggregation.
    """
    logger.info("=" * 65)
    logger.info("Loading kinematic features")
    logger.info("=" * 65)

    df_kin = load_kinematic_features(
        csv_path=csv_path,
        pose_ready_dir=pose_ready_dir,
        n_jobs=n_jobs,
        debug_n=debug_n,
        raw_output_dir=raw_output_dir,
    )

    result = {
        REPR_MOTOR_ONLY: df_kin,
        REPR_FULL_KINEMATIC: df_kin,   # same df, different column filter
    }

    if embedding_dir is not None:
        logger.info("=" * 65)
        logger.info("Loading embedding features from %s", embedding_dir)
        logger.info("=" * 65)
        df_emb = load_embedding_features(
            csv_path=csv_path,
            embedding_dir=embedding_dir,
            debug_n=debug_n,
        )
        result[REPR_EMBEDDING] = df_emb

    return result
