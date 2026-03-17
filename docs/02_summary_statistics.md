# Summary Statistics

## Overview

Each frame-level metric time series is summarised into **11 scalar
statistics** per subject. These statistics are computed on the
**Tukey-fence-clipped** version of the series to remove extreme outliers
before aggregation.

## Tukey fence clipping

Before computing statistics, each numeric series is clipped using the
**Tukey fence** method:

$$
\text{lower} = Q_1 - k \times \text{IQR}, \quad
\text{upper} = Q_3 + k \times \text{IQR}
$$

where $Q_1$, $Q_3$ are the 25th and 75th percentiles, $\text{IQR} = Q_3 - Q_1$,
and $k = 3$ (a wide fence — less aggressive than the classic $k = 1.5$).

Values below lower or above upper are **clamped** to the fence boundary
(not removed). This preserves the original number of frames while reducing
the influence of tracking errors or extreme outliers.

## Statistic definitions

| # | Name | Formula / description |
|---|------|----------------------|
| 1 | **mean** | $\bar{x} = \frac{1}{n}\sum x_i$ |
| 2 | **std** | $s = \sqrt{\frac{1}{n-1}\sum(x_i - \bar{x})^2}$ (sample standard deviation, ddof=1) |
| 3 | **q10** | 10th percentile |
| 4 | **q25** | 25th percentile ($Q_1$) |
| 5 | **median** | 50th percentile |
| 6 | **q75** | 75th percentile ($Q_3$) |
| 7 | **q90** | 90th percentile |
| 8 | **iqr** | Interquartile range: $Q_3 - Q_1$ |
| 9 | **cv** | Coefficient of variation: $s / |\bar{x}|$ (NaN if $\bar{x} = 0$) |
| 10 | **skewness** | Fisher's skewness: $\frac{m_3}{m_2^{3/2}}$ where $m_k$ is the $k$-th central moment |
| 11 | **kurtosis** | Fisher's excess kurtosis: $\frac{m_4}{m_2^2} - 3$ |

### Boolean metrics

For binary (boolean) metrics (e.g. `kp_set_changed`), only the **mean** is
computed (proportion of `True` frames).

## Column naming convention

Each summary statistic column follows the pattern:

```
{metric_name}__{variant}__{stat}
```

For example:
- `child_speed_right_wrist__norm__mean` — mean normalised right wrist speed
- `interpersonal_distance_centroid__norm__cv` — coefficient of variation of
  normalised centroid distance

## Minimum valid frames

A minimum of **50 valid (non-NaN) frames** is required to compute summary
statistics. Subjects below this threshold receive NaN for all statistics of
that metric.

## Configurable statistic selection

By default all 11 statistics are computed and returned. You can select a
subset via the `--stats` CLI flag:

```bash
# Use a named preset
python -m ccnEvaluation.src.run_pipeline --stats basic      # mean, std
python -m ccnEvaluation.src.run_pipeline --stats moments     # mean, std, skewness, kurtosis

# Or list individual statistics
python -m ccnEvaluation.src.run_pipeline --stats mean,std,median
```

Internally, all 11 statistics are always computed (some depend on others via
Tukey clipping), but only the selected subset is kept in the final feature
matrix. This lets you compare model performance across different feature
dimensionalities without modifying the source code.

Available presets are defined in `config.py → STAT_PRESETS`.

## Tukey clipping diagnostics

When `track_clipping=True` (enabled automatically during feature loading),
each call to `_stats_float` counts how many values were clipped by the Tukey
fence and emits a `DEBUG`-level log line:

```
Tukey clip: 1.23% of 4500 values clipped (k=3)
```

After all subjects are processed, each loader (`load_kinematic_features`,
`load_embedding_features`) logs an `INFO`-level aggregate summary:

```
Tukey clipping summary (kinematic): mean=0.45%, max=3.21% across 42 metrics
```

To see per-column clip details, run with `--log-level DEBUG`.
