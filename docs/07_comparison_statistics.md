# Statistical Comparison Across Representations

## Goal

Determine whether the three feature representations (motor-only, full
kinematics, embeddings) produce **significantly different** prediction
performance, and quantify the practical magnitude of any difference.

## Method: Wilcoxon Signed-Rank Test

For each target, the **50 outer-fold scores** (from the 5×10 repeated CV)
are paired across representations for the same (repeat, fold) combination.
The **Wilcoxon signed-rank test** compares the paired scores.

### Why Wilcoxon?

- **Non-parametric**: Does not assume normally distributed differences
  between paired scores
- **Paired**: Accounts for fold-to-fold variability (each fold uses the same
  data partition for all representations)
- **Standard in ML comparison** (Demšar 2006)

### Pairing

Scores are aligned by `(repeat, fold)`. For repeat *r*, fold *f*, the score
for representation A is paired with the score for representation B from the
*same* outer partition.

## Multiple comparison correction: Bonferroni

With 3 representations, there are $\binom{3}{2} = 3$ pairwise comparisons
per target. To control the family-wise error rate:

$$
p_{\text{corrected}} = \min(p \times 3, \; 1.0)
$$

**Adjusted significance threshold**: $\alpha = 0.05 / 3 \approx 0.0167$.

## Effect size: Rank-Biserial Correlation

The rank-biserial $r$ quantifies the magnitude of the difference:

$$
r = 1 - \frac{2W}{n(n+1)/2}
$$

where $W$ is the Wilcoxon statistic and $n$ is the number of non-zero
differences.

### Interpretation (matched-pairs rank-biserial)

| |r| | Interpretation |
|-----|----------------|
| < 0.1 | Negligible |
| 0.1–0.3 | Small |
| 0.3–0.5 | Medium |
| > 0.5 | Large |

## Output

`comparison/comparison_results.csv` with columns:

| Column | Description |
|--------|-------------|
| `rep_a`, `rep_b` | The two representations being compared |
| `metric` | The metric compared (auc_roc or rmse) |
| `stat` | Wilcoxon W statistic |
| `p_value` | Raw p-value |
| `p_corrected` | Bonferroni-corrected p-value |
| `effect_size_r` | Rank-biserial correlation |
| `significant` | Whether p_corrected < 0.05 |
| `n_pairs` | Number of paired scores |
| `n_nonzero` | Number of non-zero differences |
| `mean_a`, `mean_b` | Mean metric value per representation |
| `mean_diff` | Mean difference (A − B) |

## Interpretation guide

1. Check `significant` column for statistically significant differences.
2. If significant, check `effect_size_r` to assess practical importance.
3. A significant result with a small effect size may not be clinically
   meaningful (especially with 50 paired scores where even tiny differences
   can reach significance).
4. Always report both the p-value and the effect size.
