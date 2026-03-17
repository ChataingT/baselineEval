# Evaluation Metrics

## Classification metrics

Used for the diagnosis target (ASD vs TD).

### AUC-ROC (Area Under the Receiver Operating Characteristic)

- **Definition**: Probability that the model ranks a randomly chosen
  positive (ASD) subject higher than a randomly chosen negative (TD)
  subject.
- **Range**: [0, 1]; 0.5 = chance, 1.0 = perfect.
- **Primary metric** for model selection in cross-validation.
- **Interpretation guide**:
  - 0.5–0.6: No discrimination
  - 0.6–0.7: Poor
  - 0.7–0.8: Acceptable
  - 0.8–0.9: Excellent
  - 0.9–1.0: Outstanding

### Balanced accuracy

- **Definition**: $\text{BAcc} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$
- **Range**: [0, 1]; 0.5 = chance.
- **Rationale**: Accounts for class imbalance by averaging per-class
  accuracy.

### F1-macro

- **Definition**: Unweighted mean of per-class F1 scores.
- $F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Range**: [0, 1].

### Sensitivity (recall, true positive rate)

- **Definition**: $\text{Sensitivity} = \frac{TP}{TP + FN}$
- **Interpretation**: Proportion of actual ASD subjects correctly identified.

### Specificity (true negative rate)

- **Definition**: $\text{Specificity} = \frac{TN}{TN + FP}$
- **Interpretation**: Proportion of actual TD subjects correctly identified.

---

## Regression metrics

Used for CSS, SA, and RRB severity targets.

### RMSE (Root Mean Squared Error)

- **Definition**: $\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$
- **Primary metric** for model selection (lower is better).
- **Interpretation**: Average prediction error in the same unit as the
  target. For CSS (range 1–10), an RMSE of 2.0 means ~2 CSS points of
  error on average.

### MAE (Mean Absolute Error)

- **Definition**: $\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|$
- **Interpretation**: Similar to RMSE but less sensitive to extreme errors.

### R² (Coefficient of Determination)

- **Definition**: $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$
- **Range**: $(-\infty, 1]$; 1.0 = perfect, 0.0 = no better than predicting
  the mean, negative = worse than mean.
- **Interpretation guide**:
  - < 0: Model is worse than always predicting the mean
  - 0.0–0.1: Very weak
  - 0.1–0.3: Weak
  - 0.3–0.5: Moderate
  - > 0.5: Strong (unusual in clinical severity prediction)

### Spearman ρ (rank correlation)

- **Definition**: Pearson correlation of the rank-transformed predictions
  and true values.
- **Range**: [−1, 1].
- **Interpretation**: Measures how well the model preserves the ordinal
  ranking of severity scores, regardless of exact values.

### Pearson r (linear correlation)

- **Definition**: Linear correlation between predicted and true values.
- **Range**: [−1, 1].
- **Interpretation**: Measures the linear relationship. Sensitive to
  outliers and scale — Spearman ρ is generally more appropriate for ordinal
  severity scores.

---

## Analysis guidance

- For **clinical utility**, focus on balanced accuracy and sensitivity
  (missing an ASD diagnosis has greater consequences than a false positive).
- For **comparing representations**, use the primary metrics (AUC-ROC for
  classification, RMSE for regression) via Wilcoxon signed-rank tests on
  the 50 outer-fold scores.
- For **assessing regression utility**, interpret RMSE relative to the
  target's scale (CSS: 1–10; SA: 1–10; RRB: 1–4).
