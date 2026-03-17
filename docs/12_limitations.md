# Limitations and Caveats

## Sample size

- **Training set**: ~99 subjects. Small for ML, especially with high-
  dimensional kinematic features (~500+ summary statistics). Overfitting
  risk is mitigated by nested CV and feature selection, but external
  validation on independent cohorts is needed.
- **Test set**: ~20 subjects. CIs are wide; held-out results should be
  treated as sanity checks, not definitive performance estimates.

## Class imbalance

- The ASD vs TD split may be imbalanced. LightGBM's `is_unbalance=True`
  and stratified splitting partially address this, but severely imbalanced
  subgroups may still produce unstable fold-level estimates.

## Encoder data leakage (embedding representation)

- If the HumanLISBET encoder was trained on any of the subjects in the
  current dataset (train or test), the embedding representation has a
  potential advantage: task-relevant features were already learned during
  encoder training. This makes the comparison between kinematic and
  embedding representations **not fully fair**.
- Documenting the training set of the encoder is essential for
  interpretation.

## Repeated use of the same data

- The 10 repeated CV shuffles use the same training subjects. While the
  folds differ, the 50 scores are not fully independent. Wilcoxon tests
  assume exchangeability of the paired differences, which approximately
  holds for different shuffles but not exactly. Corrected resampled
  t-tests (Nadeau & Bengio, 2003) could be more appropriate but are
  rarely used in practice.

## Feature engineering choices

- **Only normalised features**: Raw features were excluded based on
  scientific reasoning (depth artefacts). However, some raw features might
  carry unique information (e.g. absolute movement speed correlating with
  motor ability). This is a deliberate trade-off.
- **Tukey fence k=3**: The wide fence keeps most values. With k=1.5
  (classic), more extreme observations would be clipped, potentially
  removing clinically relevant movement bursts (e.g. repetitive behaviours).
- **11 summary statistics**: Temporal ordering information is lost when
  summarising with statistics. If temporal dynamics matter (e.g. patterns
  of behaviour change within a session), these features will miss them.

## Multiple comparison burden

- 4 targets × 3 representation pairs = 12 total Wilcoxon tests. Bonferroni
  correction is applied **within** each target (3 pairs) but not across
  targets. If cross-target correction were applied (12 comparisons),
  significance thresholds would be more conservative.

## Generalisability

- All data is from ADOS-2 sessions recorded in a specific clinical setting.
  Pose estimation quality, room layout, and clinician behaviour may differ
  across centres. The pipeline would need re-validation on data from other
  sites.

## Missing SHAP implementation

- Feature importance is currently not computed. Without it, clinical
  interpretation of which movement features drive predictions is limited.
  See `11_shap_notes.md` for the planned implementation.
