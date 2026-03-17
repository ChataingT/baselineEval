# Embedding Representation — Notes and TODOs

## Status: PLACEHOLDER

The embedding representation (`REPR_EMBEDDING`) is not yet implemented.
The `load_embedding_features()` function in `loading.py` raises
`NotImplementedError` with detailed TODOs.

## Open questions

### 1. What is the embedding dimension?
- The HumanLISBET encoder produces frame-level embeddings of dimension *d*.
- What is *d*? (e.g. 64, 128, 256, 512?)
- This determines the feature matrix width.

### 2. What is the temporal window size?
- Does the encoder use a single frame or a sliding window of frames?
- If windowed, what is the stride?
- This affects alignment with the pose time series.

### 3. Where are the embedding files stored?
- Expected format: `.pt`, `.npy`, or `.pkl` per subject.
- Expected directory: `dataset/embeddings/` or `models/lisbet64x8-calms21U/`?
- File naming convention: by `uuid` or `sujet_id`?

### 4. How to align embeddings to sessions?
- If the encoder produces one embedding per temporal window, how do windows
  map to the session timeline?
- Is there a timestamp or frame-index file?
- Must the alignment match the kinematic frame indices?

### 5. What aggregation method?
- Current plan: mean + std per embedding dimension across all windows,
  producing a 2d-dimensional subject-level feature vector.
- Alternative: use all 11 summary statistics per dimension (= 11d features).
- Alternative: use temporal pooling (max, attention-weighted mean).
- Which approach best preserves the temporal structure that the encoder
  learned?

### 6. Should PCA be applied to embeddings?
- If *d* is already small (e.g. 64), PCA may remove too much variance.
- Current rule: skip PCA if n_features < 64.
- If 2d > 64 (e.g. d=64, mean+std → 128 features), PCA will be applied.

### 7. Encoder data leakage caveat
- If the HumanLISBET encoder was trained on any of the subjects in the
  test set, the embedding comparison is **not fair** (the encoder has
  "seen" those subjects during its own training).
- This must be documented in the paper as a potential confound.
- Ideally, re-encode with a model trained on an independent dataset.

## Implementation plan

When the above questions are answered:

1. Add file-loading logic to `load_embedding_features()`.
2. Implement the chosen aggregation method.
3. Return the same `(X, y, df_meta, feature_names)` tuple as
   `load_kinematic_features()`.
4. Remove the `--skip-embedding` flag from the default SLURM script.
5. Verify PCA behaviour with the actual embedding dimension.
6. Re-run the full pipeline and update the comparison results.
