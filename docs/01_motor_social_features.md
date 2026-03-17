# Motor and Social Feature Definitions

## Feature categories

Features are split into two categories based on whether they describe
**individual** (motor) or **dyadic** (social/interpersonal) movement
qualities. The split is determined by the metric prefix in the column name.

---

## Motor (individual) features

These capture the movement characteristics of a single person (child or
clinician). Each metric exists with both `child_` and `clinician_` prefixes.

### Speed

- **Prefix**: `child_speed_`, `clinician_speed_`
- **Definition**: Instantaneous speed (magnitude of velocity vector) of
  specific body parts, computed frame-to-frame.
- **Unit**: pixels/frame (raw) or trunk-heights/frame (normalised).
- **Interpretation**: How fast a person's body part is moving at each frame.

### Acceleration

- **Prefix**: `child_acceleration_`, `clinician_acceleration_`
- **Definition**: Rate of change of speed between consecutive frames.
- **Unit**: pixels/frame² (raw) or trunk-heights/frame² (normalised).
- **Interpretation**: Measures jerkiness or smoothness of movement. High
  acceleration variance → discontinuous, erratic movements.

### Velocity

- **Prefix**: `child_velocity_`, `clinician_velocity_`
- **Definition**: Signed velocity components (directional), as opposed to
  the unsigned speed.
- **Unit**: Same as speed but signed.
- **Interpretation**: Captures directionality. Useful for distinguishing
  approach vs withdrawal movements.

### Kinetic energy

- **Prefix**: `child_kinetic_energy`, `clinician_kinetic_energy`
- **Definition**: $\text{KE} = \tfrac{1}{2} \sum_k m_k \, v_k^2$
  (summed over all keypoints *k*; $m_k = 1$ by convention).
- **Unit**: Arbitrary (consistent within normalisation).
- **Interpretation**: A holistic measure of overall body movement activity.
  High KE → the person is moving a lot.

### Total distance

- **Prefix**: `child_total_distance_`, `clinician_total_distance_`
- **Definition**: Cumulative distance travelled by a body part over the
  observation window.
- **Unit**: pixels (raw) or trunk-heights (normalised).
- **Interpretation**: Total extent of movement. Unlike speed, this is
  cumulative (not instantaneous).

### Keypoint set changed

- **Prefix**: `child_kp_set_changed`, `clinician_kp_set_changed`
- **Definition**: Binary flag per frame: 1 if the set of visible keypoints
  changed compared to the previous frame, 0 otherwise.
- **Interpretation**: Proxy for pose reliability / occlusion. Frequent
  keypoint set changes suggest unstable tracking or partial occlusion.

---

## Social (interpersonal) features

These capture the dyadic relationship between child and clinician.

### Interpersonal distance

- **Prefix**: `interpersonal_distance_`
- **Sub-metrics**: `_centroid` (between body centroids), `_trunk` (between
  trunk keypoints).
- **Definition**: Euclidean distance between the two people's reference
  points at each frame.
- **Unit**: pixels (raw) or trunk-heights (normalised).
- **Interpretation**: Proxemics — how close the participants are. Relevant
  for social engagement and comfort.

### Interpersonal approach

- **Prefix**: `interpersonal_approach`
- **Definition**: Rate of change of interpersonal distance. Negative values
  indicate approach (getting closer); positive values indicate withdrawal.
- **Interpretation**: Captures the dynamic of social approach/avoidance
  behaviour.

### Facingness

- **Prefix**: `facingness`
- **Definition**: A measure of mutual orientation. Typically the cosine of
  the angle between each person's facing direction and the vector connecting
  them.
- **Range**: Generally [−1, 1]; higher = more face-to-face.
- **Interpretation**: Mutual gaze / social orientation. Important for joint
  attention.

### Congruent motion

- **Prefix**: `congruent_motion`
- **Definition**: Correlation or similarity of the child's and clinician's
  velocity vectors within a temporal window.
- **Interpretation**: Measures motor synchrony. High congruence = the dyad
  moves in similar ways simultaneously (e.g. mirroring, imitation).

### Agitation (global kinetic energy)

- **Prefix**: `agitation_global_ke`
- **Definition**: Combined kinetic energy of both child and clinician.
- **Interpretation**: Overall dyadic activity level. Distinguished from
  individual KE by summing over both participants.

---

## Normalisation

All features are available in two variants (identified by `__raw__` or
`__norm__` in the column name). **Only the normalised variant** (`__norm__`)
is used in this pipeline.

Normalisation divides each metric by the participant's trunk height (the
distance between neck and hip keypoints), removing body-size and depth
artefacts from the monocular 2-D projection.
