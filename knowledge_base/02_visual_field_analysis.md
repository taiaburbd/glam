# Visual Field Analysis for Deep Learning

## VF Data Structure (24-2 Test)

### Point Layout

The HFA 24-2 pattern tests **54 points** (76 less the 2 blind spot points) arranged in a 6×9-ish grid covering ±24° horizontally and ±24° vertically.

```
Point indices (approximate spatial layout, right eye):
Row 1:  [ -  -  -  -  -  -  -  - ]   (superior)
Row 2:  [ -  -  -  -  -  -  -  - ]
Row 3:  [ -  -  -  -  -  -  -  - ]
...
Row 8:  [ -  -  -  -  -  -  -  - ]   (inferior)

Each point:
  value: sensitivity in dB (0 = no light perception, ~30 = normal)
  TD (Total Deviation): point_value - age_normal_value
  PD (Pattern Deviation): corrects for generalized sensitivity loss
```

### Data Representation for DL

```python
# Single VF test: shape (54,)
vf_test = np.array([28, 27, 30, 25, ...])  # raw sensitivity dB

# Longitudinal series: shape (T, 54)
vf_series = np.array([
    [28, 27, 30, ...],   # Visit 1 (baseline)
    [26, 25, 28, ...],   # Visit 2
    [24, 23, 26, ...],   # Visit 3
    ...
])

# Target: rate of change per location
vf_slopes = np.array([...])  # shape (54,) — dB/year per point
```

---

## Progression Detection Methods

### Trend-Based (what we're predicting)

- **Linear regression** of MD over time → slope = dB/year
- More sensitive, requires ≥5 reliable tests
- OLS is standard; weighted OLS gives less weight to unreliable tests

### Event-Based (comparison)

- **Guided Progression Analysis (GPA)**: Detects if ≥3 points worsen by >6dB
- Requires only 2 baseline tests
- Less sensitive but more specific

### Why Trend-Based for DL

- Provides continuous target (dB/year, %/year) — better for regression
- Captures global rate rather than individual point changes
- More predictable from baseline structural features

---

## Important VF Characteristics for Modeling

### Spatial Correlation

Adjacent VF points are highly correlated — a scotoma covers multiple neighboring points. This means:

- Raw 54-point vector has high redundancy
- Spatial CNN features may be more informative than point-by-point MLPs
- Consider 2D spatial maps for CNN processing

### Temporal Characteristics

- Visit interval: typically 3–6 months
- Total follow-up: 3–10 years for reliable progression detection
- **Problem**: Variable intervals, missing visits, unreliable tests

### Glaucomatous Defect Patterns


| Pattern             | Location                  | Clinical Significance     |
| ------------------- | ------------------------- | ------------------------- |
| Paracentral scotoma | Within 10° of fixation    | High impact on reading    |
| Arcuate defect      | Superior/inferior arcuate | Classic POAG pattern      |
| Nasal step          | Horizontal midline defect | Common early finding      |
| Temporal wedge      | Temporal VF               | Often spared until late   |
| Central field loss  | Final stage               | Severely impacts function |


### Myopia-Specific Artifacts

- **Peripheral ring scotoma**: Myopic refraction error at lens edge
- **Nasal step from tilted disc**: Can mimic glaucomatous defect
- **Temporal VF loss**: From peripapillary atrophy encroaching on VF

---

## Preprocessing Pipeline for VF Data

```python
# 1. Filter unreliable tests
mask = (
    (false_positive_rate <= 0.33) &
    (fixation_loss_rate <= 0.20) &
    (false_negative_rate <= 0.33)
)

# 2. Handle missing VF points (e.g., blind spot, edge artifacts)
vf_data = vf_data.fillna(vf_data.median())

# 3. Normalize to [0, 1]
vf_norm = (vf_data + 35) / 35  # clip to [-35, 0] first

# 4. Compute Total Deviation (subtract age-normal reference)
td = vf_data - age_normal_reference  # shape (54,)

# 5. For temporal sequences: pad to fixed length
max_visits = 10
padded = np.zeros((max_visits, 54))
padded[:n_actual_visits] = vf_series[:n_actual_visits]
```

---

## Feature Engineering Ideas

### Pointwise Slope Features (Traditional ML baseline)

```python
# Per-point OLS slopes as features
slopes = np.zeros(54)
for i in range(54):
    slopes[i] = ols_slope(times, vf_series[:, i])
```

### Spatial Cluster Features

```python
# Superior hemifield MD
superior_md = vf_series[:, superior_indices].mean(axis=1)  # per visit
inferior_md = vf_series[:, inferior_indices].mean(axis=1)

# Hemifield asymmetry (glaucoma indicator)
hemifield_asymmetry = superior_md - inferior_md
```

### Temporal Features

- Number of reliable tests
- Inter-visit interval variability
- Total follow-up duration
- Rate of change acceleration (2nd derivative)

---

## Clinical Decision Thresholds


| MD Rate (dB/year) | Progression Category | Clinical Action                 |
| ----------------- | -------------------- | ------------------------------- |
| 0 to -0.5         | Stable               | Observe                         |
| -0.5 to -1.0      | Slow                 | Consider intensifying treatment |
| -1.0 to -2.0      | Moderate             | Intensify treatment             |
| < -2.0            | Rapid                | Urgent intervention             |


**For the model**: predicting whether a patient will be in the "rapid" category
within 5 years is the most clinically impactful classification task derived from
the regression output.

---

## References

- Bengtsson B, Heijl A. "SITA Fast, a New Rapid Perimetric Threshold Test." Acta Ophthalmol, 1998
- Chauhan BC et al. "Practical Recommendations for Measuring Rates of Visual Field Change." Br J Ophthalmol, 2008
- de Moraes CG et al. "Rates of Progressive Structural and Functional Loss." Ophthalmology, 2011
- Caprioli J, Mock D, Bitrian E. "A Method to Measure and Predict Rates of Regional Visual Field Decay." Ophthalmology, 2011

