# Slide 10 — Results: Regression Performance

---

## Test Set Performance (n = 644 patient-eyes)

| Model | MD MAE (dB/yr) | MD RMSE | MD R² |
|---|---|---|---|
| Naive (global mean) | 0.524 | 0.797 | — |
| Ridge (baseline MD) | 0.527 | 0.802 | — |
| **GLAM (this work)** | **0.139** | **0.238** | **0.927** |

- **73.5% reduction in MAE** vs. naive baseline
- **VFI MAE: 0.265 %/year** | VFI R² = 0.941

---

## What This Means Clinically

- A naive model predicts ~−0.13 dB/year for every patient → classifies everyone as stable
- GLAM accurately estimates rates down to **~−3 dB/year** with near-zero bias
- **Prediction bias: 0.001 dB/year** — essentially no systematic over/under-prediction

---

## Subgroup Error Analysis (MD Quintiles)

| Quintile | MAE (dB/yr) |
|---|---|
| Q1 — Most stable (≥ 0) | 0.092 |
| Q2 | 0.107 |
| Q3 | 0.131 |
| Q4 | 0.148 |
| Q5 — Fastest (< −0.82) | 0.205 |

Error increases with progression severity — expected given greater inherent variability in rapid progression.
