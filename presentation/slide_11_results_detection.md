# Slide 11 — Results: Fast-Progressor Detection

---

## Clinical Task

- **Fast progressor:** MD rate < −1 dB/year
- Only **8.7%** of test set (56 of 644 eyes) — highly imbalanced
- Missing a fast progressor = irreversible vision loss

---

## ROC Performance


| Metric                     | Value                                                     |
| -------------------------- | --------------------------------------------------------- |
| AUC-ROC                    | **0.990**                                                 |
| Optimal threshold (Youden) | −0.58 dB/year                                             |
| Sensitivity                | **98.2%** (55/56 fast progressors detected)               |
| Specificity                | **90.6%** (534/588 slow progressors correctly classified) |
| PPV                        | 50.0%                                                     |
| F1 Score                   | 0.663                                                     |


---

## Clinical Trade-Off

- **Only 1 fast progressor missed** out of 56 in the test set
- ~1 in 10 stable patients flagged for unnecessary review — clinically acceptable
- At high-specificity operating point (Sens ≥ 95%):
  - Specificity: 93.4%, PPV: 56.0%
  - Only **1.5 false-positive reviews per true fast progressor detected**

---

## Comparison with Prior Work


| Study                | AUC       | Data                             |
| -------------------- | --------- | -------------------------------- |
| Bowd et al. (2024)   | 0.97      | OCT + VF + clinical (n > 10,000) |
| **GLAM (this work)** | **0.990** | **VF only** (n = 644 test)       |


