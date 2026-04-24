# Slide 14 — Conclusion & Future Work

---

## Summary

- Presented **GLAM**, a deep learning framework for predicting glaucoma VF progression rates from longitudinal 24-2 TD sequences
- Trained on **4,276 patient-eyes** from the open-access UWHVF dataset

---

## Key Results

| Metric | Value |
|---|---|
| MD MAE | **0.139 dB/year** (73.5% better than baseline) |
| MD R² | **0.927** |
| AUC-ROC (fast-progressor detection) | **0.990** |
| Sensitivity / Specificity | **98.2% / 90.6%** |
| VFI MAE | 0.265 %/year |

- Near-perfect fast-progressor discrimination **using VF data alone** — no OCT required

---

## Future Work

1. **Multimodal extension** — integrate OCT/fundus from GRAPE or Harvard HGDP
2. **Time-aware modeling** — add visit timing via temporal positional encodings
3. **Cross-dataset generalization** — train on UWHVF, test on external cohort
4. **Prospective clinical validation** — real-world glaucoma clinic deployment
5. **Fairness analysis** — demographic subgroup performance (age, sex, ethnicity)

---

## Take-Home Message

> Serial visual field tests alone contain enough spatio-temporal information to predict glaucoma progression with high accuracy. GLAM demonstrates that automated fast-progressor identification from routine VF series is clinically feasible without any additional imaging.
