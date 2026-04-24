# Slide 03 — Related Work

---

## What Has Been Done?

| Study | Approach | Key Result |
|---|---|---|
| Bowd et al. (2024) | Multimodal DL (OCT + VF + clinical) | AUC 0.97, detects progression 3.5 yrs earlier |
| Hybrid-VF-Net (2025) | CNN + RNN for VF forecasting | Lower error than linear extrapolation |
| Habib et al. (2022) | Random forests on UWHVF | MD slope R² ≈ 0.65–0.75 |

---

## Gap in the Literature

- Most high-performing models require **multimodal data** (OCT + VF + fundus)
- OCT is **not uniformly available** in all clinical settings
- **VF data alone** is routinely captured — can it be enough?

---

## Our Contribution

- A **VF-only deep learning model** (GLAM) that matches or exceeds multimodal baselines
- Uses **BiLSTM + Attention Fusion** on longitudinal 54-point TD sequences
- Trained and evaluated on the **largest publicly available VF dataset** (UWHVF)
