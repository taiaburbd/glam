# Slide 13 — Limitations

---

## Key Limitations to Acknowledge

### 1. Label Derivation from Input
- Both MD and VFI targets are computed from the **same TD sequences** used as model input
- Not data leakage per se, but model performance cannot be directly compared to studies using **instrumentally-measured MD**
- External validation required (e.g., GRAPE, Harvard HGDP datasets)

### 2. VFI Is Not Independent
- VFI proxy = `100 + 2 × MD_proxy` — a linear function of MD
- VFI R² of 0.941 is **not evidence of a distinct second task**

### 3. No Structural Imaging
- UWHVF has no OCT or fundus photos — structural branch is inactive (zero inputs)
- Cannot assess the contribution of structural-functional fusion in this study
- Prior work shows multimodal fusion consistently outperforms VF-alone models, especially in high myopia

### 4. Single-Dataset Evaluation
- One academic glaucoma clinic (UWHVF) — referral bias limits generalizability
- Cross-dataset evaluation is a critical next step

### 5. No Visit Timing in Model Input
- BiLSTM treats visits as an ordered sequence without knowing **actual time elapsed between visits**
- Incorporating elapsed time (e.g., time-aware RNNs) expected to improve performance
