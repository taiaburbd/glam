# Slide 06 — GLAM Architecture Overview

---

## GLAM — Four-Component Pipeline

```
Input: T visits × 54 TD points  +  5 clinical features
         │
         ▼
[VisualFieldEncoder]   [Clinical Embedder]   [Structural Branch*]
  54 → 256-dim           5 → 128-dim            64-dim zeros
         │                     │                      │
         ▼
   [BiLSTM × 2]
   256-hidden → 512-dim functional vector
         │
         └────────────────────┬──────────────────┘
                              ▼
                   [Attention Fusion Module]
                    Learns soft weights α over modalities
                              │
                              ▼
                   [Prediction Head]
                    MD rate (dB/yr) + VFI rate (%/yr)
                    + Aleatoric uncertainty (log-variance)
```

**Total trainable parameters: 13,605,319**

*Structural branch is inactive (zero input) — UWHVF has no imaging data.
