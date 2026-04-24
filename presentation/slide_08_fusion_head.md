# Slide 08 — Clinical Branch, Attention Fusion & Prediction Head

---

## Clinical Embedder

- Input: 5 normalized clinical features
- Architecture: **2-layer MLP** (5 → 64 → 128) with LayerNorm + GELU
- Output: **128-dimensional clinical embedding**

---

## Attention Fusion Module

- All three branches projected to a **common 512-dim space**
- Soft attention weights computed over modalities:

```
α = softmax(MLP([W_s·h_s ‖ W_f·h_f ‖ W_c·h_c]))
fused = Σᵢ αᵢ · Wᵢ · hᵢ
```

- α is **interpretable** — reveals which modality drives each prediction
- Output: **512-dimensional fused representation**

---

## Prediction Head

- Shared linear: 512 → 256 with LayerNorm + GELU + Dropout(0.3)
- **Task-specific heads:**
  - MD rate (dB/year) — regression
  - VFI rate (%/year) — regression
  - Log-variance for each (aleatoric uncertainty estimation)
