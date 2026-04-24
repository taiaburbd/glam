# Slide 07 — VisualFieldEncoder & BiLSTM

---

## VisualFieldEncoder

- Input: 54-point TD vector per visit
- Architecture: **2-layer FC with LayerNorm + GELU**
  - 54 → 256 → 256 dimensions
- Applied **identically at each time step** (shared weights)
- Output: Sequence of shape **(B, T, 256)**

```
f_t = GELU(LN(W₂ · GELU(LN(W₁ · td_t + b₁)) + b₂))
```

---

## Bidirectional LSTM (BiLSTM)

- Processes the encoded VF sequence over time
- **2 layers**, hidden size **256 per direction → 512 combined**
- Dropout 0.3 between layers
- **Sequence packing** handles variable-length visit sequences
- Output: **512-dimensional functional feature vector** (concat of forward + backward final states)

---

## Why BiLSTM?

- Forward pass: captures accumulation of damage over time
- Backward pass: captures spatial coherence of established defects
- Handles variable visit counts and irregular spacing naturally
