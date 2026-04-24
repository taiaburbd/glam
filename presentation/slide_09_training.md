# Slide 09 — Training Details

---

## Loss Function

- **Weighted Dual Huber Regression Loss:**

```
L = w_MD · L_Huber(MD_pred, MD_true; δ=1.0)
  + 0.8 · L_Huber(VFI_pred, VFI_true; δ=1.0)
```

- Fast progressors (MD < −1 dB/year): **weight = 2.5**
- All others: weight = 1.0
- Huber loss: robust to outlier progression rates
- MD:VFI ratio 1.0:0.8 — reflects clinical primacy of MD

---

## Optimizer & Schedule

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 10⁻⁴ |
| Gradient clipping | L2 norm 1.0 |
| LR schedule | Cosine annealing → min 10⁻⁶ over 60 epochs |
| Early stopping patience | 12 epochs |
| Batch size | 64 |
| Random seed | 42 |

---

## Training Outcome

- **Converged at epoch 29**
- Validation MD MAE at best checkpoint: **0.153 dB/year**
- Hardware: Single NVIDIA GPU, PyTorch 2.x
