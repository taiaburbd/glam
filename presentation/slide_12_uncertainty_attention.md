# Slide 12 — Uncertainty & Attention Analysis

---

## Uncertainty Calibration

- GLAM outputs **aleatoric uncertainty** via a log-variance head
- Empirical coverage of nominal 90% prediction intervals: **99.8%**
  - Model is conservative (over-dispersed) — wider intervals than strictly needed
- Predicted σ correlates positively with absolute error (Spearman ρ = 0.41, p < 0.001)
  - Model correctly assigns **higher uncertainty to harder predictions**
- Note: Isotonic regression recalibration recommended before clinical deployment

---

## Modality Attention Weights (Test Set)

| Modality | All (n=644) | Fast progressors (n=56) | Slow (n=588) |
|---|---|---|---|
| Structural (placeholder zeros) | 0.321 | 0.329 | 0.317 |
| **Functional (BiLSTM)** | **0.410** | 0.396 | 0.413 |
| Clinical (MLP) | 0.271 | 0.275 | 0.270 |

- Functional branch receives highest weight — as expected
- Structural branch is zero but receives non-zero attention: model distributes attention across available channels when one is uninformative
- Fast progressors show marginal shift toward temporal VF patterns vs. slow progressors
