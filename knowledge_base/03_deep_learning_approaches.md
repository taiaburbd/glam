# Deep Learning Approaches for Glaucoma Progression

## Architecture Overview

### The Winning Formula (2024-2025 Literature)
```
Multimodal Input:
  ├── Structural (fundus/OCT image) → CNN backbone
  ├── Functional (VF series) → VF encoder → temporal model
  └── Clinical covariates → MLP embedder
                    ↓
            Attention Fusion
                    ↓
          Dual Regression Head
          (MD rate + VFI rate)
```

### Why Multimodal?
- Structural OCT alone: AUC 0.83–0.87
- Functional VF alone: AUC 0.79–0.85
- **Multimodal fusion: AUC 0.87–0.97** (3.5 years earlier detection)
- Especially important in myopes where structural-functional correlation is weak

---

## Component 1: CNN Backbones

### ConvNeXt-V2 (Recommended)
```python
import torchvision.models as tv
backbone = tv.convnext_small(weights=tv.ConvNeXt_Small_Weights.IMAGENET1K_V1)
# Remove classifier head, keep features + avgpool
# Output: 768-dim feature vector
```
- Best accuracy on medical imaging classification (2024)
- ImageNet pretrained → fine-tune on fundus/OCT
- Small variant: 50M params (practical for batch training)

### EfficientNet-B3 (Speed/Accuracy Tradeoff)
```python
backbone = tv.efficientnet_b3(weights=tv.EfficientNet_B3_Weights.IMAGENET1K_V1)
# Output: 1536-dim
```
- 12M params — faster training and inference
- Good when OCT images are high resolution (>512×512)

### When to Use Which
| Scenario | Recommendation |
|----------|----------------|
| Small dataset (<500 patients) | EfficientNet-B3 (less overfitting) |
| Large dataset (>1000 patients) | ConvNeXt-Small |
| High-res OCT volumes | ConvNeXt with 3D extension |
| Limited GPU memory | EfficientNet-B0 |

---

## Component 2: Temporal Models

### Bidirectional LSTM (Primary Choice)
```python
lstm = nn.LSTM(
    input_size=256,
    hidden_size=256,
    num_layers=2,
    bidirectional=True,
    batch_first=True
)
# Final output: 512-dim (256 × 2 directions)
```
- Best for VF time series (proven in multiple 2024 studies)
- Captures asymmetric temporal dynamics (early fast, later plateau)
- Handles variable-length sequences via packing

### Temporal Transformer (Alternative)
```python
encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
```
- Better for long sequences (>10 visits)
- More parameter-efficient than LSTM for long contexts
- Pre-norm (norm_first=True) for stable training

### Hybrid: Conv-LSTM
- Captures spatial structure of VF map AND temporal dynamics simultaneously
- The "Hybrid-VF-Net" architecture (2025) uses this approach
- More complex to implement

---

## Component 3: Attention Fusion

### Why Attention?
- In normal glaucoma: structure and function are correlated → roughly equal weights
- In myopic glaucoma: structure is distorted → model should learn to rely more on function
- Attention weights are interpretable (clinical insight)

### Implementation
```python
class AttentionFusion(nn.Module):
    def forward(self, structural, functional, clinical):
        # Project to common space
        s = self.proj_structural(structural)   # (B, D)
        f = self.proj_functional(functional)   # (B, D)
        c = self.proj_clinical(clinical)       # (B, D)

        # Compute soft attention weights
        concat = torch.cat([s, f, c], dim=1)
        weights = F.softmax(self.attn(concat), dim=1)  # (B, 3)

        # Weighted fusion
        fused = weights[:,0:1]*s + weights[:,1:2]*f + weights[:,2:3]*c
        return fused, weights  # weights are interpretable!
```

### Visualizing Weights (Clinical Insight)
```python
# After training, analyze: which modality does the model trust more?
# For fast progressors: do functional features dominate?
# For stable patients: does clinical data (IOP) dominate?
weights_by_group = {
    "fast_progressors": mean_weights[fast_mask],
    "slow_progressors": mean_weights[slow_mask],
}
```

---

## Training Strategy

### Transfer Learning (Critical for Small Datasets)
```python
# Phase 1: Freeze backbone, train fusion + head (5-10 epochs)
for param in model.backbone.parameters():
    param.requires_grad = False

# Phase 2: Unfreeze all, fine-tune with lower LR
for param in model.backbone.parameters():
    param.requires_grad = True
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
```

### Loss Function for Dual Regression
```python
# Weighted Huber Loss (robust to outliers, up-weights fast progressors)
def loss(md_pred, vfi_pred, md_true, vfi_true):
    # Up-weight fast progressors (clinically most important)
    w = torch.where(md_true < -1.0, 2.0, 1.0)
    md_loss = (F.huber_loss(md_pred, md_true, reduction='none') * w).mean()
    vfi_loss = (F.huber_loss(vfi_pred, vfi_true, reduction='none') * w).mean()
    return 1.0 * md_loss + 0.8 * vfi_loss
```

### Learning Rate Schedule
```python
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
# With 5-epoch warmup for stable initial training
```

### Data Augmentation (VF-specific)
```python
# For image branch: standard ophthalmology augmentations
transforms = [
    RandomHorizontalFlip(p=0.0),  # NEVER flip — laterality matters!
    RandomRotation(degrees=15),    # Small rotations only
    ColorJitter(brightness=0.2, contrast=0.2),
    GaussianBlur(kernel_size=3),
]

# For VF branch: temporal noise injection
def add_vf_noise(vf_series, sigma=0.5):
    """Add Gaussian noise to simulate test-retest variability."""
    noise = torch.randn_like(vf_series) * sigma
    return vf_series + noise
```

---

## Uncertainty Quantification

### Why It Matters for Clinical Use
- Clinicians need to know model confidence
- High uncertainty → request more frequent follow-up
- Low uncertainty + fast progression → urgent treatment change

### Gaussian NLL (Aleatoric Uncertainty)
```python
class UncertaintyHead(nn.Module):
    def forward(self, x):
        mean = self.mean_head(x)
        log_var = self.var_head(x)
        return mean, log_var

def nll_loss(pred, log_var, target):
    var = torch.exp(log_var).clamp(min=1e-6)
    return 0.5 * (log_var + (target - pred)**2 / var).mean()
```

### Monte Carlo Dropout (Epistemic Uncertainty)
```python
def mc_predict(model, x, n_samples=50):
    model.train()  # Keep dropout active
    preds = [model(x)['md_pred'] for _ in range(n_samples)]
    return torch.stack(preds).mean(0), torch.stack(preds).std(0)
```

---

## Interpretability

### SHAP for Clinical Insights
```python
import shap
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_sample)

# Which clinical features drive fast progression prediction?
shap.summary_plot(shap_values, feature_names=CLINICAL_FEATURE_NAMES)
```

### Attention Weight Analysis
```python
# Log modality weights during validation
# Expectation: in myopic glaucoma, functional (VF) should dominate
# when structural images are distorted by myopia artifacts
```

### GradCAM for Image Branch
```python
from pytorch_grad_cam import GradCAM
cam = GradCAM(model=backbone, target_layers=[backbone.features[-1]])
visualization = cam(input_tensor=fundus_image)
# Should highlight optic disc and RNFL region
```

---

## Baseline Comparisons

Always compare against:
1. **Linear regression on MD/VFI over time** (clinical standard)
2. **Random forest on VF slope features** (strong traditional ML baseline)
3. **LSTM on VF series only** (ablation: no structural)
4. **ConvNeXt on image only** (ablation: no temporal)
5. **Full multimodal model** (your contribution)

```python
results = {
    "OLS_baseline": compute_ols_baseline(test_data),
    "RF_features": compute_rf_baseline(test_data),
    "LSTM_only": evaluate(lstm_model, test_loader),
    "CNN_only": evaluate(cnn_model, test_loader),
    "GLAM_full": evaluate(glam_model, test_loader),
}
```

---

## References
- Bowd C et al. "Predicting Glaucomatous Progression: A Deep Learning Approach." JAMA Ophthalmol, 2024
- Luo J et al. "Harvard Glaucoma Detection and Progression Dataset." ICCV, 2023
- Hybrid-VF-Net: "A Hybrid Deep Learning Approach for Visual Field Test Forecasting." Ophthalmol Science, 2025
- FairDist: "Equity-Enhanced Glaucoma Progression with Knowledge Distillation." npj Digital Medicine, 2025
