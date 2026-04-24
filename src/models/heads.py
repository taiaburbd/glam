"""
Prediction heads for dual regression: MD (dB/year) and VFI (%/year).
"""

import torch
import torch.nn as nn

from .temporal import TemporalTransformer


class ProgressionHead(nn.Module):
    """
    Dual-output regression head for simultaneous MD and VFI prediction.
    - MD progression: dB/year (typically -0.5 to -3.0 for glaucoma)
    - VFI progression: %/year (typically -1.0 to -5.0 for glaucoma)
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()

        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads (separate to allow different learning dynamics)
        self.md_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),  # MD rate in dB/year
        )

        self.vfi_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),  # VFI rate in %/year
        )

        # Uncertainty estimation (optional: aleatoric uncertainty)
        self.md_log_var = nn.Linear(256, 1)
        self.vfi_log_var = nn.Linear(256, 1)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) — fused multimodal features
        Returns:
            dict with:
              md_pred:     (B, 1) — predicted MD progression rate (dB/year)
              vfi_pred:    (B, 1) — predicted VFI progression rate (%/year)
              md_log_var:  (B, 1) — log variance for MD (uncertainty)
              vfi_log_var: (B, 1) — log variance for VFI (uncertainty)
        """
        shared = self.shared(x)
        return {
            "md_pred": self.md_head(shared),
            "vfi_pred": self.vfi_head(shared),
            "md_log_var": self.md_log_var(shared),
            "vfi_log_var": self.vfi_log_var(shared),
        }


class GLAMModel(nn.Module):
    """
    Full GLAM model: ConvNeXt + BidirLSTM + AttentionFusion → ProgressionHead.
    End-to-end multimodal architecture for myopic glaucoma progression prediction.
    """

    def __init__(
        self,
        backbone: nn.Module,
        vf_encoder: nn.Module,
        temporal_model: nn.Module,
        fusion: nn.Module,
        clinical_embedder: nn.Module,
        prediction_head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.vf_encoder = vf_encoder
        self.temporal_model = temporal_model
        self.fusion = fusion
        self.clinical_embedder = clinical_embedder
        self.prediction_head = prediction_head

    def forward(
        self,
        images: torch.Tensor,       # (B, C, H, W) — baseline fundus/OCT
        vf_series: torch.Tensor,    # (B, T, n_points) — longitudinal VF data
        clinical: torch.Tensor,     # (B, n_clinical) — clinical covariates
        vf_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # 1. Structural features from image
        structural_feats = self.backbone(images)             # (B, struct_dim)

        # 2. Functional VF features with temporal modeling
        vf_encoded = self.vf_encoder(vf_series)              # (B, T, hidden_dim)
        if isinstance(self.temporal_model, TemporalTransformer):
            mask = None
            if vf_lengths is not None:
                B, T, _ = vf_encoded.shape
                pos = torch.arange(T, device=vf_encoded.device, dtype=vf_lengths.dtype).unsqueeze(0).expand(B, -1)
                mask = pos >= vf_lengths.unsqueeze(1)
            _, functional_feats = self.temporal_model(vf_encoded, mask=mask)
        else:
            _, functional_feats = self.temporal_model(vf_encoded, lengths=vf_lengths)

        # 3. Clinical embedding
        clinical_feats = self.clinical_embedder(clinical)    # (B, embed_dim)

        # 4. Multimodal fusion with attention
        fused, modality_weights = self.fusion(
            structural_feats, functional_feats, clinical_feats
        )

        # 5. Dual prediction
        preds = self.prediction_head(fused)
        preds["modality_weights"] = modality_weights

        return preds
