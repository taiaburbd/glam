"""
Attention-based multimodal fusion.
Fuses structural (fundus/OCT), functional (visual field), and clinical features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Cross-attention fusion of multimodal features.
    Learns dynamic weights across modalities — critical for myopic glaucoma
    where structural/functional correlation is often decoupled.
    """

    def __init__(
        self,
        structural_dim: int,   # ConvNeXt/EfficientNet output
        functional_dim: int,   # LSTM/Transformer output for VF
        clinical_dim: int,     # Clinical covariates embedding
        fused_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Project each modality to common space
        self.proj_structural = nn.Linear(structural_dim, fused_dim)
        self.proj_functional = nn.Linear(functional_dim, fused_dim)
        self.proj_clinical = nn.Linear(clinical_dim, fused_dim)

        # Learned attention weights per modality
        self.modality_attention = nn.Sequential(
            nn.Linear(fused_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),   # 3 modalities
        )

        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
        )
        self.out_dim = fused_dim

    def forward(
        self,
        structural: torch.Tensor,   # (B, structural_dim)
        functional: torch.Tensor,   # (B, functional_dim)
        clinical: torch.Tensor,     # (B, clinical_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused:   (B, fused_dim) — final fused representation
            weights: (B, 3) — modality attention weights (for interpretability)
        """
        s = self.proj_structural(structural)   # (B, fused_dim)
        f = self.proj_functional(functional)   # (B, fused_dim)
        c = self.proj_clinical(clinical)       # (B, fused_dim)

        # Compute attention weights from concatenated projections
        concat = torch.cat([s, f, c], dim=1)  # (B, fused_dim*3)
        weights = F.softmax(self.modality_attention(concat), dim=1)  # (B, 3)

        # Weighted sum of modalities
        fused = (
            weights[:, 0:1] * s
            + weights[:, 1:2] * f
            + weights[:, 2:3] * c
        )
        fused = self.fusion_mlp(fused)
        return fused, weights


class ClinicalEmbedder(nn.Module):
    """
    Embeds clinical covariates into a dense vector.
    Handles: age, IOP, axial length, CCT, RNFL thickness, etc.
    """

    def __init__(self, n_clinical_features: int, embed_dim: int = 128):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(n_clinical_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)  # (B, embed_dim)
