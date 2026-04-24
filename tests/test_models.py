"""Model shape and forward-pass sanity tests."""

import pytest
import torch

from src.models.backbones import ConvNeXtBackbone, EfficientNetBackbone, VisualFieldEncoder
from src.models.temporal import BidirectionalLSTM, TemporalTransformer
from src.models.fusion import AttentionFusion, ClinicalEmbedder
from src.models.heads import ProgressionHead, GLAMModel


BATCH = 4
T = 6          # visits
N_VF = 54      # VF points
N_CLIN = 12    # clinical features


@pytest.fixture
def batch():
    return {
        "image": torch.randn(BATCH, 3, 224, 224),
        "vf_series": torch.randn(BATCH, T, N_VF),
        "clinical": torch.randn(BATCH, N_CLIN),
        "vf_lengths": torch.tensor([T] * BATCH),
    }


def test_convnext_output_shape(batch):
    backbone = ConvNeXtBackbone(pretrained=False)
    out = backbone(batch["image"])
    assert out.shape == (BATCH, 768)


def test_efficientnet_output_shape(batch):
    backbone = EfficientNetBackbone(pretrained=False)
    out = backbone(batch["image"])
    assert out.shape == (BATCH, 1536)


def test_vf_encoder_output_shape(batch):
    encoder = VisualFieldEncoder(n_points=N_VF, hidden_dim=256)
    out = encoder(batch["vf_series"])
    assert out.shape == (BATCH, T, 256)


def test_bilstm_output_shape():
    lstm = BidirectionalLSTM(input_dim=256, hidden_dim=256)
    x = torch.randn(BATCH, T, 256)
    output, last = lstm(x)
    assert output.shape == (BATCH, T, 512)
    assert last.shape == (BATCH, 512)


def test_transformer_output_shape():
    transformer = TemporalTransformer(input_dim=256, d_model=256, nhead=8, num_layers=2)
    x = torch.randn(BATCH, T, 256)
    output, pooled = transformer(x)
    assert output.shape == (BATCH, T, 256)
    assert pooled.shape == (BATCH, 256)


def test_attention_fusion_output():
    fusion = AttentionFusion(
        structural_dim=768, functional_dim=512, clinical_dim=128, fused_dim=256
    )
    s = torch.randn(BATCH, 768)
    f = torch.randn(BATCH, 512)
    c = torch.randn(BATCH, 128)
    fused, weights = fusion(s, f, c)
    assert fused.shape == (BATCH, 256)
    assert weights.shape == (BATCH, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(BATCH), atol=1e-5)


def test_progression_head_output():
    head = ProgressionHead(input_dim=512)
    x = torch.randn(BATCH, 512)
    preds = head(x)
    assert "md_pred" in preds
    assert "vfi_pred" in preds
    assert preds["md_pred"].shape == (BATCH, 1)
    assert preds["vfi_pred"].shape == (BATCH, 1)


def test_glam_full_forward(batch):
    """End-to-end forward pass through the full model."""
    backbone = ConvNeXtBackbone(pretrained=False)
    vf_encoder = VisualFieldEncoder(n_points=N_VF, hidden_dim=128)
    temporal = BidirectionalLSTM(input_dim=128, hidden_dim=128)
    clinical_emb = ClinicalEmbedder(n_clinical_features=N_CLIN, embed_dim=64)
    fusion = AttentionFusion(
        structural_dim=768, functional_dim=256, clinical_dim=64, fused_dim=256
    )
    head = ProgressionHead(input_dim=256)

    model = GLAMModel(backbone, vf_encoder, temporal, fusion, clinical_emb, head)
    preds = model(
        images=batch["image"],
        vf_series=batch["vf_series"],
        clinical=batch["clinical"],
        vf_lengths=batch["vf_lengths"],
    )

    assert preds["md_pred"].shape == (BATCH, 1)
    assert preds["vfi_pred"].shape == (BATCH, 1)
    assert "modality_weights" in preds
