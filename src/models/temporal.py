"""
Temporal models for longitudinal visual field data.
Handles variable-length sequences of patient visits.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for capturing temporal progression patterns.
    Proven effective for VF time-series (spatiotemporal pattern extraction).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.out_dim = hidden_dim * 2

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim) — padded sequence of visit features
            lengths: (B,) actual sequence lengths per patient (for packing)
        Returns:
            output: (B, T, hidden_dim*2) — all hidden states
            last:   (B, hidden_dim*2) — final hidden state (for prediction)
        """
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, (hn, _) = self.lstm(packed)
            output, _ = pad_packed_sequence(output_packed, batch_first=True)
        else:
            output, (hn, _) = self.lstm(x)

        output = self.layer_norm(output)
        # Concatenate forward/backward final hidden states
        last = torch.cat([hn[-2], hn[-1]], dim=1)
        return output, last


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal visit sequences.
    Better at capturing long-range dependencies across many visits.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_visits: int = 20,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Embedding(max_visits, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim)
            mask: (B, T) — True for padding positions
        Returns:
            output: (B, T, d_model)
            pooled: (B, d_model) — mean-pooled representation
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        x = self.input_proj(x) + self.pos_encoding(positions)
        output = self.transformer(x, src_key_padding_mask=mask)

        if mask is not None:
            # Mean pool over non-padded positions
            valid_mask = ~mask  # (B, T)
            pooled = (output * valid_mask.unsqueeze(-1)).sum(1) / valid_mask.sum(1, keepdim=True)
        else:
            pooled = output.mean(1)

        return output, pooled
