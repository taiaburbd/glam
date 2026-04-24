"""
Loss functions for dual MD/VFI progression rate regression.
Handles uncertainty estimation and clinical weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedDualRegressionLoss(nn.Module):
    """
    Weighted loss for simultaneous MD (dB/year) and VFI (%/year) regression.
    Up-weights fast progressors (clinically more important to detect).
    """

    def __init__(
        self,
        md_weight: float = 1.0,
        vfi_weight: float = 0.8,
        fast_progressor_threshold_md: float = -1.0,  # dB/year
        fast_progressor_boost: float = 2.0,
    ):
        super().__init__()
        self.md_weight = md_weight
        self.vfi_weight = vfi_weight
        self.fp_threshold = fast_progressor_threshold_md
        self.fp_boost = fast_progressor_boost

    def forward(
        self,
        md_pred: torch.Tensor,     # (B, 1)
        vfi_pred: torch.Tensor,    # (B, 1)
        md_target: torch.Tensor,   # (B,)
        vfi_target: torch.Tensor,  # (B,)
    ) -> dict[str, torch.Tensor]:
        md_target = md_target.unsqueeze(1)
        vfi_target = vfi_target.unsqueeze(1)

        # Sample weights: boost fast progressors
        sample_weights = torch.where(
            md_target < self.fp_threshold,
            torch.full_like(md_target, self.fp_boost),
            torch.ones_like(md_target),
        )

        md_loss = (F.huber_loss(md_pred, md_target, reduction="none") * sample_weights).mean()
        vfi_loss = (F.huber_loss(vfi_pred, vfi_target, reduction="none") * sample_weights).mean()

        total_loss = self.md_weight * md_loss + self.vfi_weight * vfi_loss
        return {
            "loss": total_loss,
            "md_loss": md_loss,
            "vfi_loss": vfi_loss,
        }


class NegativeLogLikelihoodLoss(nn.Module):
    """
    Gaussian NLL for uncertainty-aware regression.
    Model predicts both mean (rate) and log_variance (confidence).
    """

    def __init__(self, md_weight: float = 1.0, vfi_weight: float = 0.8):
        super().__init__()
        self.md_weight = md_weight
        self.vfi_weight = vfi_weight

    def _gaussian_nll(
        self,
        pred: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # NLL = 0.5 * (log_var + (target - pred)^2 / var)
        var = torch.exp(log_var).clamp(min=1e-6)
        nll = 0.5 * (log_var + (target - pred).pow(2) / var)
        return nll.mean()

    def forward(
        self,
        md_pred: torch.Tensor,
        md_log_var: torch.Tensor,
        vfi_pred: torch.Tensor,
        vfi_log_var: torch.Tensor,
        md_target: torch.Tensor,
        vfi_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t = md_target.unsqueeze(1)
        v = vfi_target.unsqueeze(1)

        md_nll = self._gaussian_nll(md_pred, md_log_var, t)
        vfi_nll = self._gaussian_nll(vfi_pred, vfi_log_var, v)
        total = self.md_weight * md_nll + self.vfi_weight * vfi_nll

        return {"loss": total, "md_loss": md_nll, "vfi_loss": vfi_nll}
