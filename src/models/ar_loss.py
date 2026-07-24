"""Loss for AR-PCNN: physics constraints + residual sparsity (Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .loss import PhysicsConstrainedLoss


@dataclass
class ARLossBreakdown:
    total: torch.Tensor
    mse: torch.Tensor
    capacity: torch.Tensor
    nonneg: torch.Tensor
    residual_reg: torch.Tensor
    mean_alpha: torch.Tensor


class AdaptiveResidualLoss(nn.Module):
    """
    L = L_physics_constrained(y_hat, y) + λ_r * mean(r^2)

    Residual L2 regularization (motivated by Phase 4 finding #1):
    prefer small corrections so the model does not re-learn full physics in r.
    """

    def __init__(
        self,
        p_rated: float = 250.0,
        capacity_margin: float = 1.1,
        lambda_capacity: float = 0.5,
        lambda_nonneg: float = 0.5,
        lambda_residual: float = 1e-4,
    ) -> None:
        super().__init__()
        self.base = PhysicsConstrainedLoss(
            p_rated=p_rated,
            capacity_margin=capacity_margin,
            lambda_capacity=lambda_capacity,
            lambda_nonneg=lambda_nonneg,
        )
        self.lambda_residual = float(lambda_residual)

    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        residual: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> ARLossBreakdown:
        br = self.base(y_hat, y_true)
        residual_reg = torch.mean(residual**2)
        total = br.total + self.lambda_residual * residual_reg
        mean_alpha = alpha.mean() if alpha is not None else torch.tensor(0.0, device=y_hat.device)
        return ARLossBreakdown(
            total=total,
            mse=br.mse,
            capacity=br.capacity,
            nonneg=br.nonneg,
            residual_reg=residual_reg,
            mean_alpha=mean_alpha,
        )
