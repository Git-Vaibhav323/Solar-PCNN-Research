"""Physics-constrained loss (paper Eqs. 12–15)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LossBreakdown:
    total: torch.Tensor
    mse: torch.Tensor
    capacity: torch.Tensor
    nonneg: torch.Tensor


class PhysicsConstrainedLoss(nn.Module):
    """
    L_total = L_MSE + λ1 * L_capacity + λ2 * L_nonneg

    L_capacity = mean(max(0, y_hat - 1.1 * P_rated)^2)
    L_nonneg   = mean(max(0, -y_hat)^2)
    """

    def __init__(
        self,
        p_rated: float = 250.0,
        capacity_margin: float = 1.1,
        lambda_capacity: float = 0.5,
        lambda_nonneg: float = 0.5,
    ) -> None:
        super().__init__()
        self.p_rated = float(p_rated)
        self.capacity_margin = float(capacity_margin)
        self.lambda_capacity = float(lambda_capacity)
        self.lambda_nonneg = float(lambda_nonneg)
        self.mse = nn.MSELoss()

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> LossBreakdown:
        mse = self.mse(y_hat, y_true)
        upper = self.capacity_margin * self.p_rated
        capacity = torch.mean(torch.clamp(y_hat - upper, min=0.0) ** 2)
        nonneg = torch.mean(torch.clamp(-y_hat, min=0.0) ** 2)
        total = mse + self.lambda_capacity * capacity + self.lambda_nonneg * nonneg
        return LossBreakdown(total=total, mse=mse, capacity=capacity, nonneg=nonneg)
