"""Multi-task loss for Phase-6 cross-domain AR-PCNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ar_loss import AdaptiveResidualLoss


@dataclass
class CDLossBreakdown:
    total: torch.Tensor
    physics_total: torch.Tensor
    physics_mse: torch.Tensor
    external_mse: torch.Tensor
    residual_reg: torch.Tensor
    mean_alpha: torch.Tensor
    n_external: torch.Tensor


class CrossDomainLoss(nn.Module):
    """
    L = L_physics + λ_ext * masked_MSE(external_kw, ninja_kw)

    External term is computed only on samples with has_external=1.
    """

    def __init__(
        self,
        p_rated: float = 250.0,
        capacity_margin: float = 1.1,
        lambda_capacity: float = 0.5,
        lambda_nonneg: float = 0.5,
        lambda_residual: float = 1e-4,
        lambda_external: float = 1.0,
    ) -> None:
        super().__init__()
        self.physics_loss = AdaptiveResidualLoss(
            p_rated=p_rated,
            capacity_margin=capacity_margin,
            lambda_capacity=lambda_capacity,
            lambda_nonneg=lambda_nonneg,
            lambda_residual=lambda_residual,
        )
        self.lambda_external = float(lambda_external)

    def forward(
        self,
        y_phys: torch.Tensor,
        y_true_phys: torch.Tensor,
        residual: torch.Tensor,
        alpha: torch.Tensor,
        external_kw: torch.Tensor,
        ninja_kw: torch.Tensor,
        has_external: torch.Tensor,
    ) -> CDLossBreakdown:
        phys = self.physics_loss(y_phys, y_true_phys, residual, alpha)
        mask = has_external.reshape(-1) > 0.5
        n_ext = mask.sum()
        if n_ext.item() > 0:
            ext_mse = torch.mean((external_kw.reshape(-1)[mask] - ninja_kw.reshape(-1)[mask]) ** 2)
        else:
            ext_mse = torch.zeros((), device=y_phys.device)
        total = phys.total + self.lambda_external * ext_mse
        return CDLossBreakdown(
            total=total,
            physics_total=phys.total,
            physics_mse=phys.mse,
            external_mse=ext_mse,
            residual_reg=phys.residual_reg,
            mean_alpha=phys.mean_alpha,
            n_external=n_ext.float(),
        )
