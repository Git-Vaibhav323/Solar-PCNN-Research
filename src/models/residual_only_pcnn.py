"""
Phase-7 ablation: Residual-only PCNN (Model B).

y = clamp(P_physics + r) with alpha fixed to 1 (no adaptive gate).
Does not modify AR-PCNN / CD modules — separate ablation class only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ar_pcnn import ARPCNNOutput


class ResidualOnlyPCNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        p_rated: float = 250.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        self.p_rated = float(p_rated)
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        self.encoder = nn.Sequential(*layers)
        self.residual_head = nn.Sequential(
            nn.Linear(prev, prev // 2 if prev >= 2 else prev),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(prev // 2 if prev >= 2 else prev, 1),
        )

    def forward(self, x, p_physics_scaled, return_details: bool = False):
        h = self.encoder(x)
        residual = self.residual_head(h)
        physics_watts = p_physics_scaled * self.p_rated
        alpha = torch.ones_like(residual)
        y_hat = torch.clamp(physics_watts + residual, min=0.0, max=1.1 * self.p_rated)
        if return_details:
            return ARPCNNOutput(
                y_hat=y_hat, residual=residual, alpha=alpha, physics_watts=physics_watts
            )
        return y_hat
