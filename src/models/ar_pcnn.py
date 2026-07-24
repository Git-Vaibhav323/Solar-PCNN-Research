"""
Adaptive Residual Physics-Constrained Neural Network (AR-PCNN) — Phase 5.

Phase-4 motivations addressed:
1. Residual head  → learn only what analytical physics cannot explain
2. Adaptive gate α → dynamically trust physics vs correction under shifting conditions
3. (Calibration is applied post-hoc in src/models/conformal.py; MC Dropout kept)

Final prediction (Watts):
    y_hat = α * P_physics + r
where α ∈ (0, 1) from a sigmoid gate and r is the residual correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ARPCNNOutput:
    y_hat: torch.Tensor
    residual: torch.Tensor
    alpha: torch.Tensor
    physics_watts: torch.Tensor


class ARPCNN(nn.Module):
    """
    Separate Phase-5 model. Does not modify the Phase-3 PCNN class.

    Inputs
    ------
    x : normalized meteorological features [B, input_dim]
    p_physics_scaled : P_theoretical / P_rated  [B, 1]  (same dataset contract as baseline)
    """

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

        # Residual correction head (Watts)
        self.residual_head = nn.Sequential(
            nn.Linear(prev, prev // 2 if prev >= 2 else prev),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(prev // 2 if prev >= 2 else prev, 1),
        )
        # Adaptive physics gate α ∈ (0, 1)
        self.gate_head = nn.Sequential(
            nn.Linear(prev, prev // 2 if prev >= 2 else prev),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(prev // 2 if prev >= 2 else prev, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        p_physics_scaled: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | ARPCNNOutput:
        h = self.encoder(x)
        residual = self.residual_head(h)
        alpha = self.gate_head(h)
        physics_watts = p_physics_scaled * self.p_rated
        y_hat = alpha * physics_watts + residual
        # Soft physical bounds (Phase-4 feasibility requirement)
        y_hat = torch.clamp(y_hat, min=0.0, max=1.1 * self.p_rated)
        if return_details:
            return ARPCNNOutput(
                y_hat=y_hat,
                residual=residual,
                alpha=alpha,
                physics_watts=physics_watts,
            )
        return y_hat
