"""Baseline Physics-Constrained Neural Network (paper Section 4.2)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PCNN(nn.Module):
    """
    Dual-branch PCNN:
      - Data branch: FFNN with BatchNorm + ReLU + Dropout (Table 6: 128→64→32)
      - Physics prior: scaled theoretical power concatenated into a fusion layer
      - Output heads (Eqs. 9–11): P_hat = P_direct × σ(f_physics)
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dims: list[int] | None = None,
        fuse_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

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

        self.fuse = nn.Sequential(
            nn.Linear(prev + 1, fuse_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head_direct = nn.Linear(fuse_dim, 1)
        self.head_gate = nn.Linear(fuse_dim, 1)

    def forward(self, x: torch.Tensor, p_physics_scaled: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.fuse(torch.cat([h, p_physics_scaled], dim=1))
        p_direct = self.head_direct(z)
        f_physics = torch.sigmoid(self.head_gate(z))
        return p_direct * f_physics
