"""
Phase 6 — Cross-domain AR-PCNN (multi-task).

Keeps the Phase-5 physics path identical:
    y_phys = clamp(α * P_physics + r)

Adds a lightweight external head for Renewables.ninja system power (kW),
sharing the same encoder so representations transfer across domains.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ar_pcnn import ARPCNN, ARPCNNOutput


@dataclass
class CrossDomainOutput:
    physics: ARPCNNOutput
    external_kw: torch.Tensor
    hidden: torch.Tensor


class ARPCNNCrossDomain(nn.Module):
    """
    Shared-encoder multi-task wrapper around AR-PCNN physics path.

    Does not modify the Phase-5 `ARPCNN` class; this is a separate Phase-6 module.
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        p_rated: float = 250.0,
        n_panels: int = 40,
        system_capacity_kw: float = 10.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        self.physics_branch = ARPCNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            p_rated=p_rated,
        )
        self.n_panels = int(n_panels)
        self.system_capacity_kw = float(system_capacity_kw)
        enc_out = hidden_dims[-1]
        self.external_head = nn.Sequential(
            nn.Linear(enc_out, enc_out // 2 if enc_out >= 2 else enc_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(enc_out // 2 if enc_out >= 2 else enc_out, 1),
        )

    @property
    def p_rated(self) -> float:
        return self.physics_branch.p_rated

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.physics_branch.encoder(x)

    def forward_physics(
        self,
        x: torch.Tensor,
        p_physics_scaled: torch.Tensor,
        return_details: bool = False,
    ):
        return self.physics_branch(x, p_physics_scaled, return_details=return_details)

    def forward_external_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        y = self.external_head(h)
        return torch.clamp(y, min=0.0, max=1.1 * self.system_capacity_kw)

    def forward(
        self,
        x: torch.Tensor,
        p_physics_scaled: torch.Tensor,
        return_details: bool = False,
    ):
        h = self.encode(x)
        residual = self.physics_branch.residual_head(h)
        alpha = self.physics_branch.gate_head(h)
        physics_watts = p_physics_scaled * self.physics_branch.p_rated
        y_hat = torch.clamp(
            alpha * physics_watts + residual,
            min=0.0,
            max=1.1 * self.physics_branch.p_rated,
        )
        phys = ARPCNNOutput(
            y_hat=y_hat,
            residual=residual,
            alpha=alpha,
            physics_watts=physics_watts,
        )
        external_kw = self.forward_external_from_hidden(h)
        if return_details:
            return CrossDomainOutput(physics=phys, external_kw=external_kw, hidden=h)
        return y_hat

    def load_physics_pretrained(self, state_dict: dict) -> None:
        """Warm-start physics branch from a Phase-5 AR-PCNN checkpoint state_dict."""
        self.physics_branch.load_state_dict(state_dict, strict=True)

    def freeze_encoder(self) -> None:
        for p in self.physics_branch.encoder.parameters():
            p.requires_grad = False

    def freeze_physics_heads(self) -> None:
        for module in (self.physics_branch.residual_head, self.physics_branch.gate_head):
            for p in module.parameters():
                p.requires_grad = False
