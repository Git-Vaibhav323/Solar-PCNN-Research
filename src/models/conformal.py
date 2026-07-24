"""
Lightweight post-hoc uncertainty calibration (Phase 5).

Keeps Monte Carlo Dropout for epistemic sampling; calibrates interval width
so empirical coverage on a held-out calibration split matches a target PICP
(motivated by Phase 4 over-wide intervals).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ConformalCalibrator:
    """
    Scale MC predictive std: interval = mean ± z * (scale * std)

    `scale` is chosen on the calibration set so observed coverage ≈ target_picp.
    Also stores a split-conformal absolute residual quantile as a fallback width.
    """

    scale: float = 1.0
    abs_residual_quantile: float = 0.0
    target_picp: float = 0.95
    z: float = 1.96
    method: str = "std_scale"

    def fit_std_scale(
        self,
        y_true: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        target_picp: float = 0.95,
        z: float = 1.96,
        grid: np.ndarray | None = None,
    ) -> "ConformalCalibrator":
        y_true = y_true.reshape(-1)
        mean = mean.reshape(-1)
        std = np.maximum(std.reshape(-1), 1e-8)
        self.target_picp = float(target_picp)
        self.z = float(z)
        self.method = "std_scale"

        if grid is None:
            grid = np.concatenate(
                [
                    np.linspace(0.05, 1.0, 40),
                    np.linspace(1.05, 5.0, 40),
                    np.linspace(5.5, 20.0, 30),
                ]
            )

        best_s, best_gap = 1.0, 1e9
        for s in grid:
            lo = mean - z * s * std
            hi = mean + z * s * std
            cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
            gap = abs(cov - target_picp)
            # Prefer meeting coverage without massive overshoot when ties
            if gap < best_gap or (np.isclose(gap, best_gap) and s < best_s):
                best_gap, best_s = gap, float(s)

        self.scale = best_s
        abs_err = np.abs(y_true - mean)
        q = float(np.quantile(abs_err, target_picp))
        self.abs_residual_quantile = q
        return self

    def intervals(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = mean.reshape(-1)
        std = np.maximum(std.reshape(-1), 1e-8)
        half = self.z * self.scale * std
        return mean - half, mean + half

    def metrics(
        self,
        y_true: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> dict:
        y_true = y_true.reshape(-1)
        lo, hi = self.intervals(mean, std)
        picp = float(np.mean((y_true >= lo) & (y_true <= hi)))
        mpiw = float(np.mean(hi - lo))
        rng = float(np.max(y_true) - np.min(y_true))
        pinw = float(mpiw / rng) if rng > 0 else float("nan")
        return {
            "picp": picp,
            "mpiw": mpiw,
            "pinw": pinw,
            "scale": self.scale,
            "target_picp": self.target_picp,
            "abs_residual_quantile": self.abs_residual_quantile,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalCalibrator":
        return cls(**{k: d[k] for k in ("scale", "abs_residual_quantile", "target_picp", "z", "method") if k in d})
