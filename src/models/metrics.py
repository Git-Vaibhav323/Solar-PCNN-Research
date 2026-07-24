"""Regression and physics-consistency metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # MAPE on daytime / non-near-zero targets to avoid explode-at-night
    mask = np.abs(y_true) > 1e-3 * (np.max(np.abs(y_true)) + 1e-8)
    if mask.any():
        mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100.0)
    else:
        mape = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def physics_consistency_metrics(
    y_pred: np.ndarray,
    p_rated: float = 250.0,
    capacity_margin: float = 1.1,
) -> dict[str, float]:
    y_pred = y_pred.reshape(-1).astype(np.float64)
    upper = capacity_margin * p_rated
    n = max(len(y_pred), 1)
    neg_frac = float(np.mean(y_pred < 0.0))
    over_frac = float(np.mean(y_pred > upper))
    # Fraction of predictions inside physical bounds
    feasible = float(np.mean((y_pred >= 0.0) & (y_pred <= upper)))
    return {
        "negative_fraction": neg_frac,
        "capacity_violation_fraction": over_frac,
        "feasible_fraction": feasible,
        "physics_consistency_score": feasible,
    }


def summarize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_rated: float,
    capacity_margin: float,
) -> dict[str, Any]:
    out = {}
    out.update(regression_metrics(y_true, y_pred))
    out.update(physics_consistency_metrics(y_pred, p_rated, capacity_margin))
    return out
