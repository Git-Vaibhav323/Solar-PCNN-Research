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

    # MAPE: mask near-zero targets — still can explode in twilight / low power.
    mask = np.abs(y_true) > 1e-3 * (np.max(np.abs(y_true)) + 1e-8)
    if mask.any():
        mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100.0)
    else:
        mape = float("nan")

    # sMAPE: symmetric; bounded in [0, 200]% — stable under near-zero generation.
    denom = np.abs(y_true) + np.abs(y_pred)
    smape_mask = denom > 1e-12
    if smape_mask.any():
        smape = float(np.mean(2.0 * np.abs(err[smape_mask]) / denom[smape_mask]) * 100.0)
    else:
        smape = float("nan")

    # nRMSE: RMSE normalized by mean observed generation (×100 → %).
    mean_abs = float(np.mean(np.abs(y_true)))
    nrmse = float(rmse / mean_abs * 100.0) if mean_abs > 1e-12 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "smape": smape,
        "nrmse": nrmse,
    }


def physics_consistency_metrics(
    y_pred: np.ndarray,
    p_rated: float = 250.0,
    capacity_margin: float = 1.1,
) -> dict[str, float]:
    y_pred = y_pred.reshape(-1).astype(np.float64)
    upper = capacity_margin * p_rated
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
