"""Photovoltaic physics helpers (paper Section 4.1)."""

from __future__ import annotations

import pandas as pd


def irradiance_efficiency(ghi: pd.Series, i_stc: float = 1000.0) -> pd.Series:
    """Eq. (2): η_irr = I_GHI / I_STC."""
    return ghi / i_stc


def temperature_efficiency(
    temperature: pd.Series,
    gamma: float = -0.004,
    t_stc: float = 25.0,
) -> pd.Series:
    """Eq. (3): η_temp = 1 + γ (T − T_STC)."""
    return 1.0 + gamma * (temperature - t_stc)


def theoretical_power(
    ghi: pd.Series,
    temperature: pd.Series,
    p_rated: float = 250.0,
    i_stc: float = 1000.0,
    gamma: float = -0.004,
    t_stc: float = 25.0,
) -> pd.Series:
    """Eq. (4): P_theoretical = P_rated × η_irr × η_temp."""
    return p_rated * irradiance_efficiency(ghi, i_stc) * temperature_efficiency(
        temperature, gamma, t_stc
    )


def bounded_power(
    p_theoretical: pd.Series,
    p_rated: float = 250.0,
    capacity_margin: float = 1.1,
) -> pd.Series:
    """Eq. (5): P_bounded = max(0, min(P_theoretical, 1.1 × P_rated))."""
    upper = capacity_margin * p_rated
    return p_theoretical.clip(lower=0.0, upper=upper)


def add_physics_targets(df: pd.DataFrame, pv: dict) -> pd.DataFrame:
    out = df.copy()
    out["eta_irr"] = irradiance_efficiency(out["GHI"], pv["i_stc"])
    out["eta_temp"] = temperature_efficiency(out["Temperature"], pv["gamma"], pv["t_stc"])
    out["P_theoretical"] = theoretical_power(
        out["GHI"],
        out["Temperature"],
        p_rated=pv["p_rated_w"],
        i_stc=pv["i_stc"],
        gamma=pv["gamma"],
        t_stc=pv["t_stc"],
    )
    out["P_bounded"] = bounded_power(
        out["P_theoretical"],
        p_rated=pv["p_rated_w"],
        capacity_margin=pv["capacity_margin"],
    )
    # Training target used by the paper loss (Eq. 13 references P_theoretical)
    out["target_power"] = out["P_bounded"]
    return out
