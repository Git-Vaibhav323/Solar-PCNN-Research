"""Feature selection and light temporal feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required features: {missing}")
    keep = ["timestamp"] + feature_cols
    optional = ["Fill Flag", "Cloud Type", "Year", "Month", "Day", "Hour", "Minute"]
    keep += [c for c in optional if c in df.columns]
    return df[keep].copy()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical hour encodings used toward the paper's reported input size of 14
    (Table 6 lists 14; Eq. (6) lists 12 meteorological variables).
    """
    out = df.copy()
    hour = out["timestamp"].dt.hour.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return out


def model_feature_columns(feature_cols: list[str], include_temporal: bool) -> list[str]:
    cols = list(feature_cols)
    if include_temporal:
        cols += ["hour_sin", "hour_cos"]
    return cols
