"""Quality checks and light cleaning for merged NSRDB data."""

from __future__ import annotations

from typing import Any

import pandas as pd


FEATURE_NONNEG = [
    "GHI",
    "DNI",
    "DHI",
    "Relative Humidity",
    "Pressure",
    "Surface Albedo",
    "Wind Speed",
    "Precipitable Water",
    "Solar Zenith Angle",
]


def quality_report(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "time_start": str(df["timestamp"].iloc[0]),
        "time_end": str(df["timestamp"].iloc[-1]),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "null_counts": {c: int(df[c].isna().sum()) for c in feature_cols if c in df.columns},
        "negative_counts": {},
        "fill_flag_counts": {},
        "hour_gaps": 0,
    }

    for col in FEATURE_NONNEG:
        if col in df.columns:
            report["negative_counts"][col] = int((df[col] < 0).sum())

    if "Fill Flag" in df.columns:
        report["fill_flag_counts"] = {
            str(k): int(v) for k, v in df["Fill Flag"].value_counts().sort_index().items()
        }

    diffs = df["timestamp"].diff().dropna()
    expected = pd.Timedelta(hours=1)
    report["hour_gaps"] = int((diffs != expected).sum())
    report["unique_hours"] = int(df["timestamp"].nunique())
    return report


def clean_nsrdb(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean merged NSRDB data while preserving paper fidelity.

    Policy (documented):
    - Drop duplicate timestamps (keep first)
    - Do not drop Fill Flag != 0 rows (paper does not filter them)
    - Clip physically impossible negatives on irradiance / humidity / etc. to 0
    - Leave remaining columns unchanged
    """
    out = df.copy()
    before = len(out)

    out = out.drop_duplicates(subset=["timestamp"], keep="first")
    n_dupes_removed = before - len(out)

    clipped: dict[str, int] = {}
    for col in FEATURE_NONNEG:
        if col not in out.columns:
            continue
        mask = out[col] < 0
        n = int(mask.sum())
        if n:
            out.loc[mask, col] = 0.0
            clipped[col] = n

    # Ensure feature columns are numeric
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    null_before = {c: int(out[c].isna().sum()) for c in feature_cols}
    if any(null_before.values()):
        out[feature_cols] = out[feature_cols].interpolate(method="linear", limit_direction="both")

    report = quality_report(out, feature_cols)
    report["duplicates_removed"] = n_dupes_removed
    report["negatives_clipped"] = clipped
    report["nulls_before_interpolate"] = null_before
    return out.reset_index(drop=True), report
