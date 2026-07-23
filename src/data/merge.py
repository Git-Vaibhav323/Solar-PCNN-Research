"""Merge NSRDB yearly CSVs into a single chronological frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NSRDB_SKIPROWS = 2


def _year_path(nsrdb_dir: Path, year: int, location_id: int, lat: float, lon: float) -> Path:
    pattern = f"{location_id}_{lat:.2f}_{lon:.2f}_{year}.csv"
    path = nsrdb_dir / pattern
    if not path.exists():
        matches = sorted(nsrdb_dir.glob(f"*_{year}.csv"))
        if not matches:
            raise FileNotFoundError(f"No NSRDB file for year {year} in {nsrdb_dir}")
        return matches[0]
    return path


def load_nsrdb_year(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=NSRDB_SKIPROWS)
    if df.empty:
        raise ValueError(f"Empty NSRDB file: {path}")
    return df


def build_timestamp(df: pd.DataFrame, tz_offset_hours: int = 5) -> pd.Series:
    """Build timezone-aware timestamps from NSRDB Year/Month/Day/Hour/Minute."""
    ts = pd.to_datetime(
        {
            "year": df["Year"].astype(int),
            "month": df["Month"].astype(int),
            "day": df["Day"].astype(int),
            "hour": df["Hour"].astype(int),
            "minute": df["Minute"].astype(int),
        }
    )
    # Local solar-site time (NSRDB Local Time Zone = +5 for this download)
    return ts.dt.tz_localize(f"Etc/GMT-{tz_offset_hours}")


def merge_nsrdb_years(
    nsrdb_dir: Path,
    years: list[int],
    location_id: int,
    latitude: float,
    longitude: float,
    tz_offset_hours: int = 5,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        path = _year_path(nsrdb_dir, year, location_id, latitude, longitude)
        part = load_nsrdb_year(path)
        part["source_file"] = path.name
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = build_timestamp(df, tz_offset_hours=tz_offset_hours)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
