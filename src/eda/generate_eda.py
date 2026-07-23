"""Exploratory analysis figures for Phase 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> Path:
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation heatmap of meteorological features (Fig. 2 analogue)")
    fig.tight_layout()
    path = _ensure_dir(out_dir) / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_diurnal_seasonal(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir = _ensure_dir(out_dir)
    paths: list[Path] = []
    work = df.copy()
    work["hour"] = work["timestamp"].dt.hour
    work["month"] = work["timestamp"].dt.month

    fig, ax = plt.subplots(figsize=(10, 4))
    diurnal = work.groupby("hour")[["GHI", "target_power"]].mean()
    ax.plot(diurnal.index, diurnal["GHI"], label="GHI (W/m²)")
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("GHI")
    ax2 = ax.twinx()
    ax2.plot(diurnal.index, diurnal["target_power"], color="tab:orange", label="P_bounded (W)")
    ax2.set_ylabel("Power (W)")
    ax.set_title("Mean diurnal profile — GHI and physics target power")
    fig.tight_layout()
    p1 = out_dir / "diurnal_profile.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    fig, ax = plt.subplots(figsize=(10, 4))
    monthly = work.groupby("month")[["GHI", "Temperature", "target_power"]].mean()
    ax.plot(monthly.index, monthly["GHI"], marker="o", label="GHI")
    ax.set_xlabel("Month")
    ax.set_ylabel("GHI (W/m²)")
    ax2 = ax.twinx()
    ax2.plot(monthly.index, monthly["Temperature"], marker="s", color="tab:red", label="Temp")
    ax2.set_ylabel("Temperature (°C)")
    ax.set_title("Seasonal (monthly) means — GHI and temperature")
    fig.tight_layout()
    p2 = out_dir / "seasonal_monthly.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(work["timestamp"], work["GHI"], lw=0.3, alpha=0.7)
    ax.set_title("GHI time series (2016–2020)")
    ax.set_ylabel("GHI (W/m²)")
    fig.tight_layout()
    p3 = out_dir / "ghi_timeseries.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)

    return paths


def plot_target_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["target_power"], bins=60, color="steelblue", edgecolor="none")
    axes[0].set_title("P_bounded distribution")
    axes[0].set_xlabel("Power (W)")
    daytime = df[df["GHI"] > 0]
    axes[1].scatter(daytime["GHI"], daytime["target_power"], s=2, alpha=0.15)
    axes[1].set_xlabel("GHI (W/m²)")
    axes[1].set_ylabel("P_bounded (W)")
    axes[1].set_title("Physics target vs GHI (daytime)")
    fig.tight_layout()
    path = _ensure_dir(out_dir) / "target_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_eda(df: pd.DataFrame, feature_cols: list[str], figures_dir: Path) -> dict:
    paths = {
        "correlation": str(plot_correlation_heatmap(df, feature_cols, figures_dir)),
        "target": str(plot_target_distribution(df, figures_dir)),
    }
    diurnal_paths = plot_diurnal_seasonal(df, figures_dir)
    paths["diurnal"] = [str(p) for p in diurnal_paths]

    corr = df[feature_cols].corr()
    summary = {
        "ghi_dni_corr": float(corr.loc["GHI", "DNI"]),
        "ghi_dhi_corr": float(corr.loc["GHI", "DHI"]),
        "ghi_rh_corr": float(corr.loc["GHI", "Relative Humidity"]),
        "ghi_temp_corr": float(corr.loc["GHI", "Temperature"]),
        "target_mean_w": float(df["target_power"].mean()),
        "target_max_w": float(df["target_power"].max()),
        "figures": paths,
    }
    return summary
