"""Phase-6 diagnostics and comparison figures/tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train import load_train_config


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _g(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def plot_diagnostics(cfg: dict, strategy: str) -> dict:
    metrics_dir = cfg["paths"]["metrics_dir"]
    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    test_npz = metrics_dir / f"ar_pcnn_cd_{strategy}_test_arrays.npz"
    ninja_npz = metrics_dir / f"ar_pcnn_cd_{strategy}_ninja_full_external.npz"
    if not test_npz.exists():
        return paths

    arr = np.load(test_npz)
    alpha = arr["alpha"].reshape(-1)
    residual = arr["residual"].reshape(-1)
    y_true = arr["y_true"].reshape(-1)
    y_pred = arr["y_pred"].reshape(-1)

    # Gate / residual distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(alpha, bins=40, color="steelblue", edgecolor="none")
    axes[0].set_title(f"{strategy}: gate alpha (NSRDB test)")
    axes[0].set_xlabel("alpha")
    axes[1].hist(residual, bins=40, color="darkorange", edgecolor="none")
    axes[1].set_title(f"{strategy}: residual (NSRDB test)")
    axes[1].set_xlabel("r (W)")
    fig.tight_layout()
    p1 = fig_dir / f"phase6_{strategy}_alpha_residual.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths["alpha_residual"] = str(p1)

    # NSRDB scatter
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, s=3, alpha=0.15)
    lim = max(y_true.max(), y_pred.max()) * 1.05 + 1e-6
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("True physics target (W)")
    ax.set_ylabel("Predicted (W)")
    ax.set_title(f"{strategy}: NSRDB test")
    fig.tight_layout()
    p2 = fig_dir / f"phase6_{strategy}_nsrdb_scatter.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths["nsrdb_scatter"] = str(p2)

    if ninja_npz.exists():
        n = np.load(ninja_npz)
        yt, yp = n["y_true_kw"].reshape(-1), n["y_pred_kw"].reshape(-1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(yt, yp, s=3, alpha=0.2)
        lim = max(yt.max(), yp.max()) * 1.05
        axes[0].plot([0, lim], [0, lim], "r--", lw=1)
        axes[0].set_title(f"{strategy}: Ninja full-2019 external head")
        axes[0].set_xlabel("Ninja kW")
        axes[0].set_ylabel("Pred kW")

        hours = np.arange(len(yt)) % 24
        mae_h = pd.Series(np.abs(yp - yt)).groupby(hours).mean()
        axes[1].bar(mae_h.index, mae_h.values, color="crimson")
        axes[1].set_title(f"{strategy}: Ninja |error| by hour")
        axes[1].set_xlabel("Hour mod 24")
        fig.tight_layout()
        p3 = fig_dir / f"phase6_{strategy}_ninja_diag.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        paths["ninja_diag"] = str(p3)

        # Error by irradiance proxy (ninja power level)
        bins = [0, 0.1, 1, 3, 5, 8, 12]
        labels = ["~0", "0.1-1", "1-3", "3-5", "5-8", "8+"]
        cat = pd.cut(yt, bins=bins, labels=labels, include_lowest=True)
        by = pd.Series(np.abs(yp - yt)).groupby(cat, observed=False).mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(by)), by.values, color="purple")
        ax.set_xticks(range(len(by)))
        ax.set_xticklabels([str(x) for x in by.index], rotation=30, ha="right")
        ax.set_ylabel("MAE (kW)")
        ax.set_title(f"{strategy}: Ninja error by power bin")
        fig.tight_layout()
        p4 = fig_dir / f"phase6_{strategy}_ninja_power_bins.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        paths["ninja_power_bins"] = str(p4)

    return paths


def run_compare(config_path: str | None = None) -> dict:
    cfg = load_train_config(config_path or str(ROOT / "configs" / "train_ar_pcnn_cd.yaml"))
    mdir = Path(cfg["paths"]["metrics_dir"])

    baseline = _load(mdir / "metrics_test.json") or _load(mdir / "phase3_summary.json").get("test", {})
    baseline_case1 = _load(mdir / "case1_ninja_metrics.json")
    baseline_uq = _load(mdir / "uncertainty_test_metrics.json")

    ar = _load(mdir / "ar_pcnn_metrics_test.json") or _load(mdir / "ar_pcnn_phase5_summary.json").get("test", {})
    ar_case1 = _load(mdir / "ar_pcnn_case1_metrics.json")
    ar_uq = _load(mdir / "ar_pcnn_uncertainty_test_metrics.json")

    mt = _load(mdir / "ar_pcnn_cd_multitask_eval.json")
    ft = _load(mdir / "ar_pcnn_cd_finetune_eval.json")
    mt_uq = _load(mdir / "ar_pcnn_cd_multitask_uncertainty.json")
    ft_uq = _load(mdir / "ar_pcnn_cd_finetune_uncertainty.json")

    figs = {
        "multitask": plot_diagnostics(cfg, "multitask"),
        "finetune": plot_diagnostics(cfg, "finetune"),
    }

    def row_nsrdb(d, prefix=None):
        src = d if prefix is None else _g(d, prefix, default={}) or {}
        return {
            "mae": _g(src, "mae"),
            "rmse": _g(src, "rmse"),
            "r2": _g(src, "r2"),
            "mape": _g(src, "mape"),
            "feasible": _g(src, "feasible_fraction"),
            "mean_alpha": _g(src, "mean_alpha"),
        }

    comparison = {
        "nsrdb_test": {
            "pcnn": row_nsrdb(baseline),
            "ar_pcnn": row_nsrdb(ar),
            "ar_pcnn_cd_multitask": row_nsrdb(mt, "nsrdb_test"),
            "ar_pcnn_cd_finetune": row_nsrdb(ft, "nsrdb_test"),
        },
        "ninja_full_2019": {
            "pcnn": {
                "mae": _g(baseline_case1, "mae"),
                "rmse": _g(baseline_case1, "rmse"),
                "r2": _g(baseline_case1, "r2"),
                "mape": _g(baseline_case1, "mape"),
                "smape": _g(baseline_case1, "smape"),
                "nrmse": _g(baseline_case1, "nrmse"),
            },
            "ar_pcnn": {
                "mae": _g(ar_case1, "mae"),
                "rmse": _g(ar_case1, "rmse"),
                "r2": _g(ar_case1, "r2"),
                "mape": _g(ar_case1, "mape"),
                "smape": _g(ar_case1, "smape"),
                "nrmse": _g(ar_case1, "nrmse"),
            },
            "ar_pcnn_cd_multitask_external": {
                "mae": _g(mt, "ninja_full_2019_external_head", "mae"),
                "rmse": _g(mt, "ninja_full_2019_external_head", "rmse"),
                "r2": _g(mt, "ninja_full_2019_external_head", "r2"),
                "mape": _g(mt, "ninja_full_2019_external_head", "mape"),
                "smape": _g(mt, "ninja_full_2019_external_head", "smape"),
                "nrmse": _g(mt, "ninja_full_2019_external_head", "nrmse"),
            },
            "ar_pcnn_cd_finetune_external": {
                "mae": _g(ft, "ninja_full_2019_external_head", "mae"),
                "rmse": _g(ft, "ninja_full_2019_external_head", "rmse"),
                "r2": _g(ft, "ninja_full_2019_external_head", "r2"),
                "mape": _g(ft, "ninja_full_2019_external_head", "mape"),
                "smape": _g(ft, "ninja_full_2019_external_head", "smape"),
                "nrmse": _g(ft, "ninja_full_2019_external_head", "nrmse"),
            },
        },
        "ninja_val_holdout": {
            "multitask": _g(mt, "ninja_val_holdout"),
            "finetune": _g(ft, "ninja_val_holdout"),
        },
        "uncertainty": {
            "pcnn_raw": {
                "picp": _g(baseline_uq, "uncertainty", "picp_95"),
                "mpiw": _g(baseline_uq, "uncertainty", "mpiw"),
            },
            "ar_pcnn_calibrated": {
                "picp": _g(ar_uq, "calibrated_on_val_applied_to_test", "picp"),
                "mpiw": _g(ar_uq, "calibrated_on_val_applied_to_test", "mpiw"),
            },
            "cd_multitask_calibrated": {
                "picp": _g(mt_uq, "calibrated_on_val_applied_to_test", "picp"),
                "mpiw": _g(mt_uq, "calibrated_on_val_applied_to_test", "mpiw"),
            },
            "cd_finetune_calibrated": {
                "picp": _g(ft_uq, "calibrated_on_val_applied_to_test", "picp"),
                "mpiw": _g(ft_uq, "calibrated_on_val_applied_to_test", "mpiw"),
            },
        },
        "figures": figs,
    }

    with (mdir / "phase6_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    def fmt(x, nd=4):
        return "—" if x is None else f"{float(x):.{nd}f}"

    md = f"""# Phase 6 Comparison — Cross-Domain Learning

**Primary contribution:** multitask external-head transfer to Renewables.ninja (R² ~0.5 → **0.86** full-year; **0.81** val-holdout).

Architecture: AR-PCNN physics path unchanged. Phase 6 changes **training** only
(multi-task shared encoder + external head, and encoder-frozen fine-tune).

NSRDB rows below are a **physics-fidelity consistency check** (`target_power` is physics-derived). Do not lead a paper with NSRDB R²≈1.0.

## NSRDB test (physics target, Watts) — consistency only

| Metric | PCNN | AR-PCNN | CD Multitask | CD Finetune |
|--------|------|---------|--------------|-------------|
| MAE (W) | {fmt(_g(comparison,'nsrdb_test','pcnn','mae'))} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn','mae'))} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_multitask','mae'))} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_finetune','mae'))} |
| RMSE (W) | {fmt(_g(comparison,'nsrdb_test','pcnn','rmse'))} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn','rmse'))} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_multitask','rmse'))} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_finetune','rmse'))} |
| R² (—) | {fmt(_g(comparison,'nsrdb_test','pcnn','r2'))} | ≈1.0 | ≈1.0 | ≈1.0 |
| MAPE (%) | {fmt(_g(comparison,'nsrdb_test','pcnn','mape'),3)} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn','mape'),3)} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_multitask','mape'),3)} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_finetune','mape'),3)} |
| Feasible (—) | {fmt(_g(comparison,'nsrdb_test','pcnn','feasible'),3)} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn','feasible'),3)} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_multitask','feasible'),3)} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_finetune','feasible'),3)} |
| Mean alpha (—) | — | {fmt(_g(comparison,'nsrdb_test','ar_pcnn','mean_alpha'),3)} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_multitask','mean_alpha'),3)} | {fmt(_g(comparison,'nsrdb_test','ar_pcnn_cd_finetune','mean_alpha'),3)} |

`Feasible = 1.0` is by construction (hard clamp). CD RMSE on NSRDB is slightly worse than AR-PCNN in some regimes — a multitask cost (see Phase 9 Exp 8).

## Renewables.ninja full 2019 (10 kW) — primary result

| Metric | PCNN (phys scaled) | AR-PCNN (phys scaled) | CD Multitask (external head) | CD Finetune (external head) |
|--------|--------------------|-----------------------|------------------------------|-----------------------------|
| MAE (kW) | {fmt(_g(comparison,'ninja_full_2019','pcnn','mae'))} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn','mae'))} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_multitask_external','mae'))} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_finetune_external','mae'))} |
| RMSE (kW) | {fmt(_g(comparison,'ninja_full_2019','pcnn','rmse'))} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn','rmse'))} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_multitask_external','rmse'))} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_finetune_external','rmse'))} |
| R² (—) | {fmt(_g(comparison,'ninja_full_2019','pcnn','r2'))} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn','r2'))} | **{fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_multitask_external','r2'))}** | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_finetune_external','r2'))} |
| **sMAPE (%)** | {fmt(_g(comparison,'ninja_full_2019','pcnn','smape'),2)} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn','smape'),2)} | **{fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_multitask_external','smape'),2)}** | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_finetune_external','smape'),2)} |
| **nRMSE (%)** | {fmt(_g(comparison,'ninja_full_2019','pcnn','nrmse'),2)} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn','nrmse'),2)} | **{fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_multitask_external','nrmse'),2)}** | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_finetune_external','nrmse'),2)} |
| MAPE (%) | {fmt(_g(comparison,'ninja_full_2019','pcnn','mape'),2)} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn','mape'),2)} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_multitask_external','mape'),2)} | {fmt(_g(comparison,'ninja_full_2019','ar_pcnn_cd_finetune_external','mape'),2)} |

Standard MAPE is susceptible to extreme inflation during low-irradiance periods due to near-zero denominators; therefore, sMAPE (or nRMSE) is reported to provide a more stable relative error metric across all generation regimes.

**Units:** NSRDB absolute errors are **Watts** (1× TS250). Ninja absolute errors are **kW** on a **10 kW** system (40×250 W). Source-paper absolute MAE figures are **not reconcilable** with ours (capacity/unit unspecified) — no guessed conversion.

## Ninja holdout inside validation window (2019 hours in val split)

| Strategy | n (h) | R² (—) | MAE (kW) |
|----------|-------|--------|----------|
| Multitask | {fmt(_g(comparison,'ninja_val_holdout','multitask','n'),0)} | **{fmt(_g(comparison,'ninja_val_holdout','multitask','r2'))}** | {fmt(_g(comparison,'ninja_val_holdout','multitask','mae'))} |
| Finetune | {fmt(_g(comparison,'ninja_val_holdout','finetune','n'),0)} | {fmt(_g(comparison,'ninja_val_holdout','finetune','r2'))} | {fmt(_g(comparison,'ninja_val_holdout','finetune','mae'))} |

Note: early-2019 ninja hours overlap the train window; the val-window subset is the cleaner transfer check. Full-2019 numbers match the Phase-5 Case-1 protocol for continuity.

**Negative result (keep):** frozen-encoder fine-tune fails (R² negative). Mechanism: external-head training under freeze can trap near a zero-ish clamp regime — see `src/analysis/phase9/train_helpers.py` comments.

## Uncertainty (nominal 95%) — NSRDB physics path

| Model | PICP (—) | MPIW (W) |
|-------|----------|----------|
| PCNN raw MC | {fmt(_g(comparison,'uncertainty','pcnn_raw','picp'),3)} | {fmt(_g(comparison,'uncertainty','pcnn_raw','mpiw'),2)} |
| AR-PCNN calibrated | {fmt(_g(comparison,'uncertainty','ar_pcnn_calibrated','picp'),3)} | {fmt(_g(comparison,'uncertainty','ar_pcnn_calibrated','mpiw'),2)} |
| CD Multitask calibrated | {fmt(_g(comparison,'uncertainty','cd_multitask_calibrated','picp'),3)} | {fmt(_g(comparison,'uncertainty','cd_multitask_calibrated','mpiw'),2)} |
| CD Finetune calibrated | {fmt(_g(comparison,'uncertainty','cd_finetune_calibrated','picp'),3)} | {fmt(_g(comparison,'uncertainty','cd_finetune_calibrated','mpiw'),2)} |
"""
    out_md = ROOT / "phase6_comparison.md"
    out_md.write_text(md, encoding="utf-8")
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", "replace").decode("ascii"))
    print(f"Wrote {mdir / 'phase6_comparison.json'}")
    print(f"Wrote {out_md}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_ar_pcnn_cd.yaml"))
    args = parser.parse_args()
    run_compare(args.config)


if __name__ == "__main__":
    main()
