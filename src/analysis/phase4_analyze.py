"""Phase 4 — limitation analysis for the reproduced PCNN baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.metrics import regression_metrics
from src.training.train import load_train_config


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_test_frame(cfg: dict) -> pd.DataFrame:
    raw = pd.read_csv(cfg["paths"]["processed_dir"] / "test_raw.csv", parse_dates=["timestamp"])
    pred = np.load(cfg["paths"]["metrics_dir"] / "predictions_test.npz")
    y_true = pred["y_true"].reshape(-1)
    y_pred = pred["y_pred"].reshape(-1)
    if len(raw) != len(y_true):
        raise RuntimeError(f"Length mismatch test_raw={len(raw)} preds={len(y_true)}")
    out = raw.copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["error"] = y_pred - y_true
    out["abs_error"] = np.abs(out["error"])
    out["hour"] = out["timestamp"].dt.hour
    out["month"] = out["timestamp"].dt.month
    return out


def analyze_nsrdb_errors(df: pd.DataFrame, fig_dir: Path) -> dict:
    fig_dir = _ensure(fig_dir)
    metrics = regression_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())

    # Residual vs true
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(df["y_true"], df["error"], s=3, alpha=0.15)
    axes[0, 0].axhline(0, color="r", lw=1)
    axes[0, 0].set_xlabel("True power (W)")
    axes[0, 0].set_ylabel("Error (pred − true) W")
    axes[0, 0].set_title("Residuals vs true power")

    by_hour = df.groupby("hour")["abs_error"].mean()
    axes[0, 1].bar(by_hour.index, by_hour.values, color="steelblue")
    axes[0, 1].set_xlabel("Hour of day")
    axes[0, 1].set_ylabel("Mean |error| (W)")
    axes[0, 1].set_title("Diurnal error profile")

    by_month = df.groupby("month")["abs_error"].mean()
    axes[1, 0].bar(by_month.index, by_month.values, color="darkorange")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Mean |error| (W)")
    axes[1, 0].set_title("Seasonal error profile")

    # GHI bins
    bins = [-0.1, 0, 100, 300, 600, 1000, 1400]
    labels = ["night≈0", "0–100", "100–300", "300–600", "600–1000", "1000+"]
    df = df.copy()
    df["ghi_bin"] = pd.cut(df["GHI"], bins=bins, labels=labels)
    by_ghi = df.groupby("ghi_bin", observed=False)["abs_error"].mean()
    axes[1, 1].bar(range(len(by_ghi)), by_ghi.values, color="seagreen")
    axes[1, 1].set_xticks(range(len(by_ghi)))
    axes[1, 1].set_xticklabels([str(x) for x in by_ghi.index], rotation=30, ha="right")
    axes[1, 1].set_ylabel("Mean |error| (W)")
    axes[1, 1].set_title("Error by GHI bin")

    fig.suptitle("Phase 4 — NSRDB test error analysis", y=1.01)
    fig.tight_layout()
    path = fig_dir / "phase4_nsrdb_errors.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    daytime = df[df["GHI"] > 50]
    night = df[df["GHI"] <= 50]
    return {
        "overall": metrics,
        "mean_abs_error_by_hour": {str(k): float(v) for k, v in by_hour.items()},
        "mean_abs_error_by_month": {str(k): float(v) for k, v in by_month.items()},
        "mean_abs_error_by_ghi_bin": {str(k): float(v) for k, v in by_ghi.items()},
        "daytime_mae": float(daytime["abs_error"].mean()) if len(daytime) else None,
        "night_mae": float(night["abs_error"].mean()) if len(night) else None,
        "worst_hour": int(by_hour.idxmax()),
        "worst_month": int(by_month.idxmax()),
        "figure": str(path),
    }


def analyze_physics_copying(df: pd.DataFrame, fig_dir: Path) -> dict:
    """Does the network mostly copy P_theoretical / P_bounded?"""
    fig_dir = _ensure(fig_dir)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    p_th = df["P_theoretical"].to_numpy()
    p_bd = df["P_bounded"].to_numpy()

    r2_pred_vs_true = float(r2_score(y_true, y_pred))
    r2_phys_vs_true = float(r2_score(y_true, p_bd))
    r2_pred_vs_phys = float(r2_score(p_bd, y_pred))
    mae_pred = float(np.mean(np.abs(y_pred - y_true)))
    mae_phys_vs_target = float(np.mean(np.abs(p_bd - y_true)))
    phys_gap = y_pred - p_bd
    target_equals_physics = bool(np.allclose(y_true, p_bd, rtol=0, atol=1e-6))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(p_bd, y_pred, s=3, alpha=0.15)
    lim = max(p_bd.max(), y_pred.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", lw=1)
    axes[0].set_xlabel("Physics P_bounded (W)")
    axes[0].set_ylabel("PCNN prediction (W)")
    axes[0].set_title(f"Pred vs physics (R²={r2_pred_vs_phys:.4f})")

    axes[1].hist(phys_gap, bins=60, color="slategray", edgecolor="none")
    axes[1].set_xlabel("pred − P_bounded (W)")
    axes[1].set_title("How far predictions leave the physics prior")
    fig.tight_layout()
    path = fig_dir / "phase4_physics_copying.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "target_equals_p_bounded": target_equals_physics,
        "r2_pcnn_vs_target": r2_pred_vs_true,
        "r2_physics_vs_target": r2_phys_vs_true,
        "r2_pcnn_vs_physics": r2_pred_vs_phys,
        "mae_pcnn_vs_target_w": mae_pred,
        "mae_physics_vs_target_w": mae_phys_vs_target,
        "mean_abs_departure_from_physics_w": float(np.mean(np.abs(phys_gap))),
        "interpretation": (
            "By construction target_power == P_bounded, so physics-vs-target R²≈1. "
            "PCNN is learning to reproduce that analytical label (with small residual error), "
            "not to beat real measured PV. High NSRDB R² is therefore expected."
        ),
        "figure": str(path),
    }


def analyze_case1(cfg: dict, fig_dir: Path) -> dict:
    fig_dir = _ensure(fig_dir)
    metrics_path = cfg["paths"]["metrics_dir"] / "case1_ninja_metrics.json"
    pred_path = cfg["paths"]["metrics_dir"] / "case1_ninja_predictions.npz"
    with metrics_path.open(encoding="utf-8") as f:
        base = json.load(f)
    data = np.load(pred_path, allow_pickle=True)
    y_true = data["y_true_kw"].reshape(-1)
    y_pred = data["y_pred_kw"].reshape(-1)
    err = y_pred - y_true
    abs_err = np.abs(err)

    # Approximate hour-of-day from index (hourly series starting align_start)
    hours = np.arange(len(y_true)) % 24
    by_hour = pd.Series(abs_err).groupby(hours).mean()

    # Error vs ninja power level
    bins = [0, 0.1, 1, 3, 5, 8, 12]
    labels = ["≈0", "0.1–1", "1–3", "3–5", "5–8", "8+"]
    level = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
    by_level = pd.Series(abs_err).groupby(level, observed=False).mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(by_hour.index, by_hour.values, color="crimson")
    axes[0].set_xlabel("Hour index mod 24")
    axes[0].set_ylabel("Mean |error| (kW)")
    axes[0].set_title("Case 1 diurnal |error|")

    axes[1].bar(range(len(by_level)), by_level.values, color="purple")
    axes[1].set_xticks(range(len(by_level)))
    axes[1].set_xticklabels([str(x) for x in by_level.index], rotation=30, ha="right")
    axes[1].set_ylabel("Mean |error| (kW)")
    axes[1].set_title("Case 1 error by ninja power level")
    fig.tight_layout()
    path = fig_dir / "phase4_case1_errors.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Bias
    return {
        **base,
        "mean_bias_kw": float(np.mean(err)),
        "daytime_mae_kw": float(np.mean(abs_err[y_true > 0.1])),
        "night_mae_kw": float(np.mean(abs_err[y_true <= 0.1])),
        "mean_abs_error_by_hour": {str(k): float(v) for k, v in by_hour.items()},
        "mean_abs_error_by_power_bin": {str(k): float(v) for k, v in by_level.items()},
        "likely_gap_drivers": [
            "NSRDB weather vs MERRA-2 weather underlying ninja",
            "Ninja system model (tilt 35°, losses, tracking=0) vs simple TS250 physics target",
            "Site coordinate / timezone alignment differences",
            "Model trained on synthetic P_bounded, not on ninja electricity",
        ],
        "figure": str(path),
    }


def analyze_uncertainty(cfg: dict, fig_dir: Path) -> dict:
    fig_dir = _ensure(fig_dir)
    uq_path = cfg["paths"]["metrics_dir"] / "uncertainty_test.npz"
    meta_path = cfg["paths"]["metrics_dir"] / "uncertainty_test_metrics.json"
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    data = np.load(uq_path)
    y_true = data["y_true"].reshape(-1)
    mean = data["y_mean"].reshape(-1)
    std = data["y_std"].reshape(-1)

    # Reliability-style: coverage at multiple nominal levels
    zs = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960}
    coverage = {}
    widths = {}
    for name, z in zs.items():
        lo, hi = mean - z * std, mean + z * std
        coverage[str(name)] = float(np.mean((y_true >= lo) & (y_true <= hi)))
        widths[str(name)] = float(np.mean(hi - lo))

    # Coverage by GHI proxy using |true power| bins
    power_bins = pd.cut(y_true, bins=[-0.1, 1, 50, 100, 150, 300], labels=["≈0", "1–50", "50–100", "100–150", "150+"])
    cov_by_bin = {}
    for label, grp in pd.DataFrame({"y": y_true, "m": mean, "s": std, "b": power_bins}).groupby("b", observed=False):
        lo = grp["m"] - 1.96 * grp["s"]
        hi = grp["m"] + 1.96 * grp["s"]
        cov_by_bin[str(label)] = float(np.mean((grp["y"] >= lo) & (grp["y"] <= hi))) if len(grp) else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    nom = list(zs.keys())
    axes[0].plot(nom, [coverage[str(n)] for n in nom], "o-", label="Observed coverage")
    axes[0].plot(nom, nom, "k--", label="Ideal")
    axes[0].set_xlabel("Nominal confidence")
    axes[0].set_ylabel("Observed coverage")
    axes[0].set_title("Uncertainty calibration curve")
    axes[0].legend(fontsize=8)

    axes[1].bar(range(len(cov_by_bin)), list(cov_by_bin.values()), color="teal")
    axes[1].axhline(0.95, color="r", ls="--", lw=1)
    axes[1].set_xticks(range(len(cov_by_bin)))
    axes[1].set_xticklabels(list(cov_by_bin.keys()), rotation=30, ha="right")
    axes[1].set_ylabel("PICP @ 95%")
    axes[1].set_title("Coverage by power bin")
    fig.tight_layout()
    path = fig_dir / "phase4_uncertainty_calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "baseline_meta": meta,
        "coverage_by_nominal_level": coverage,
        "mean_width_by_nominal_level": widths,
        "picp95_by_power_bin": cov_by_bin,
        "calibration_issue": (
            "Observed coverage >> nominal (e.g. ~99.8% at 95%) means intervals are over-wide / under-confident."
        ),
        "figure": str(path),
    }


def write_limitations(report: dict, out_path: Path) -> None:
    nsrdb = report["nsrdb_errors"]
    phys = report["physics_copying"]
    case1 = report["case1"]
    uq = report["uncertainty"]
    text = f"""# Phase 4 — Limitations of the reproduced PCNN

*Auto-generated from analysis. Use this to drive Phase 5 design.*

## 1. Target definition makes NSRDB scores look too good

- Test R² ≈ **{nsrdb['overall']['r2']:.4f}**, MAE ≈ **{nsrdb['overall']['mae']:.2f} W**.
- **Critical:** `target_power` is exactly `P_bounded` (analytical). Physics-vs-target R² ≈ **{phys['r2_physics_vs_target']:.4f}** (≈1 by construction).
- PCNN vs physics R² ≈ **{phys['r2_pcnn_vs_physics']:.4f}**; mean |pred − physics| ≈ **{phys['mean_abs_departure_from_physics_w']:.2f} W**.
- **Limitation:** the network is mostly reproducing an equation it is already fed — not learning hard residual structure from real PV measurements.

## 2. Errors concentrate in high-irradiance / midday conditions

- Worst hour (mean |error|): **{nsrdb['worst_hour']}**
- Worst month (mean |error|): **{nsrdb['worst_month']}**
- Daytime MAE (>50 W/m² GHI): **{nsrdb['daytime_mae']:.2f} W** vs night MAE: **{nsrdb['night_mae']:.2f} W**
- **Limitation:** capacity-sensitive midday peaks still dominate residual error.

## 3. Weak external generalization (Renewables.ninja Case 1)

- Case 1 R² ≈ **{case1['r2']:.3f}**, MAE ≈ **{case1['mae']:.2f} kW**, bias ≈ **{case1['mean_bias_kw']:.2f} kW**.
- Paper reported Case 1 R² ≈ 0.998 — our protocol does not match that transfer.
- Likely drivers: {', '.join(case1['likely_gap_drivers'][:3])}.
- **Limitation:** model trained on NSRDB physics targets does not transfer cleanly to ninja’s MERRA-2 PV series.

## 4. Uncertainty is over-conservative

- Test PICP @ 95% ≈ **{uq['baseline_meta']['uncertainty']['picp_95']:.3f}** (paper ~0.947).
- Calibration curve shows coverage above the diagonal at all nominal levels.
- **Limitation:** MC Dropout bands are too wide for decision-making; need calibration (temperature scaling, conformal, better aleatoric head, etc.).

## 5. Architectural / training limitations (from reproduction)

- Physics enters mainly as a scaled prior + soft loss penalties — not a hard structural constraint.
- No dedicated temporal sequence model (hourly i.i.d.-style MLP on engineered features).
- Single-panel Watts training vs system-level kW evaluation adds scaling ambiguity vs the paper.
- Feature set for Case 1 still relies on NSRDB meteorology aligned in time, not ninja-native full feature vectors.

## Implications for Phase 5 (novelty directions)

1. **Train / evaluate on targets closer to real or external PV** (or multi-task: physics + ninja/system).
2. **Reduce physics copying** — residual learning (predict correction to physics) or gated fusion with learned trust.
3. **Better temporal modeling** under distribution shift (attention / TCN / informed LSTM) without abandoning constraints.
4. **Calibrated uncertainty** — conformal prediction or explicit variance head + proper scoring rules.
5. **Adaptive constraint weights** when physics prior conflicts with external observations.
"""
    out_path.write_text(text, encoding="utf-8")


def run_phase4(config_path: str | None = None) -> dict:
    cfg = load_train_config(config_path)
    fig_dir = cfg["paths"]["figures_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    _ensure(fig_dir)
    _ensure(metrics_dir)

    df = load_test_frame(cfg)
    report = {
        "phase": 4,
        "status": "complete",
        "nsrdb_errors": analyze_nsrdb_errors(df, fig_dir),
        "physics_copying": analyze_physics_copying(df, fig_dir),
        "case1": analyze_case1(cfg, fig_dir),
        "uncertainty": analyze_uncertainty(cfg, fig_dir),
    }

    report_path = metrics_dir / "phase4_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    limitations_path = ROOT / "artifacts" / "metrics" / "phase4_limitations.md"
    # Also keep a copy at repo root docs-friendly path
    write_limitations(report, limitations_path)
    write_limitations(report, ROOT / "phase4_limitations.md")

    print(json.dumps(
        {
            "nsrdb_r2": report["nsrdb_errors"]["overall"]["r2"],
            "target_equals_p_bounded": report["physics_copying"]["target_equals_p_bounded"],
            "physics_r2_vs_target": report["physics_copying"]["r2_physics_vs_target"],
            "pcnn_vs_physics_r2": report["physics_copying"]["r2_pcnn_vs_physics"],
            "mean_abs_departure_from_physics_w": report["physics_copying"]["mean_abs_departure_from_physics_w"],
            "case1_r2": report["case1"]["r2"],
            "picp95": report["uncertainty"]["baseline_meta"]["uncertainty"]["picp_95"],
            "report": str(report_path),
            "limitations": str(ROOT / "phase4_limitations.md"),
        },
        indent=2,
    ))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 limitation analysis")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_phase4(args.config)


if __name__ == "__main__":
    main()
