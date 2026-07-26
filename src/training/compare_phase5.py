"""Fair Phase-3 vs Phase-5 comparison table (identical protocol)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def run_compare(metrics_dir: Path | None = None) -> dict:
    metrics_dir = metrics_dir or (ROOT / "artifacts" / "metrics")

    baseline_test = _load(metrics_dir / "metrics_test.json") or _load(metrics_dir / "phase3_summary.json").get("test", {})
    if "mae" not in baseline_test and (metrics_dir / "phase3_summary.json").exists():
        baseline_test = _load(metrics_dir / "phase3_summary.json").get("test", {})

    ar_test = _load(metrics_dir / "ar_pcnn_metrics_test.json")
    if not ar_test:
        ar_test = _load(metrics_dir / "ar_pcnn_phase5_summary.json").get("test", {})

    baseline_case1 = _load(metrics_dir / "case1_ninja_metrics.json")
    ar_case1 = _load(metrics_dir / "ar_pcnn_case1_metrics.json")

    baseline_uq = _load(metrics_dir / "uncertainty_test_metrics.json")
    ar_uq = _load(metrics_dir / "ar_pcnn_uncertainty_test_metrics.json")

    def g(d, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    comparison = {
        "nsrdb_test": {
            "baseline_pcnn": {
                "mae": g(baseline_test, "mae"),
                "rmse": g(baseline_test, "rmse"),
                "r2": g(baseline_test, "r2"),
                "mape": g(baseline_test, "mape"),
                "feasible_fraction": g(baseline_test, "feasible_fraction"),
            },
            "ar_pcnn": {
                "mae": g(ar_test, "mae"),
                "rmse": g(ar_test, "rmse"),
                "r2": g(ar_test, "r2"),
                "mape": g(ar_test, "mape"),
                "feasible_fraction": g(ar_test, "feasible_fraction"),
                "mean_alpha": g(ar_test, "mean_alpha"),
                "mean_abs_residual": g(ar_test, "mean_abs_residual"),
            },
        },
        "case1_ninja": {
            "baseline_pcnn": {
                "mae": g(baseline_case1, "mae"),
                "rmse": g(baseline_case1, "rmse"),
                "r2": g(baseline_case1, "r2"),
                "mape": g(baseline_case1, "mape"),
                "smape": g(baseline_case1, "smape"),
                "nrmse": g(baseline_case1, "nrmse"),
            },
            "ar_pcnn": {
                "mae": g(ar_case1, "mae"),
                "rmse": g(ar_case1, "rmse"),
                "r2": g(ar_case1, "r2"),
                "mape": g(ar_case1, "mape"),
                "smape": g(ar_case1, "smape"),
                "nrmse": g(ar_case1, "nrmse"),
            },
        },
        "uncertainty_test": {
            "baseline_pcnn_uncalibrated": {
                "picp": g(baseline_uq, "uncertainty", "picp_95"),
                "mpiw": g(baseline_uq, "uncertainty", "mpiw"),
            },
            "ar_pcnn_uncalibrated": {
                "picp": g(ar_uq, "uncalibrated", "picp_95"),
                "mpiw": g(ar_uq, "uncalibrated", "mpiw"),
            },
            "ar_pcnn_calibrated": {
                "picp": g(ar_uq, "calibrated_on_val_applied_to_test", "picp"),
                "mpiw": g(ar_uq, "calibrated_on_val_applied_to_test", "mpiw"),
                "scale": g(ar_uq, "calibrated_on_val_applied_to_test", "scale"),
            },
        },
    }

    out_json = metrics_dir / "phase5_comparison.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    def fmt(x, nd=4):
        if x is None:
            return "—"
        return f"{x:.{nd}f}"

    md = f"""# Phase 5 Comparison — Baseline PCNN vs AR-PCNN

Identical Phase-2 splits, preprocessing, and evaluation protocol.

**Headline for publication:** use Phase 6 Ninja multitask transfer, not these NSRDB scores. NSRDB targets are physics-derived (`target_power` ≈ `P_bounded`), so near-perfect R² here is a consistency check.

## NSRDB test (single panel, Watts) — physics-fidelity sanity check

| Metric | Baseline PCNN | AR-PCNN |
|--------|---------------|---------|
| MAE (W) | {fmt(g(comparison,'nsrdb_test','baseline_pcnn','mae'))} | {fmt(g(comparison,'nsrdb_test','ar_pcnn','mae'))} |
| RMSE (W) | {fmt(g(comparison,'nsrdb_test','baseline_pcnn','rmse'))} | {fmt(g(comparison,'nsrdb_test','ar_pcnn','rmse'))} |
| R² (—) | {fmt(g(comparison,'nsrdb_test','baseline_pcnn','r2'))} | ≈1.0 |
| MAPE (%) | {fmt(g(comparison,'nsrdb_test','baseline_pcnn','mape'), 3)} | {fmt(g(comparison,'nsrdb_test','ar_pcnn','mape'), 3)} |
| Feasible (—) | {fmt(g(comparison,'nsrdb_test','baseline_pcnn','feasible_fraction'), 3)} | {fmt(g(comparison,'nsrdb_test','ar_pcnn','feasible_fraction'), 3)} |
| Mean alpha (—) | — | {fmt(g(comparison,'nsrdb_test','ar_pcnn','mean_alpha'), 3)} |
| Mean |r| (W) | — | {fmt(g(comparison,'nsrdb_test','ar_pcnn','mean_abs_residual'), 3)} |

`Feasible = 1.0` is guaranteed by the hard capacity clamp in the forward pass (architectural constraint), not a learned achievement.

## Case 1 — Renewables.ninja (10 kW) — primary external metric for Phase 5

| Metric | Baseline PCNN | AR-PCNN |
|--------|---------------|---------|
| MAE (kW) | {fmt(g(comparison,'case1_ninja','baseline_pcnn','mae'))} | {fmt(g(comparison,'case1_ninja','ar_pcnn','mae'))} |
| RMSE (kW) | {fmt(g(comparison,'case1_ninja','baseline_pcnn','rmse'))} | {fmt(g(comparison,'case1_ninja','ar_pcnn','rmse'))} |
| R² (—) | {fmt(g(comparison,'case1_ninja','baseline_pcnn','r2'))} | {fmt(g(comparison,'case1_ninja','ar_pcnn','r2'))} |
| **sMAPE (%)** | {fmt(g(comparison,'case1_ninja','baseline_pcnn','smape'), 2)} | {fmt(g(comparison,'case1_ninja','ar_pcnn','smape'), 2)} |
| **nRMSE (%)** | {fmt(g(comparison,'case1_ninja','baseline_pcnn','nrmse'), 2)} | {fmt(g(comparison,'case1_ninja','ar_pcnn','nrmse'), 2)} |
| MAPE (%) | {fmt(g(comparison,'case1_ninja','baseline_pcnn','mape'), 2)} | {fmt(g(comparison,'case1_ninja','ar_pcnn','mape'), 2)} |

Standard MAPE is susceptible to extreme inflation during low-irradiance periods due to near-zero denominators; therefore, sMAPE (or nRMSE) is reported to provide a more stable relative error metric across all generation regimes. MAPE is retained only for continuity with earlier tables.

**Units:** NSRDB absolute errors are **Watts** (1× TS250 panel, `p_rated_w=250`). Case-1 absolute errors are **kW** for a **10 kW** system (40×250 W). Do not mix rows across tables. The source paper’s absolute-error figures (e.g. MAE≈0.0213 in their tables) are **not reconcilable** with ours: system capacity / unit for those published numbers is not specified clearly enough to convert — we do not guess `0.0213 kW ↔ 21.3 W`.

## Uncertainty (test, nominal 95%) — intervals on NSRDB physics path (Watts)

| Setting | PICP (—) | MPIW (W) |
|---------|----------|----------|
| Baseline MC Dropout | {fmt(g(comparison,'uncertainty_test','baseline_pcnn_uncalibrated','picp'), 3)} | {fmt(g(comparison,'uncertainty_test','baseline_pcnn_uncalibrated','mpiw'), 2)} |
| AR-PCNN MC Dropout (raw) | {fmt(g(comparison,'uncertainty_test','ar_pcnn_uncalibrated','picp'), 3)} | {fmt(g(comparison,'uncertainty_test','ar_pcnn_uncalibrated','mpiw'), 2)} |
| AR-PCNN + conformal scale | {fmt(g(comparison,'uncertainty_test','ar_pcnn_calibrated','picp'), 3)} | {fmt(g(comparison,'uncertainty_test','ar_pcnn_calibrated','mpiw'), 2)} |

Calibration scale on val → test: **{fmt(g(comparison,'uncertainty_test','ar_pcnn_calibrated','scale'), 3)}**
"""
    out_md = ROOT / "Documentation" / "phase5_comparison.md"
    out_md.write_text(md, encoding="utf-8")
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", errors="replace").decode("ascii"))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default=None)
    args = parser.parse_args()
    run_compare(Path(args.metrics_dir) if args.metrics_dir else None)


if __name__ == "__main__":
    main()
