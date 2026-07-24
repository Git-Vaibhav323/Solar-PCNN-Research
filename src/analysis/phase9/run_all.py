"""
Phase 9 master runner — publication enhancement & advanced validation.

Usage:
  python -m src.analysis.phase9.run_all
  python -m src.analysis.phase9.run_all --skip-train   # analysis only (uses existing phase9 ckpts)
  python -m src.analysis.phase9.run_all --quick-train  # cap epochs for smoke test
"""

from __future__ import annotations

import argparse
import sys
import time

from . import (
    exp01_lambda_sensitivity,
    exp02_gate_interpretability,
    exp03_residual_analysis,
    exp04_multi_seed_stability,
    exp05_runtime_complexity,
    exp06_feature_importance,
    exp07_embedding_visualization,
    exp08_regime_errors,
    exp09_noise_robustness,
    report,
)
from .common import ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 — run all publication experiments")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip lambda/seed training (analysis-only; requires existing phase9 checkpoints)",
    )
    parser.add_argument(
        "--experiments",
        default="all",
        help="Comma-separated exp numbers 1-9 or 'all'",
    )
    args = parser.parse_args()

    ensure_dirs()
    exps = set(range(1, 10)) if args.experiments == "all" else {int(x) for x in args.experiments.split(",")}

    t0 = time.time()
    print("=" * 60)
    print("Phase 9 — Publication Enhancement & Advanced Validation")
    print("=" * 60)

    if 1 in exps:
        print("\n[1/9] Hyperparameter sensitivity ...")
        exp01_lambda_sensitivity.run(skip_train=args.skip_train)

    if 2 in exps:
        print("\n[2/9] Gate interpretability ...")
        exp02_gate_interpretability.run()

    if 3 in exps:
        print("\n[3/9] Residual analysis ...")
        exp03_residual_analysis.run()

    if 4 in exps:
        print("\n[4/9] Multi-seed stability ...")
        exp04_multi_seed_stability.run(skip_train=args.skip_train)

    if 5 in exps:
        print("\n[5/9] Runtime & complexity ...")
        exp05_runtime_complexity.run()

    if 6 in exps:
        print("\n[6/9] Feature importance ...")
        exp06_feature_importance.run()

    if 7 in exps:
        print("\n[7/9] Embedding visualization ...")
        exp07_embedding_visualization.run()

    if 8 in exps:
        print("\n[8/9] Regime-wise errors ...")
        exp08_regime_errors.run()

    if 9 in exps:
        print("\n[9/9] Noise robustness ...")
        exp09_noise_robustness.run()

    print("\n[Report] Assembling phase9_report.md ...")
    path = report.run()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min.")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
