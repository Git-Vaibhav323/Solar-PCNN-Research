"""Assemble Phase 9 final publication report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .common import CSV_DIR, FIG_DIR, METRICS_DIR, REPORT_DIR, ROOT

SECTIONS = [
    ("exp01_lambda_sensitivity.md", "1. Hyperparameter sensitivity"),
    ("exp02_gate_interpretability.md", "2. Gate interpretability"),
    ("exp03_residual_analysis.md", "3. Residual analysis"),
    ("exp04_multi_seed_stability.md", "4. Multi-seed stability"),
    ("exp05_runtime_complexity.md", "5. Runtime & complexity"),
    ("exp06_feature_importance.md", "6. Feature importance"),
    ("exp07_embedding_visualization.md", "7. Embedding visualization"),
    ("exp08_regime_errors.md", "8. Regime-wise errors"),
    ("exp09_noise_robustness.md", "9. Noise robustness"),
]


def run() -> Path:
    lines = [
        "# Phase 9 — Publication Enhancement & Advanced Validation",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "Architecture frozen (Phases 3–8). All new experiments write to `artifacts/phase9/` and `data/figures/phase9/`.",
        "",
        "---",
        "",
    ]

    for fname, title in SECTIONS:
        p = REPORT_DIR / fname
        lines.append(f"## {title}")
        lines.append("")
        if p.exists():
            lines.append(p.read_text(encoding="utf-8").strip())
        else:
            lines.append("*Section not yet generated — run the corresponding experiment.*")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend(
        [
            "## Key strengths",
            "",
            "- Reproducible metrics verified in Phase 8 (173/173 audit checks).",
            "- Ablation isolates residual, gate, calibration, and multitask contributions.",
            "- Multitask CD lifts Ninja R² ~0.5 → ~0.86 with preserved NSRDB physics fidelity.",
            "- Gate α interpretable; conformal UQ reaches ~95% PICP.",
            "",
            "## Remaining limitations",
            "",
            "- NSRDB labels are physics-derived; near-unity R² is partly by construction.",
            "- Full-2019 ninja evaluation overlaps early-2019 train hours for multitask.",
            "- λ and seed sweeps require additional GPU time; results stored under `artifacts/phase9/`.",
            "",
            "## Recommendations for publication",
            "",
            "1. Report R² to 4–6 decimals or as “≈1.0” with MAE in Watts — avoid literal `R²=1.0000`.",
            "2. Lead with Ninja val-holdout (~0.81 R²) for cleaner transfer claim.",
            "3. Include λ sensitivity figure showing physics–ninja trade-off.",
            "4. Disclose Phase 4 circular-label threat in methods/limitations.",
            "5. Use figures from `data/figures/phase9/` at 300 DPI.",
            "",
            "## Output index",
            "",
            f"- CSV: `{CSV_DIR.relative_to(ROOT)}`",
            f"- Figures: `{FIG_DIR.relative_to(ROOT)}`",
            f"- Metrics JSON: `{METRICS_DIR.relative_to(ROOT)}`",
            f"- Per-experiment notes: `{REPORT_DIR.relative_to(ROOT)}`",
            "",
        ]
    )

    out = ROOT / "phase9_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
