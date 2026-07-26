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
            "- **Primary result:** Multitask CD lifts external Ninja R² ~0.5 → ~0.86 (val-holdout ~0.81) with masked-loss integrity checks.",
            "- Reproducible metrics verified in Phase 8 (independent recalc from arrays).",
            "- Ablation isolates residual, gate, calibration, and multitask contributions.",
            "- Gate α interpretable; conformal UQ reaches ~95% PICP.",
            "",
            "## Remaining limitations",
            "",
            "- NSRDB labels are physics-derived; near-unity R² is a **consistency check**, not a forecasting claim.",
            "- Full-2019 ninja evaluation overlaps early-2019 train hours for multitask — prefer val-holdout.",
            "- λ=0 leaves the external head untrained (Ninja R² negative) — expected control; not a multitask failure.",
            "- Feasibility / hard physics clamps are architectural constraints (true by construction), not learned achievements.",
            "",
            "## Recommendations for publication",
            "",
            "1. Lead with Ninja multitask transfer (val-holdout ~0.81); demote NSRDB to physics-fidelity sanity check.",
            "2. Report NSRDB as “≈1.0” + MAE (W) — avoid literal `R²=1.0000`.",
            "3. Include the λ-sensitivity figure; note Ninja peaks near λ≈0.25 and λ=0 is an untrained-head control.",
            "4. Disclose Phase 4 circular-label threat and clamp-by-construction feasibility.",
            "5. Do not align numeric tables to the source paper without resolved unit/protocol parity.",
            "6. Use figures from `data/figures/phase9/` at 300 DPI; prefer external-head importance over physics-path noise.",
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

    out = ROOT / "Documentation" / "phase9_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
