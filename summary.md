# Solar-PCNN Research — Project Summary

*Living document. Written in plain language. Update this after each phase wraps up.*

**Last updated:** 25 Jul 2026 (submission-integrity cleanup)

---

## What this project is about

We reproduce a Physics-Constrained Neural Network (PCNN) for solar PV, then improve **cross-domain transfer** to Renewables.ninja via multi-task learning — always grounded in diagnostics, not random model growth.

**Primary claim:** masked multitask learning lifts Ninja R² ~0.5 → ~0.86 (val-holdout ~0.81).  
**Not a primary claim:** near-perfect NSRDB R² (labels are physics-derived).

Rule: **reproduce first, invent later** — keep earlier models frozen for fair comparison.

---

## Where we are

| Phase | What it means | Status |
|-------|----------------|--------|
| 1 | Collect data | Done |
| 2 | Clean & prepare data | Done |
| 3 | Rebuild the paper’s PCNN in PyTorch | Done (frozen baseline) |
| 4 | Dig into what works / what fails | Done |
| 5 | Novel AR-PCNN + calibrated uncertainty | Done (architecture frozen) |
| 6 | Cross-domain learning (NSRDB + ninja) | Done |
| 7 | Validation, ablation, interpretability, stats | Done |
| 8 | Scientific audit of all reported results | Done (magnitude-aware tol) |
| 9 | Publication enhancement & advanced validation | Done (full λ-sweep + multi-seed) |

---

## Phase 1–3 (short recap)

- NSRDB 2016–2020 + Renewables.ninja 2019 collected and prepared.
- Chronological splits; train-only normalization; physics target for TS250 panel.
- Baseline dual-branch PCNN trained on GPU.
- NSRDB test (consistency): R² ≈ 0.998, MAE ≈ 2.3 W, feasible=1.0 by clamp.
- Ninja Case 1: R² ≈ 0.54 (we do **not** align to published paper Case-1 numbers).
- MC Dropout: PICP ≈ 0.999 (too wide).

---

## Phase 4 — Limitations we designed against

1. Model mostly copies analytical physics (target ≈ `P_bounded`).
2. Poor ninja transfer under distribution / system shift.
3. Over-conservative uncertainty intervals.

Full memo: `phase4_limitations.md`.

---

## Phase 5 — AR-PCNN (frozen)

`ŷ = α × P_physics + r` (+ capacity clamp) with conformal-calibrated MC Dropout.

- NSRDB ≈ perfect as expected for physics labels; calibrated PICP ≈ 0.94.
- Ninja R² ≈ 0.49 (no free transfer gain from gating alone).

Details: `phase5_comparison.md`.

---

## Phase 6 — Cross-domain multi-task (**main result**)

Same physics path + external kW head; masked ninja loss.

- Multitask: ninja R² ≈ **0.86** (val holdout ≈ **0.81**); NSRDB stays near-perfect as consistency; PICP ≈ 0.96.
- Frozen-encoder fine-tune: failed (R² negative) — useful negative result.

Details: `phase6_comparison.md`.

---

## Phase 7 — Validation & publication evidence

Architecture **not** changed. Focus: integrity, ablation, interpretability, error analysis, statistics, figures.

### Integrity

**11/11** reproducibility checks passed (chronological splits, no leakage, train-only scaler, ninja not in Phase-5 loss, 2020 test has zero ninja labels, shared metrics).

### Ablation (identical protocol)

| Model | Ninja R² (—) | NSRDB R² (—) | PICP note |
|-------|--------------|--------------|-----------|
| A Original PCNN | 0.54 | 0.998 | raw MC over-wide |
| B Residual only | 0.49 | ~1.0 | residual collapses NSRDB error |
| C Residual + Gate | 0.49 | ~1.0 | α ≈ 0.94; interpretable |
| D + Calibrated UQ | 0.49 | ~1.0 | PICP → **0.95** (same point preds as C) |
| E + Multitask | **0.86** | ~1.0 | only arm that fixes domain shift |

NSRDB absolute errors (not shown above) are in **W**; Ninja R² is on a **10 kW** system.

### What we learned (plain language)

- High NSRDB scores partly reflect physics-derived labels — treat as fidelity, not forecasting.
- **Calibration (D)** is what makes uncertainty trustworthy; gating alone does not.
- **Multi-task (E)** is what lifts Renewables.ninja performance; architecture tricks without external labels do not.
- Gate α stays high: the model trusts physics by day and adjusts more at night; ninja adaptation lives in the external head.

### Deliverables

- Report: `phase7_validation.md`
- Metrics: `artifacts/metrics/phase7_report.json`
- Figures: `data/figures/phase7_publication/`
- Runner: `python -m src.analysis.phase7_run`

---

## Phase 8 — Scientific audit

Independent recalculation of MAE/RMSE/R²/MAPE/sMAPE/nRMSE/PICP/MPIW from saved prediction arrays.

- **178/178 checks PASS** (magnitude-aware tolerance: abs 1e−3 for O(1); 5% relative for |value|<1e−2)
- Publication readiness: **READY with caveats**
- Full ledger: `phase8_audit.md` / `artifacts/metrics/phase8_audit.json`
- Re-run: `python -m src.analysis.phase8_audit`

---

## Main lessons (project)

1. Phase 4 was right: architecture alone cannot fix domain shift.
2. Residual + gate gives interpretability and near-perfect physics reconstruction (by design of the label).
3. Conformal calibration fixes UQ coverage.
4. External supervision + shared encoder lifts ninja R² ~0.5 → ~0.86 — **this is the paper contribution**.
5. Multi-task > frozen-head fine-tune.
6. Phase 7 ablations isolate each contribution with shared metrics and integrity checks.

---

## What’s next (paper)

- [ ] Write methods/results around **Ninja multitask transfer** (not NSRDB R²≈1)
- [ ] Select final publication figures from `data/figures/phase7_publication/`
- [ ] Optional: complete full `lambda_external` sweep before claiming λ-sensitivity
- [ ] Optional: metered PV targets if available

---

## Key files

| Item | Path |
|------|------|
| Baseline PCNN | `src/models/pcnn.py`, `src/training/train.py` |
| AR-PCNN | `src/models/ar_pcnn.py`, `src/training/train_ar_pcnn.py` |
| Cross-domain | `src/models/ar_pcnn_cd.py`, `src/training/train_ar_pcnn_cd.py` |
| Residual ablation (B) | `src/models/residual_only_pcnn.py`, `src/training/train_ablation_residual.py` |
| Phase 7 runner | `src/analysis/phase7_run.py` |
| Phase 4 memo | `phase4_limitations.md` |
| Phase 5 table | `phase5_comparison.md` |
| Phase 6 table | `phase6_comparison.md` |
| Phase 7 validation | `phase7_validation.md` |
| Phase 8 audit | `phase8_audit.md`, `src/analysis/phase8_audit.py` |

---

## Phase 9 — Publication enhancement (optional)

New analysis-only code under `src/analysis/phase9/`. Does **not** modify Phase 3–8 artifacts.

**Run all experiments:**
```powershell
.\.venv\Scripts\Activate.ps1
python -m src.analysis.phase9.run_all
```

**Analysis only** (skip λ/seed retraining; uses frozen Phase-6 checkpoint where applicable):
```powershell
python -m src.analysis.phase9.run_all --skip-train
```

**Integrity note:** `--skip-train` evaluates existing checkpoints only. Full λ training needs `run_all` without `--skip-train` (completed 25 Jul 2026).

Outputs: `artifacts/phase9/`, `data/figures/phase9/`, `phase9_report.md`

---

## One-line status

**Primary result is Ninja multitask transfer (~0.86 / holdout ~0.81); NSRDB near-unity R² is disclosed as physics-label consistency. Audit 178/178. Full λ-sweep complete (Ninja peaks ~λ=0.25).**
