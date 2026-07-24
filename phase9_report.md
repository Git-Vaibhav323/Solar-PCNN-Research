# Phase 9 — Publication Enhancement & Advanced Validation

*Evidence-audited rewrite: 2026-07-24. Architecture frozen (Phases 3–8). Outputs under `artifacts/phase9/` and `data/figures/phase9/`.*

**Evidence legend:** `M` = directly measured · `C` = computed from metrics · `F` = inferred from figures · `I` = interpretation · `U` = unsupported by current artifacts (removed or rewritten)

---

## Scope of generated artifacts

| Experiment | Status of evidence |
|------------|--------------------|
| Exp 1 λ sensitivity | **Incomplete** — only λ=1.0 evaluated (Phase-6 frozen ckpt). Other λ values not trained. |
| Exp 2 Gate α | Complete (AR-PCNN + CD) |
| Exp 3 Residuals | Complete |
| Exp 4 Multi-seed | **Incomplete** — only seed=42 |
| Exp 5 Runtime | Complete (params, inference, GPU MB); training times empty |
| Exp 6 Feature importance | Complete (permutation) |
| Exp 7 Embeddings | PCA/t-SNE figures+CSV; UMAP not available; no quantitative domain-distance metric |
| Exp 8 Regime errors | Complete |
| Exp 9 Noise | Complete |

---

## 1. Hyperparameter sensitivity (`lambda_external`)

### Claim ledger

| Original statement | Type | Support | Verdict |
|--------------------|------|---------|---------|
| Swept λ ∈ {0, 0.05, 0.1, 0.25, 0.5, 1.0} | U | Only λ=1.0 row in CSV | **Unsupported** |
| Each run warm-starts from Phase-5 AR-PCNN | I/code | Training helper design; not verified for missing λ runs | Not claimable for this report |
| NSRDB R² stays ≈1 across λ | U | Single λ only | **Unsupported** |
| Lowest NSRDB MAE at λ=1.0 (0.0002545 W) | M (value) / U (“lowest”) | `exp01_lambda_sensitivity.csv` has MAE=0.0002545 at λ=1.0 only | Value OK; “lowest” not OK |
| Ninja R² peaks at λ=1.0 (0.8639) | M (value) / U (“peaks”) | same CSV: ninja_r2=0.8639 | Value OK; “peaks” not OK |
| Higher λ pulls encoder toward ninja; λ≈0.25–1.0 balances | I / U | No multi-λ comparison | **Unsupported** |
| PICP near 0.95 for all trained λ | M (one λ) / U (“all”) | CSV picp=0.957 at λ=1.0 | Partial |

### Conservative rewrite

- **Measured (λ=1.0 only):** NSRDB R² = 0.999999999139, MAE = 2.545×10⁻⁴ W, RMSE = 2.231×10⁻³ W; Ninja R² = 0.8639, MAE = 0.504 kW; calibrated PICP = 0.957, MPIW = 0.0126.
- **Source:** `artifacts/phase9/csv/exp01_lambda_sensitivity.csv`, `artifacts/phase9/metrics/exp01_lambda_sensitivity.json`, figure `data/figures/phase9/exp01_lambda_sensitivity.png` (single-point plot).
- **Not yet supported:** any statement comparing multiple λ values. Full sweep requires `python -m src.analysis.phase9.run_all --experiments 1` without `--skip-train`.

---

## 2. Gate (α) interpretability

### Claim ledger

| Original statement | Type | Support | Verdict |
|--------------------|------|---------|---------|
| AR-PCNN mean α ≈ 0.937 | M | `exp02_gate_summary.csv`: 0.93651 | Supported |
| Daylight (GHI>50) α clusters near 1 | M | mean_day = 1.0 | Supported |
| Model trusts analytical physics | I | — | Interpretation only |
| Night α spreads below 1 when GHI≈0 | M + F | mean_night=0.886, min=0.753; `exp02_gate_AR_PCNN.png` | Supported (with figure) |
| Gate down-weights physics at night | I | — | Interpretation |
| CD mean α ≈ 1.000 | M | mean_all=0.99962 | Supported (state 0.9996) |
| Multitask preserves physics trust | I | — | Interpretation |
| Hourly α highest at midday | F | No hourly CSV; only figure | Figure-inferred |
| Monthly variation modest | F | No monthly CSV of α means | Figure-inferred |
| No strong T/DNI monotonic trend | F / I | Scatter panels in gate figures | Soft; keep as qualitative |

### Conservative rewrite

- **Measured:** AR-PCNN mean α = 0.9365 (day 1.000, night 0.886; range [0.753, 1.000]). CD Multitask mean α = 0.9996 (day 1.000, night 0.9993; range [0.994, 1.000]).
- **Source:** `artifacts/phase9/csv/exp02_gate_summary.csv`, `artifacts/phase9/metrics/exp02_gate.json`, figures `exp02_gate_AR_PCNN.png`, `exp02_gate_CD_Multitask.png`.
- **Interpretation (not a measurement):** High daytime α is consistent with heavy reliance on the analytical physics prior on the NSRDB physics path. Nighttime α dispersion for AR-PCNN is visible in the GHI scatter panel of `exp02_gate_AR_PCNN.png`.

---

## 3. Residual analysis

### Claim ledger

| Original statement | Type | Support | Verdict |
|--------------------|------|---------|---------|
| mean \|r\| ≈ 7.46e-05 W, p95 ≈ 7.63e-05 W | M | `exp03_residual_stats.csv` (AR_PCNN) | Supported |
| Tiny because target is physics-derived | I | Phase 4 finding, not Phase 9 measure | Interpretation / prior result |
| \|r\| grows with irradiance / midday | F | `exp03_residual_analysis.png` | Figure-inferred |
| Corrections matter most when power large | I | — | Interpretation |
| Monthly/seasonal means stable | M | `exp03_residual_by_season.csv`: winter 7.39e-5, summer 7.44e-5, other 7.58e-5 | Supported |
| CD preserves similarly small residuals | M / soft | CD mean \|r\|=2.06e-4 (larger than AR) | Rewrite: report both numbers |

### Conservative rewrite

- **Measured (AR-PCNN):** mean \|r\| = 7.46×10⁻⁵ W, std(r) = 1.15×10⁻⁵ W, max \|r\| = 1.31×10⁻⁴ W, p95 \|r\| = 7.63×10⁻⁵ W.
- **Measured (CD Multitask physics path):** mean \|r\| = 2.06×10⁻⁴ W, p95 \|r\| = 8.06×10⁻⁴ W.
- **Measured seasonal (AR):** winter / summer / other mean \|r\| = 7.39 / 7.44 / 7.58 ×10⁻⁵ W (n = 3634 / 2184 / 2952).
- **Source:** `exp03_residual_stats.csv`, `exp03_residual_by_season.csv`, `exp03_residual.json`, `exp03_residual_analysis.png`.
- **Interpretation:** Absolute residuals remain ≪ 1 W on the NSRDB physics target; qualitative irradiance/hour structure is visible in the figure but not tabulated as a regression slope.

---

## 4. Multi-seed stability

### Claim ledger

| Original statement | Type | Support | Verdict |
|--------------------|------|---------|---------|
| Section “not yet generated” | outdated | `exp04_multi_seed_runs.csv` exists with 1 row | Update |
| mean ± std across seeds | U | Only seed 42 | **Unsupported** |

### Conservative rewrite

- **Measured (seed 42 only):** NSRDB R² = 0.999999999139, MAE = 2.545×10⁻⁴ W, RMSE = 2.231×10⁻³ W; Ninja R² = 0.8639, MAE = 0.504 kW; PICP = 0.968.
- **Source:** `artifacts/phase9/csv/exp04_multi_seed_runs.csv`.
- **Not yet supported:** mean ± std across seeds {42, 123, 456, 999, 2026}. Requires training without `--skip-train`.

---

## 5. Runtime & complexity

### Claim ledger

| Original statement | Type | Support | Verdict |
|--------------------|------|---------|---------|
| Shared 128→64→32 encoder | M/config | Architecture + param counts | Supported as design fact |
| CD +545 params | C | 14339 − 13794 = 545 | Supported |
| Inference scales linearly with batch size | U | Only one protocol measured | **Unsupported** |
| AR/CD negligible overhead vs PCNN | I | Times 0.0054 / 0.0073 / 0.0080 ms/sample | Soft interpretation |
| Training times from Phase 3–6 logs | U | `training_time_sec_reported` is NaN in CSV | **Unsupported** |

### Conservative rewrite

| Model | Parameters | Inference (ms/sample) | GPU peak (MB) |
|-------|------------|------------------------|---------------|
| PCNN | 13,282 | 0.00544 | 8.77 |
| AR-PCNN | 13,794 | 0.00728 | 8.77 |
| CD-Multitask | 14,339 | 0.00802 | 8.77 |

- **Source:** `exp05_runtime_complexity.csv`, `exp05_runtime_complexity.json`, `exp05_runtime_complexity.png`.
- **Computed:** CD external head adds 545 parameters relative to AR-PCNN.
- **Not reported here:** training wall-clock (column empty) or batch-size scaling.

---

## 6. Feature importance

### Claim ledger

| Original statement | Type | Support | Verdict |
|--------------------|------|---------|---------|
| Permutation importance, ΔMAE, 5 repeats | M | Experiment design + CSVs | Supported |
| SHAP not installed | M | Runtime note | Supported |
| Top AR drivers: SZA, hour_cos, DNI | M | `exp06_importance_physics_ar.csv` | Supported |
| Irradiance/geometry dominate as expected | I | — | Soft interpretation |
| External CD ranks similar features | soft | CD top: GHI, DNI, SZA — different order from AR | Partial; rewrite |
| Shared encoder concentrates on GHI/DNI/SZA | soft | True for CD external path; AR ranks GHI last (negative ΔMAE) | Rewrite carefully |

### Conservative rewrite

- **Method:** permutation importance (mean ΔMAE over 5 shuffles) on NSRDB test features.
- **AR-PCNN physics path (top 3):** Solar Zenith Angle (ΔMAE = 2.13×10⁻³), hour_cos (1.03×10⁻³), DNI (9.35×10⁻⁷).
- **CD external head (top 3):** GHI (0.484), DNI (0.307), Solar Zenith Angle (0.300).
- **Source:** `exp06_importance_physics_ar.csv`, `exp06_importance_external_cd.csv`, `exp06_importance_physics_pcnn.csv`, matching PNGs, `exp06_feature_importance.json`.
- **Note:** Importance rankings differ between the physics path and the external head; do not claim a single shared ranking.

---

## 7. Embedding visualization

### Claim ledger

| Original statement | Type | Support | Verdict |
|--------------------|------|---------|---------|
| Before multitask: partial overlap, limited alignment | F / I | PCA/t-SNE figures only | Soft; no distance metric |
| After multitask: Ninja closer to NSRDB | F / I | figures | Soft |
| Multitask aligns feature space… | I | — | Interpretation |
| UMAP | U | `exp07_embeddings.json`: umap=false | **Unsupported** |

### Conservative rewrite

- **Measured:** Embeddings extracted for AR-PCNN and CD Multitask; each concatenated NSRDB+Ninja matrix shape (6000, 32). PCA and t-SNE 2-D projections saved.
- **Source:** `exp07_embeddings.json`, `exp07_pca_before_multitask_AR.csv`, `exp07_pca_after_multitask_CD.csv`, figures `exp07_pca_*.png`, `exp07_tsne_*.png`.
- **Not measured:** quantitative domain-alignment statistics (e.g. MMD, centroid distance). Visual statements about “closer alignment” remain qualitative figure readings only.
- **UMAP:** not generated (`umap=false`).

---

## 8. Regime-wise errors

### Claim ledger

| Original statement | Type | Support | Verdict |
|--------------------|------|---------|---------|
| Night/low-GHI: all models near-zero MAE | U for PCNN | night PCNN MAE=0.714; AR/CD ~7–9×10⁻⁵ | Rewrite |
| Afternoon/high-GHI: PCNN largest MAE | M | afternoon PCNN MAE=4.74; high_ghi=4.99 | Supported |
| AR/CD ~10⁻⁴ W | M | AR ~7×10⁻⁵; CD afternoon MAE=7.4×10⁻⁴ | Mostly; cite exact |
| Cloudy vs clear: residual models flatten gaps | M + I | clear PCNN MAE=7.09 vs cloudy 2.67; AR both ~7.3×10⁻⁵ | Soft OK with numbers |
| Summer/winter differences small for AR/CD | M | AR summer/winter MAE 7.29e-5 / 7.21e-5 | Supported |

### Conservative rewrite

Selected **measured** MAE (W) from `exp08_regime_errors.csv`:

| Regime | PCNN | AR-PCNN | CD-Multitask |
|--------|------|---------|--------------|
| afternoon | 4.738 | 7.43×10⁻⁵ | 7.35×10⁻⁴ |
| high_ghi | 4.987 | 7.62×10⁻⁵ | 1.11×10⁻⁴ |
| night | 0.714 | 7.13×10⁻⁵ | 9.01×10⁻⁵ |
| clear_sky | 7.092 | 7.63×10⁻⁵ | 1.10×10⁻⁴ |
| cloudy | 2.668 | 7.30×10⁻⁵ | 5.23×10⁻⁴ |
| summer | 2.650 | 7.29×10⁻⁵ | 1.39×10⁻⁴ |
| winter | 2.256 | 7.21×10⁻⁵ | 2.43×10⁻⁴ |

- **Source:** `exp08_regime_errors.csv`, `exp08_regime_errors.json`, `exp08_regime_errors.png`.
- **Supported summary:** On NSRDB physics targets, PCNN MAE is largest in high-irradiance / clear / afternoon regimes; AR-PCNN MAE stays ~7×10⁻⁵ W across tabulated regimes.

---

## 9. Noise robustness

### Claim ledger

| Original statement | Type | Support | Verdict |
|--------------------|------|---------|---------|
| Noise at 5/10/20% of column σ on named features | M | CSV design | Supported |
| GHI/DNI cause largest R² drop for PCNN | Partial | GHI ΔR²≈0.0086 at 20%; DNI ΔR²≈0.0003 | **GHI yes; DNI overstated** |
| AR/CD degrade more gracefully due to prior+gate | C + I | AR/CD ΔR² ≈ 0 at 20% GHI | Effect measured; mechanism is interpretation |
| Temp/wind smaller effect | M | CSV | Supported |
| Consistent with lower feature importance | I | Link to Exp 6 | Soft |
| At 20% GHI, PCNN loses more R² than CD | C | ΔR² PCNN 0.0086 vs CD ~1e-11 | Supported |

### Conservative rewrite

- **Measured ΔR² (baseline − 20% GHI noise):** PCNN 0.00858; AR-PCNN ≈ 0; CD-Multitask ≈ 1.4×10⁻¹¹.
- **Measured:** Temperature and Wind Speed noise at 20% change PCNN R² by < 0.0001.
- **DNI at 20%:** PCNN R² drops only 0.00034 — smaller than GHI; do not group DNI with GHI as equally severe.
- **Source:** `exp09_noise_robustness.csv`, `exp09_noise_robustness.json`, `exp09_noise_robustness.png`.
- **Caveat:** All targets are NSRDB physics labels; robustness here is not external-domain robustness.
- **Interpretation (optional):** Smaller degradation for AR/CD is consistent with an explicit physics prior, but this experiment does not isolate gate vs residual as the causal factor.

---

## Key strengths (re-sourced)

| Statement | Type | Support |
|-----------|------|---------|
| Phase 8: 173/173 audit checks | M | `phase8_audit.md`, `artifacts/metrics/phase8_audit.json` |
| Ablation isolates residual/gate/UQ/multitask | M | `phase7_validation.md`, `artifacts/metrics/phase7_report.json` |
| Multitask Ninja R² ~0.5 → ~0.86 | M | Phase 5 Case-1 ~0.54/0.49; Phase 6/9 CD 0.8639 (`exp01` / Phase 6 eval) |
| Gate α interpretable; PICP ~0.95 | M | Exp 2 CSV; Exp 1 PICP 0.957; Phase 5/6 UQ JSONs |

## Remaining limitations (supported)

| Statement | Type | Support |
|-----------|------|---------|
| NSRDB labels physics-derived | Prior result | `phase4_limitations.md` |
| Full-2019 ninja overlaps early train hours | Prior result | `phase6_comparison.md`, Phase 8 audit |
| λ and seed sweeps incomplete | M | Exp 1/4 CSVs have 1 row each |

## Recommendations (procedural, not empirical claims)

1. Report NSRDB R² as ≈1.0 with MAE in Watts — avoid literal `R²=1.0000` (`phase8_audit.md`).
2. Prefer Ninja val-holdout R² ≈ 0.814 (`ar_pcnn_cd_multitask_eval.json`) over full-2019 for transfer claims.
3. Do not publish multi-λ trade-off plots until Exp 1 is fully trained.
4. Disclose Phase 4 label circularity in methods/limitations.
5. Use 300 DPI figures under `data/figures/phase9/`.

---

## Output index

- CSV: `artifacts/phase9/csv/`
- Figures: `data/figures/phase9/`
- Metrics JSON: `artifacts/phase9/metrics/`
- Per-experiment notes: `artifacts/phase9/reports/`
- Claim audit (this file): `phase9_report.md`
