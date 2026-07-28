# Phase 9 — Publication Enhancement & Advanced Validation

*Generated: 2026-07-26 23:26*

Architecture frozen (Phases 3–8). All new experiments write to `artifacts/phase9/` and `data/figures/phase9/`.

---

## 1. Hyperparameter sensitivity

## Hyperparameter sensitivity (lambda_external)

- Swept λ ∈ [0.0, 0.05, 0.1, 0.25, 0.5, 1.0] with frozen architecture; each run warm-starts from Phase-5 AR-PCNN.
- **Physics fidelity:** NSRDB R² stays ≈1 across λ; lowest NSRDB MAE at λ=0.0 (MAE=8.00633e-06 W) — expected when the external loss is off.
- **Cross-domain:** Ninja R² peaks at λ=0.25 (R²=0.8710); λ∈{0.05…1.0} all ≈0.86–0.87. **λ=0 → Ninja R²≈−0.51** (external head never trained — control, not a seed failure).
- **Trade-off:** Any λ>0 is enough to learn the ninja head; λ≈0.25–1.0 keeps physics MAE tiny while holding Ninja R². PICP is weakest at very low λ (0.05–0.1) before conformal coverage recovers near λ≥0.25.
- **UQ:** PICP after conformal scaling is listed per λ in the CSV.

---

## 2. Gate interpretability

## Gate (α) interpretability

- **AR-PCNN:** mean α ≈ 0.937. During daylight (GHI>50), α clusters near 1 — the model trusts analytical physics.
- **Night / low irradiance:** α spreads below 1 when GHI≈0; the gate down-weights physics where the prior is near zero.
- **CD Multitask:** mean α ≈ 1.000 on the NSRDB physics path — multitask training preserves physics trust.
- **Seasonal / diurnal:** Hourly α is highest at midday; monthly variation is modest.
- **Temperature / DNI:** No strong monotonic α trend; gate responds primarily to irradiance regime.

---

## 3. Residual analysis

## Residual analysis

- Residuals on NSRDB are **tiny** (mean |r| ≈ 7.46e-05 W, 95th pct ≈ 7.63e-05 W) because the target is physics-derived.
- |r| grows slightly with irradiance and midday hours — corrections matter most when power is large.
- Monthly and seasonal means are stable; no strong winter/summer bias in the residual branch.
- CD multitask mean |r| ≈ 2.06e-04 W (vs AR 7.46e-05 W). Multitask can inflate physics-path error in some regimes (see Exp 8 afternoon / medium_ghi RMSE) — a real cost of sharing the encoder with the external head, not identical fidelity.

---

## 4. Multi-seed stability

## Multi-seed stability

Seeds evaluated: [42, 123, 456, 999, 2026]

NSRDB absolute errors are **Watts** (1× TS250). Ninja absolute errors are **kW** (10 kW system).

**λ verification (raw artifacts, not this markdown):** every seed run used lambda_external=1.0.
Evidence: artifacts/phase9/configs/train_seed*.yaml and machine-written
artifacts/phase9/metrics/summary_seed*.json (field lambda_external). The runs CSV
also records lambda_external per row. There is **no** multi-seed sweep at λ=0.25 —
only Exp1 varies λ (fixed seed=42 × 6 λ values).

| Metric | mean ± std |
|--------|------------|
| NSRDB R² (—) | 1.000000 ± 0.000000 |
| NSRDB MAE (W) | 0.00138945 ± 0.00252742 |
| Ninja R² (—) | 0.8772 ± 0.0140 |
| Ninja MAE (kW) | 0.4329 ± 0.0459 |
| PICP (—) | 0.944 ± 0.028 |

All 5 seeds succeed on Ninja (per-seed R²: 42→0.864, 123→0.899, 456→0.867, 999→0.881, 2026→0.876). Do not confuse this with the separate frozen-finetune failure (R²≈−0.51).

---

## 5. Runtime & complexity

## Runtime & complexity

- All three models share the same 128→64→32 encoder scale; CD adds only a small external head (+545 params).
- Inference latency scales linearly with batch size; AR-PCNN/CD add negligible overhead vs PCNN on GPU.
- Training times are taken from original Phase 3–6 run summaries where available.

---

## 6. Feature importance

## Feature importance

- Method: **permutation importance** (ΔMAE, 5 repeats). Physics branches use NSRDB targets; `external_cd` uses the multitask external head on the same loader features.
- SHAP not installed; used permutation importance.
- AR-PCNN **physics-path** permutation ΔMAE is at noise scale (max |ΔMAE|≈2.13e-03 W) because NSRDB residuals are already ~10⁻⁴ W. **Do not interpret ranked features on this branch as real drivers.**
- **Meaningful signal — external CD head:** top drivers **GHI, DNI, Solar Zenith Angle** (ΔMAE ≈ 0.48, 0.31, 0.30). Irradiance / solar geometry dominate the ninja kW head, as expected.

---

## 7. Embedding visualization

## Embedding visualization

- **Before multitask (AR-PCNN):** NSRDB and Ninja embeddings partially overlap but retain domain-specific clusters — limited alignment.
- **After multitask (CD):** Shared encoder pulls Ninja embeddings closer to NSRDB meteorological structure while preserving domain labels in t-SNE/UMAP.
- **Interpretation:** Multitask learning aligns feature space for cross-domain transfer without collapsing physics-specific structure.

---

## 8. Regime-wise errors

## Regime-wise error analysis

- **Night / low-GHI:** All models show near-zero absolute error (targets near zero).
- **High-GHI / afternoon:** Baseline PCNN MAE is largest; AR-PCNN stays ~10⁻⁴ W; CD MAE stays small but RMSE can degrade (encoder shared with ninja head).
- **Cloudy vs clear:** Residual models flatten regime gaps vs PCNN; multitask is not free on every regime.
- **Seasonal:** Summer/winter differences remain small in MAE for AR/CD on the NSRDB physics target.
- **Multitask cost (disclose):** CD physics-path RMSE can be much larger than AR-PCNN in some regimes despite still-small MAE — afternoon RMSE 4.46e-03 vs 7.46e-05; medium_ghi RMSE 5.90e-03 vs 7.46e-05. Do not claim identical physics residuals after multitask.

---

## 9. Noise robustness

## Noise robustness

- Gaussian noise injected on normalized test features at 5%, 10%, 20% of column σ.
- **GHI/DNI** perturbations cause the largest R² drop for PCNN; AR-PCNN and CD degrade more gracefully due to physics prior + gate.
- **Temperature / wind** noise has smaller effect — consistent with lower feature importance.
- At 20% GHI noise, baseline PCNN loses more absolute R² than multitask CD, supporting robustness claims with caveats (still NSRDB physics target).

---

## Key strengths

- **Primary result:** Multitask CD lifts external Ninja R² ~0.5 → ~0.86 (val-holdout ~0.81) with masked-loss integrity checks.
- Reproducible metrics verified in Phase 8 (independent recalc from arrays).
- Ablation isolates residual, gate, calibration, and multitask contributions.
- Gate α interpretable; conformal UQ reaches ~95% PICP.

## Remaining limitations

- NSRDB labels are physics-derived; near-unity R² is a **consistency check**, not a forecasting claim.
- Full-2019 ninja evaluation overlaps early-2019 train hours for multitask — prefer val-holdout.
- λ=0 leaves the external head untrained (Ninja R² negative) — expected control; not a multitask failure.
- Feasibility / hard physics clamps are architectural constraints (true by construction), not learned achievements.

## Recommendations for publication

1. Lead with Ninja multitask transfer (val-holdout ~0.81); demote NSRDB to physics-fidelity sanity check.
2. Report NSRDB as “≈1.0” + MAE (W) — avoid literal `R²=1.0000`.
3. Include the λ-sensitivity figure; note Ninja peaks near λ≈0.25 and λ=0 is an untrained-head control.
4. Disclose Phase 4 circular-label threat and clamp-by-construction feasibility.
5. Do not align numeric tables to the source paper without resolved unit/protocol parity.
6. Use figures from `data/figures/phase9/` at 300 DPI; prefer external-head importance over physics-path noise.

## Output index

- CSV: `artifacts\phase9\csv`
- Figures: `data\figures\phase9`
- Metrics JSON: `artifacts\phase9\metrics`
- Per-experiment notes: `artifacts\phase9\reports`
