## Multi-seed stability

Seeds evaluated: [42, 123, 456, 999, 2026]

NSRDB absolute errors are **Watts** (1× TS250). Ninja absolute errors are **kW** (10 kW system).

**λ verification (raw artifacts, not this markdown):** every seed run used lambda_external=1.0.
Evidence: artifacts/phase9/configs/train_seed*.yaml and machine-written
artifacts/phase9/metrics/summary_seed*.json (field lambda_external). The runs CSV
now also records lambda_external per row. There is **no** multi-seed sweep at λ=0.25 —
only Exp1 varies λ (fixed seed=42 × 6 λ values).

| Metric | mean ± std |
|--------|------------|
| NSRDB R² (—) | 1.000000 ± 0.000000 |
| NSRDB MAE (W) | 0.00138945 ± 0.00252742 |
| Ninja R² (—) | 0.8772 ± 0.0140 |
| Ninja MAE (kW) | 0.4329 ± 0.0459 |
| PICP (—) | 0.944 ± 0.026 |

All 5 seeds succeed on Ninja (per-seed R²: 42→0.864, 123→0.899, 456→0.867, 999→0.881, 2026→0.876). Do not confuse this with the separate frozen-finetune failure (R²≈−0.51).
