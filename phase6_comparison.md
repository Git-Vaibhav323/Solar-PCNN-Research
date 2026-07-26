# Phase 6 Comparison — Cross-Domain Learning

**Primary contribution:** multitask external-head transfer to Renewables.ninja (R² ~0.5 → **0.86** full-year; **0.81** val-holdout).

Architecture: AR-PCNN physics path unchanged. Phase 6 changes **training** only
(multi-task shared encoder + external head, and encoder-frozen fine-tune).

NSRDB rows below are a **physics-fidelity consistency check** (`target_power` is physics-derived). Do not lead a paper with NSRDB R²≈1.0.

## NSRDB test (physics target, Watts) — consistency only

| Metric | PCNN | AR-PCNN | CD Multitask | CD Finetune |
|--------|------|---------|--------------|-------------|
| MAE (W) | 2.2947 | 0.0001 | 0.0003 | 0.0001 |
| RMSE (W) | 3.7612 | 0.0001 | 0.0022 | 0.0001 |
| R² (—) | 0.9976 | ≈1.0 | ≈1.0 | ≈1.0 |
| MAPE (%) | 10.452 | 0.000 | 0.002 | 0.000 |
| Feasible (—) | 1.000 | 1.000 | 1.000 | 1.000 |
| Mean alpha (—) | — | 0.937 | 1.000 | 0.937 |

`Feasible = 1.0` is by construction (hard clamp). CD RMSE on NSRDB is slightly worse than AR-PCNN in some regimes — a multitask cost (see Phase 9 Exp 8).

## Renewables.ninja full 2019 (10 kW) — primary result

| Metric | PCNN (phys scaled) | AR-PCNN (phys scaled) | CD Multitask (external head) | CD Finetune (external head) |
|--------|--------------------|-----------------------|------------------------------|-----------------------------|
| MAE (kW) | 0.8736 | 0.9125 | 0.5044 | 1.6320 |
| RMSE (kW) | 1.5429 | 1.6209 | 0.8391 | 2.7993 |
| R² (—) | 0.5398 | 0.4921 | **0.8639** | -0.5149 |
| **sMAPE (%)** | 133.54 | 133.14 | **108.24** | 200.00 |
| **nRMSE (%)** | 94.54 | 99.32 | **51.41** | 171.53 |
| MAPE (%) | 166.62 | 171.76 | 79.81 | 100.00 |

Standard MAPE is susceptible to extreme inflation during low-irradiance periods due to near-zero denominators; therefore, sMAPE (or nRMSE) is reported to provide a more stable relative error metric across all generation regimes.

**Units:** NSRDB absolute errors are **Watts** (1× TS250). Ninja absolute errors are **kW** on a **10 kW** system (40×250 W). Source-paper absolute MAE figures are **not reconcilable** with ours (capacity/unit unspecified) — no guessed conversion.

## Ninja holdout inside validation window (2019 hours in val split)

| Strategy | n (h) | R² (—) | MAE (kW) |
|----------|-------|--------|----------|
| Multitask | 7007 | **0.8143** | 0.5490 |
| Finetune | 7007 | -0.5104 | 1.4860 |

Note: early-2019 ninja hours overlap the train window; the val-window subset is the cleaner transfer check. Full-2019 numbers match the Phase-5 Case-1 protocol for continuity.

**Negative result (keep):** frozen-encoder fine-tune fails (R² negative). Mechanism: external-head training under freeze can trap near a zero-ish clamp regime — see `src/analysis/phase9/train_helpers.py` comments.

## Uncertainty (nominal 95%) — NSRDB physics path

| Model | PICP (—) | MPIW (W) |
|-------|----------|----------|
| PCNN raw MC | 0.999 | 46.19 |
| AR-PCNN calibrated | 0.944 | 0.02 |
| CD Multitask calibrated | 0.963 | 0.01 |
| CD Finetune calibrated | 0.944 | 0.02 |
