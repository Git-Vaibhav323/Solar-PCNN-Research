# Phase 5 Comparison — Baseline PCNN vs AR-PCNN

Identical Phase-2 splits, preprocessing, and evaluation protocol.

**Headline for publication:** use Phase 6 Ninja multitask transfer, not these NSRDB scores. NSRDB targets are physics-derived (`target_power` ≈ `P_bounded`), so near-perfect R² here is a consistency check.

## NSRDB test (single panel, Watts) — physics-fidelity sanity check

| Metric | Baseline PCNN | AR-PCNN |
|--------|---------------|---------|
| MAE (W) | 2.2947 | 0.0001 |
| RMSE (W) | 3.7612 | 0.0001 |
| R² (—) | 0.9976 | ≈1.0 |
| MAPE (%) | 10.452 | 0.000 |
| Feasible (—) | 1.000 | 1.000 |
| Mean alpha (—) | — | 0.937 |
| Mean |r| (W) | — | 0.000 |

`Feasible = 1.0` is guaranteed by the hard capacity clamp in the forward pass (architectural constraint), not a learned achievement.

## Case 1 — Renewables.ninja (10 kW) — primary external metric for Phase 5

| Metric | Baseline PCNN | AR-PCNN |
|--------|---------------|---------|
| MAE (kW) | 0.8736 | 0.9125 |
| RMSE (kW) | 1.5429 | 1.6209 |
| R² (—) | 0.5398 | 0.4921 |
| **sMAPE (%)** | 133.54 | 133.14 |
| **nRMSE (%)** | 94.54 | 99.32 |
| MAPE (%) | 166.62 | 171.76 |

Standard MAPE is susceptible to extreme inflation during low-irradiance periods due to near-zero denominators; therefore, sMAPE (or nRMSE) is reported to provide a more stable relative error metric across all generation regimes. MAPE is retained only for continuity with earlier tables.

**Units:** NSRDB absolute errors are **Watts** (1× TS250 panel, `p_rated_w=250`). Case-1 absolute errors are **kW** for a **10 kW** system (40×250 W). Do not mix rows across tables. The source paper’s absolute-error figures (e.g. MAE≈0.0213 in their tables) are **not reconcilable** with ours: system capacity / unit for those published numbers is not specified clearly enough to convert — we do not guess `0.0213 kW ↔ 21.3 W`.

## Uncertainty (test, nominal 95%) — intervals on NSRDB physics path (Watts)

| Setting | PICP (—) | MPIW (W) |
|---------|----------|----------|
| Baseline MC Dropout | 0.999 | 46.19 |
| AR-PCNN MC Dropout (raw) | 0.746 | 0.00 |
| AR-PCNN + conformal scale | 0.944 | 0.02 |

Calibration scale on val → test: **4.392**
