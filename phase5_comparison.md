# Phase 5 Comparison — Baseline PCNN vs AR-PCNN

Identical Phase-2 splits, preprocessing, and evaluation protocol.

## NSRDB test (single panel, Watts)

| Metric | Baseline PCNN | AR-PCNN |
|--------|---------------|---------|
| MAE | 2.2947 | 0.0001 |
| RMSE | 3.7612 | 0.0001 |
| R² | 0.9976 | 1.0000 |
| MAPE (%) | 10.452 | 0.000 |
| Feasible | 1.000 | 1.000 |
| Mean alpha | — | 0.937 |
| Mean |r| (W) | — | 0.000 |

## Case 1 — Renewables.ninja (10 kW)

| Metric | Baseline PCNN | AR-PCNN |
|--------|---------------|---------|
| MAE (kW) | 0.8736 | 0.9125 |
| RMSE (kW) | 1.5429 | 1.6209 |
| R² | 0.5398 | 0.4921 |
| MAPE (%) | — | 171.763 |

## Uncertainty (test, nominal 95%)

| Setting | PICP | MPIW |
|---------|------|------|
| Baseline MC Dropout | 0.999 | 46.19 |
| AR-PCNN MC Dropout (raw) | 0.746 | 0.00 |
| AR-PCNN + conformal scale | 0.944 | 0.02 |

Calibration scale on val → test: **4.392**
