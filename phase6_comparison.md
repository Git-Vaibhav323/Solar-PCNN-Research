# Phase 6 Comparison — Cross-Domain Learning

Architecture: AR-PCNN physics path unchanged. Phase 6 changes **training** only
(multi-task shared encoder + external head, and encoder-frozen fine-tune).

## NSRDB test (physics target, Watts)

| Metric | PCNN | AR-PCNN | CD Multitask | CD Finetune |
|--------|------|---------|--------------|-------------|
| MAE | 2.2947 | 0.0001 | 0.0003 | 0.0001 |
| RMSE | 3.7612 | 0.0001 | 0.0022 | 0.0001 |
| R2 | 0.9976 | 1.0000 | 1.0000 | 1.0000 |
| MAPE % | 10.452 | 0.000 | 0.002 | 0.000 |
| Feasible | 1.000 | 1.000 | 1.000 | 1.000 |
| Mean alpha | — | 0.937 | 1.000 | 0.937 |

## Renewables.ninja full 2019 (10 kW)

| Metric | PCNN (phys scaled) | AR-PCNN (phys scaled) | CD Multitask (external head) | CD Finetune (external head) |
|--------|--------------------|-----------------------|------------------------------|-----------------------------|
| MAE kW | 0.8736 | 0.9125 | 0.5044 | 1.6320 |
| RMSE kW | 1.5429 | 1.6209 | 0.8391 | 2.7993 |
| R2 | 0.5398 | 0.4921 | 0.8639 | -0.5149 |

## Ninja holdout inside validation window (2019 hours in val split)

| Strategy | n | R2 | MAE kW |
|----------|---|----|--------|
| Multitask | 7007 | 0.8143 | 0.5490 |
| Finetune | 7007 | -0.5104 | 1.4860 |

Note: early-2019 ninja hours overlap the train window; the val-window subset is the cleaner transfer check. Full-2019 numbers match the Phase-5 Case-1 protocol for continuity.

## Uncertainty (nominal 95%)

| Model | PICP | MPIW |
|-------|------|------|
| PCNN raw MC | 0.999 | 46.19 |
| AR-PCNN calibrated | 0.944 | 0.02 |
| CD Multitask calibrated | 0.963 | 0.01 |
| CD Finetune calibrated | 0.944 | 0.02 |
