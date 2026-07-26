## Multi-seed stability

Seeds evaluated: [42, 123, 456, 999, 2026]

NSRDB absolute errors are **Watts** (1x TS250). Ninja absolute errors are **kW** (10 kW system).

| Metric | mean +/- std |
|--------|------------|
| NSRDB R2 (-) | 1.000000 +/- 0.000000 |
| NSRDB MAE (W) | 0.00138945 +/- 0.00252742 |
| Ninja R2 (-) | 0.8772 +/- 0.0140 |
| Ninja MAE (kW) | 0.4329 +/- 0.0459 |
| PICP (-) | 0.944 +/- 0.028 |

All 5 seeds succeed on Ninja (per-seed R2: 42->0.864, 123->0.899, 456->0.867, 999->0.881, 2026->0.876). Do not confuse this with the separate frozen-finetune failure (R2~-0.51).
