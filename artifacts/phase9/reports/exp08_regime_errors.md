## Regime-wise error analysis

- **Night / low-GHI:** All models show near-zero absolute error (targets near zero).
- **High-GHI / afternoon:** Baseline PCNN MAE is largest; AR-PCNN stays ~10⁻⁴ W; CD MAE stays small but RMSE can degrade (encoder shared with ninja head).
- **Cloudy vs clear:** Residual models flatten regime gaps vs PCNN; multitask is not free on every regime.
- **Seasonal:** Summer/winter differences remain small in MAE for AR/CD on the NSRDB physics target.
- **Multitask cost (disclose):** CD physics-path RMSE can be much larger than AR-PCNN in some regimes despite still-small MAE — afternoon RMSE 4.46e-03 vs 7.46e-05; medium_ghi RMSE 5.90e-03 vs 7.46e-05. Do not claim identical physics residuals after multitask.
