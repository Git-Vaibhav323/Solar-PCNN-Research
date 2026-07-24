## Gate (α) interpretability

- **AR-PCNN:** mean α ≈ 0.937. During daylight (GHI>50), α clusters near 1 — the model trusts analytical physics.
- **Night / low irradiance:** α spreads below 1 when GHI≈0; the gate down-weights physics where the prior is near zero.
- **CD Multitask:** mean α ≈ 1.000 on the NSRDB physics path — multitask training preserves physics trust.
- **Seasonal / diurnal:** Hourly α is highest at midday; monthly variation is modest.
- **Temperature / DNI:** No strong monotonic α trend; gate responds primarily to irradiance regime.
