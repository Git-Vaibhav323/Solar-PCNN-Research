## Residual analysis

- Residuals on NSRDB are **tiny** (mean |r| ≈ 7.46e-05 W, 95th pct ≈ 7.63e-05 W) because the target is physics-derived.
- |r| grows slightly with irradiance and midday hours — corrections matter most when power is large.
- Monthly and seasonal means are stable; no strong winter/summer bias in the residual branch.
- CD multitask mean |r| ≈ 2.06e-04 W (vs AR 7.46e-05 W). Multitask can inflate physics-path error in some regimes (see Exp 8 afternoon / medium_ghi RMSE) — a real cost of sharing the encoder with the external head, not identical fidelity.
