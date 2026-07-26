## Feature importance

- Method: **permutation importance** (ΔMAE, 5 repeats). Physics branches use NSRDB targets; `external_cd` uses the multitask external head on the same loader features.
- SHAP not installed; used permutation importance.
- AR-PCNN **physics-path** permutation ΔMAE is at noise scale (max |ΔMAE|≈2.13e-03 W) because NSRDB residuals are already ~10⁻⁴ W. **Do not interpret ranked features on this branch as real drivers.**
- **Meaningful signal — external CD head:** top drivers **GHI, DNI, Solar Zenith Angle** (ΔMAE ≈ 0.48, 0.31, 0.30). Irradiance / solar geometry dominate the ninja kW head, as expected.
