## Feature importance

- Method: **permutation importance** (ΔMAE on NSRDB test, 5 repeats).
- SHAP not installed; used permutation importance.
- Top drivers for AR-PCNN physics path: **Solar Zenith Angle, hour_cos, DNI** — irradiance and solar geometry dominate, as expected for PV.
- External CD head ranks similar meteorological features; shared encoder concentrates on GHI/DNI/SZA.
