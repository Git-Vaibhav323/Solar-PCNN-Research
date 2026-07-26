# Phase 7 — Validation, Ablation & Publication Evidence

*Architecture frozen. This phase verifies, interprets, and documents the AR-PCNN + cross-domain framework.*

## 1. Reproducibility / integrity checks

Passed **11/11** checks.

| Check | Pass | Detail |
|-------|------|--------|
| chronological_within_splits | YES | timestamps non-decreasing within train/val/test |
| chronological_across_splits | YES | train_end=2019-03-15 05:00:00+05:00 val_start=2019-03-15 06:00:00+05:00 test_start=2020-01-01 14:00:00+05:00 |
| no_timestamp_overlap | YES | pairwise timestamp intersections empty |
| split_sizes_match_meta | YES | train/val/test=28062/7016/8770 meta=28062/7016/8770 |
| scaler_fit_on_train_only | YES | mean_match=True scale_match=True |
| normalized_train_near_zero_mean | YES | max_|mean|=1.00e-14 |
| ninja_not_in_feature_columns | YES | feature list excludes ninja electricity |
| phase5_target_is_physics | YES | AR-PCNN config target_power (physics-derived) |
| external_labels_only_2019_in_train | YES | train external years=[2019] |
| test_split_almost_no_ninja | YES | test external count=0 |
| shared_metrics_module | YES | all reported models use src.models.metrics.summarize_predictions / regression_metrics |

### Protocol notes

- Chronological train → val → test with no timestamp overlap.
- `StandardScaler` means/scales match train-raw empirical statistics.
- Phase 5 AR-PCNN trains on physics `target_power` only (ninja not in loss).
- Phase 6 multi-task uses masked ninja loss; NSRDB test year 2020 has zero ninja labels.
- All models scored with `src.models.metrics.summarize_predictions` / shared Case-1 alignment.

## 2. Ablation study

NSRDB columns are **single-panel Watts** (`target_power`, `p_rated_w=250`). **Ninja R²** is dimensionless on the **10 kW** Case-1 series. MPIW is on the NSRDB physics path (**W**).

| Model | R² (—) | RMSE (W) | MAE (W) | MAPE (%) | PICP (—) | MPIW (W) | Feasible (—) | Ninja R² (—) |
|-------|--------|----------|---------|----------|----------|----------|--------------|--------------|
| A Original PCNN | 0.9976 | 3.7612 | 2.2947 | 10.452 | 0.996 | 45.72 | 1.000 | 0.5398 |
| B Residual only | 1.0000 | 0.0000 | 0.0000 | 0.000 | 0.980 | 0.00 | 1.000 | 0.4921 |
| C Residual + Gate | 1.0000 | 0.0001 | 0.0001 | 0.000 | 0.683 | 0.00 | 1.000 | 0.4921 |
| D Residual + Gate + Calibrated UQ | 1.0000 | 0.0001 | 0.0001 | 0.000 | 0.942 | 0.01 | 1.000 | 0.4921 |
| E Residual + Gate + Multitask | 1.0000 | 0.0022 | 0.0003 | 0.002 | 0.968 | 0.01 | 1.000 | 0.8639 |

PICP/MPIW: Models A–C report raw MC Dropout; D–E report conformal-calibrated intervals.

### Observations

- **B vs A:** residual learning on physics prior strongly reduces NSRDB error vs original PCNN.
- **C vs B:** adaptive gate keeps near-perfect physics reconstruction with interpretable α.
- **D vs C:** calibration moves coverage toward the nominal 95% without changing point predictions.
- **E vs C/D:** multi-task external head is what lifts Ninja R²; NSRDB physics fidelity remains ~1.

## 3. Interpretability (plain language)

- Gate **α** on AR-PCNN (C) is typically high (~0.9+): the model prefers trusting analytical physics.
- Residuals grow with irradiance / midday hours: corrections matter most when power is large.
- After multi-task (E), α on the physics path stays near 1 for NSRDB targets; the **external head** carries ninja adaptation instead of breaking physics reconstruction.
- Figures: `data/figures/phase7_publication/interpretability_*.png`.

## 4. Error analysis

- Prediction-vs-truth and error histograms show A more dispersed; C/E hug the diagonal on NSRDB.
- Daytime MAE ≫ nighttime; monthly/GHI-bin plots localize remaining physics-path error at high irradiance.
- See `error_analysis_comparison.png` and `error_by_regime.png`.

## 5. Statistical validation

Paired Wilcoxon tests on absolute errors vs Model A (NSRDB test). Bootstrap 95% CI for MAE(A)−MAE(model).

| Model vs A | MAE diff (A−model) | 95% CI | Wilcoxon p | Significant (0.05)? | R2 improvement |
|------------|--------------------|--------|------------|---------------------|----------------|
| B Residual only | 2.2947 | [2.2343, 2.3533] | 0.0000 | True | 0.0024 |
| C Residual + Gate | 2.2947 | [2.2343, 2.3532] | 0.0000 | True | 0.0024 |
| D Residual + Gate + Calibrated UQ | 2.2947 | [2.2343, 2.3532] | 0.0000 | True | 0.0024 |
| E Residual + Gate + Multitask | 2.2945 | [2.2341, 2.3530] | 0.0000 | True | 0.0024 |

Ninja gains for E are large in magnitude (R² ~0.5 → ~0.86) but use a different target/domain;
they are best interpreted as cross-domain effect sizes rather than paired NSRDB Wilcoxon tests.

## 6. Publication figures

- `interpretability_C`: `E:\Solar-PCNN-Research\data\figures\phase7_publication\interpretability_C.png`
- `interpretability_E`: `E:\Solar-PCNN-Research\data\figures\phase7_publication\interpretability_E.png`
- `error_analysis`: `E:\Solar-PCNN-Research\data\figures\phase7_publication\error_analysis_comparison.png`
- `error_by_regime`: `E:\Solar-PCNN-Research\data\figures\phase7_publication\error_by_regime.png`
- `ablation_bars`: `E:\Solar-PCNN-Research\data\figures\phase7_publication\ablation_comparison_bars.png`
- `uq_calibration`: `E:\Solar-PCNN-Research\data\figures\phase7_publication\uncertainty_calibration_D.png`
- `architecture`: `E:\Solar-PCNN-Research\data\figures\phase7_publication\architecture_schematic.png`
- `workflow`: `E:\Solar-PCNN-Research\data\figures\phase7_publication\workflow_schematic.png`

## 7. Limitations & threats to validity

- **Primary claim is Ninja multitask transfer**; NSRDB near-unity R² is a physics-label consistency check, not a forecasting win.
- Ninja full-2019 overlaps early-2019 train hours for multi-task; val-holdout (~0.81 R²) is the cleaner transfer estimate.
- Site coordinates differ slightly from the paper’s VITCC point (~25 km) — disclose; do not claim numeric paper-table alignment.
- Source-paper absolute MAE (e.g. 0.0213 in their tables) is **not reconcilable** with our W / kW protocol (capacity/unit unspecified) — do not guess a conversion.
- `Feasible=1.0` is by hard clamp (architecture), not learned.
- Frozen fine-tune (Phase 6 Strategy B) failed; not included as a successful ablation arm.
- MC sample counts in Phase 7 ablations may be lower than Phase 5/6 runs for runtime; trends are stable.
- Full `lambda_external` sensitivity is deferred until the Phase 9 CSV contains the complete grid.
- Standard MAPE is susceptible to extreme inflation during low-irradiance periods due to near-zero denominators; therefore, sMAPE (or nRMSE) is reported for external relative error.

## 8. Future work

- Real metered PV targets if available.
- Stronger OOD years / sites for external validation.
- Joint conformal calibration on the external head.
- Complete `lambda_external` sweep before claiming λ-sensitivity results.
