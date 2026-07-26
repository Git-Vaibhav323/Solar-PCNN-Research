# Phase 8 — Scientific Audit of Experimental Results

*Verification only. Models were not modified.*

**Generated:** audit of Phase 5–7 artifacts under `artifacts/metrics/`.

## Verdict

**Publication readiness:** READY with caveats  
Checks passed: **178/178**  
Critical failures: **0** | Major failures: **0**

### Executive summary

1. All Phase 5–7 headline metrics recalculate from saved prediction arrays (not hard-coded literals).
2. Comparison tables in `Documentation/phase5_comparison.md`, `Documentation/phase6_comparison.md`, and `Documentation/phase7_validation.md` match recalculated values within rounding.
3. No train/val/test timestamp overlap; scaler is train-only; multitask ninja loss is masked; NSRDB test has zero ninja labels.
4. Phase 7 sklearn / NaN warnings do **not** affect scientific validity.
5. Caveats for reviewers: physics-derived NSRDB labels; full-2019 ninja overlaps early train hours (prefer val-holdout ≈ 0.81 R²).

## 1. Metrics sourced from saved arrays (not hard-coded)

| Check | Pass | Detail |
|-------|------|--------|
| metrics_sourced_from_code:evaluate.py | PASS | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:evaluate_ar_pcnn.py | PASS | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:case1_ninja.py | PASS | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:case1_ar_pcnn.py | PASS | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:evaluate_ar_pcnn_cd.py | PASS | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:uncertainty.py | PASS | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:uncertainty_ar_pcnn.py | PASS | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:phase7_run.py | PASS | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:compare_phase5.py | PASS | summarize/regression=False load/save_arrays=True |
| metrics_sourced_from_code:compare_phase6.py | PASS | summarize/regression=False load/save_arrays=True |
| phase5_7_docs_are_derived_artifacts | PASS | comparison JSON artifacts exist alongside markdown tables |

Point metrics in Phase 5–7 were produced by `summarize_predictions` / equivalent on saved `y_true`/`y_pred` arrays (or Case-1 equivalents). Comparison markdowns are generated from JSON summaries of those runs.

## 2. Independent recalculation vs saved JSON

| Check | Pass | Detail |
|-------|------|--------|
| metrics_sourced_from_code:phase7_run.py | PASS | summarize/regression=True load/save_arrays=True |
| phase3_pcnn_nsrdb_test_mae_matches_json | PASS | recalc=2.29473 reported=2.2947349637050176 Δ=8.87e-09 |
| phase3_pcnn_nsrdb_test_rmse_matches_json | PASS | recalc=3.76123 reported=3.761233935951727 Δ=0 |
| phase3_pcnn_nsrdb_test_r2_matches_json | PASS | recalc=0.997553 reported=0.9975527854333056 Δ=0 |
| phase3_pcnn_nsrdb_test_mape_matches_json | PASS | recalc=10.4521 reported=10.45208070811051 Δ=0 |
| phase5_ar_pcnn_nsrdb_test_mae_matches_json | PASS | recalc=7.28434e-05 reported=7.284336416112849e-05 Δ=5.37e-12 |
| phase5_ar_pcnn_nsrdb_test_rmse_matches_json | PASS | recalc=7.37106e-05 reported=7.371064790110818e-05 Δ=0 |
| phase5_ar_pcnn_nsrdb_test_r2_matches_json | PASS | recalc=1 reported=0.9999999999990601 Δ=0 |
| phase5_ar_pcnn_nsrdb_test_mape_matches_json | PASS | recalc=0.000302044 reported=0.0003020439393663318 Δ=0 |
| phase5_case1_pcnn_mae_matches_json | PASS | recalc=0.873558 reported=0.8735581319551258 Δ=0 |
| phase5_case1_pcnn_rmse_matches_json | PASS | recalc=1.54292 reported=1.5429226314288524 Δ=0 |
| phase5_case1_pcnn_r2_matches_json | PASS | recalc=0.539788 reported=0.5397878981046011 Δ=0 |
| phase5_case1_pcnn_mape_matches_json | PASS | recalc=166.622 reported=166.62194856458473 Δ=0 |
| phase5_case1_ar_pcnn_mae_matches_json | PASS | recalc=0.912467 reported=0.9124674422228375 Δ=0 |
| phase5_case1_ar_pcnn_rmse_matches_json | PASS | recalc=1.62086 reported=1.6208554516340703 Δ=0 |
| phase5_case1_ar_pcnn_r2_matches_json | PASS | recalc=0.492123 reported=0.4921232773293678 Δ=0 |
| phase5_case1_ar_pcnn_mape_matches_json | PASS | recalc=171.763 reported=171.76252667871975 Δ=0 |
| phase7_B_residual_mae_matches_json | PASS | recalc=6.74829e-06 reported=6.748289039768406e-06 Δ=2.12e-13 |
| phase7_B_residual_rmse_matches_json | PASS | recalc=1.05558e-05 reported=1.0555793703741656e-05 Δ=0 |
| phase7_B_residual_r2_matches_json | PASS | recalc=1 reported=0.9999999999999807 Δ=0 |
| phase7_B_residual_mape_matches_json | PASS | recalc=6.82628e-05 reported=6.826278971217283e-05 Δ=0 |
| cd_multitask_nsrdb_mae | PASS | recalc=0.0002545 reported=0.00025450040243999176 |
| cd_multitask_nsrdb_rmse | PASS | recalc=0.00223126 reported=0.0022312574555698526 |
| cd_multitask_nsrdb_r2 | PASS | recalc=1 reported=0.9999999991387867 |
| cd_multitask_nsrdb_mape | PASS | recalc=0.00185166 reported=0.0018516639166036345 |
| cd_multitask_ninja_mae | PASS | recalc=0.504367 reported=0.5043669195172209 |
| cd_multitask_ninja_rmse | PASS | recalc=0.839053 reported=0.8390529478745805 |
| cd_multitask_ninja_r2 | PASS | recalc=0.863903 reported=0.8639030252917119 |
| cd_finetune_nsrdb_mae | PASS | recalc=7.28434e-05 reported=7.284336416112849e-05 |
| cd_finetune_nsrdb_rmse | PASS | recalc=7.37106e-05 reported=7.371064790110818e-05 |
| cd_finetune_nsrdb_r2 | PASS | recalc=1 reported=0.9999999999990601 |
| cd_finetune_nsrdb_mape | PASS | recalc=0.000302044 reported=0.0003020439393663318 |
| cd_finetune_ninja_mae | PASS | recalc=1.632 reported=1.6320034266133638 |
| cd_finetune_ninja_rmse | PASS | recalc=2.79934 reported=2.7993373669297537 |
| cd_finetune_ninja_r2 | PASS | recalc=-0.514887 reported=-0.5148869313126689 |
| phase7_A_pcnn_mae_vs_report_json | PASS | recalc=2.29473 report=2.294734990928097 |
| phase7_A_pcnn_rmse_vs_report_json | PASS | recalc=3.76123 report=3.761234036969936 |
| phase7_A_pcnn_r2_vs_report_json | PASS | recalc=0.997553 report=0.9975527853018523 |
| phase7_A_pcnn_mape_vs_report_json | PASS | recalc=10.4521 report=10.45208090460635 |
| phase7_B_residual_only_mae_vs_report_json | PASS | recalc=6.74829e-06 report=6.748289039768406e-06 |
| phase7_B_residual_only_rmse_vs_report_json | PASS | recalc=1.05558e-05 report=1.0555793703741656e-05 |
| phase7_B_residual_only_r2_vs_report_json | PASS | recalc=1 report=0.9999999999999807 |
| phase7_B_residual_only_mape_vs_report_json | PASS | recalc=6.82628e-05 report=6.826278971217283e-05 |
| phase7_C_residual_gate_mae_vs_report_json | PASS | recalc=7.28434e-05 report=7.284336397736285e-05 |
| phase7_C_residual_gate_rmse_vs_report_json | PASS | recalc=7.37106e-05 report=7.371064774159155e-05 |
| phase7_C_residual_gate_r2_vs_report_json | PASS | recalc=1 report=0.9999999999990601 |
| phase7_C_residual_gate_mape_vs_report_json | PASS | recalc=0.000302044 report=0.0003020439393663318 |
| phase7_D_residual_gate_uq_mae_vs_report_json | PASS | recalc=7.28434e-05 report=7.284336397736285e-05 |
| phase7_D_residual_gate_uq_rmse_vs_report_json | PASS | recalc=7.37106e-05 report=7.371064774159155e-05 |
| phase7_D_residual_gate_uq_r2_vs_report_json | PASS | recalc=1 report=0.9999999999990601 |
| phase7_D_residual_gate_uq_mape_vs_report_json | PASS | recalc=0.000302044 report=0.0003020439393663318 |
| phase7_E_multitask_mae_vs_report_json | PASS | recalc=0.0002545 report=0.0002545004027270478 |
| phase7_E_multitask_rmse_vs_report_json | PASS | recalc=0.00223126 report=0.0022312574555796265 |
| phase7_E_multitask_r2_vs_report_json | PASS | recalc=1 report=0.9999999991387867 |
| phase7_E_multitask_mape_vs_report_json | PASS | recalc=0.00185166 report=0.0018516639166036345 |
| uq_pcnn_picp | PASS | recalc=0.998632 reported=0.9986316989737742 |
| uq_pcnn_mpiw | PASS | recalc=46.1935 reported=46.19350051879883 |
| uq_ar_pcnn_picp_raw | PASS | recalc=0.742075 reported=0.7459521094640821 |
| uq_ar_pcnn_mpiw_raw | PASS | recalc=0.00345678 reported=0.0034567820839583874 |
| uq_ar_pcnn_picp_cal | PASS | recalc=0.944356 reported=0.9443557582668187 scale=4.392307692307693 |
| uq_ar_pcnn_mpiw_cal | PASS | recalc=0.0151832 reported=0.015183172188699245 |
| uq_ar_pcnn_saved_bands_coverage | PASS | band_coverage=0.944356 reported_picp=0.9443557582668187 |
| uq_cd_multitask_json_exists | PASS | present=True |
| uq_cd_finetune_json_exists | PASS | present=True |
| phase7_A_ninja_matches_case1_array | PASS | case1=0.539788 p7=0.5397878947281368 |
| phase7_E_ninja_matches_cd_array | PASS | cd=0.863903 p7=0.8639030248960824 |

### Recalculated headline numbers

Absolute MAE/RMSE units: **W** = single-panel NSRDB Watts; **kW** = 10 kW Case-1/Ninja. Do not mix columns across unit regimes.

| Experiment | Unit | MAE | RMSE | R² | MAPE (%) |
|------------|------|-----|------|----|----------|
| phase3_pcnn_nsrdb_test | W | 2.29473 | 3.76123 | 0.997553 | 10.4521 |
| phase5_ar_pcnn_nsrdb_test | W | 7.28434e-05 | 7.37106e-05 | 1 | 0.000302044 |
| phase5_case1_pcnn | kW | 0.873558 | 1.54292 | 0.539788 | 166.622 |
| phase5_case1_ar_pcnn | kW | 0.912467 | 1.62086 | 0.492123 | 171.763 |
| phase7_B_residual | W | 6.74829e-06 | 1.05558e-05 | 1 | 6.82628e-05 |
| cd_multitask_nsrdb | W | 0.0002545 | 0.00223126 | 1 | 0.00185166 |
| cd_multitask_ninja | kW | 0.504367 | 0.839053 | 0.863903 | 79.8087 |
| cd_finetune_nsrdb | W | 7.28434e-05 | 7.37106e-05 | 1 | 0.000302044 |
| cd_finetune_ninja | kW | 1.632 | 2.79934 | -0.514887 | 100 |
| phase7_A_pcnn | W | 2.29473 | 3.76123 | 0.997553 | 10.4521 |
| phase7_B_residual_only | W | 6.74829e-06 | 1.05558e-05 | 1 | 6.82628e-05 |
| phase7_C_residual_gate | W | 7.28434e-05 | 7.37106e-05 | 1 | 0.000302044 |
| phase7_D_residual_gate_uq | W | 7.28434e-05 | 7.37106e-05 | 1 | 0.000302044 |
| phase7_E_multitask | W | 0.0002545 | 0.00223126 | 1 | 0.00185166 |

## 3. Comparison tables vs recalculated values

| Check | Pass | Detail |
|-------|------|--------|
| phase6_comparison_json_mt_nsrdb_mae | PASS | array=0.0002545 json=0.00025450040243999176 |
| phase6_comparison_json_mt_nsrdb_rmse | PASS | array=0.00223126 json=0.0022312574555698526 |
| phase6_comparison_json_mt_nsrdb_r2 | PASS | array=1 json=0.9999999991387867 |
| phase6_comparison_json_mt_nsrdb_mape | PASS | array=0.00185166 json=0.0018516639166036345 |
| phase6_comparison_json_mt_ninja_r2 | PASS | array=0.863903 json=0.8639030252917119 |
| phase6_comparison_json_exists | PASS | phase6_comparison.json |
| phase5_md_contains_nsrdb_r2 | PASS | markdown includes NSRDB R2 0.9976 matching arrays |
| phase5_md_contains_ninja_r2 | PASS | markdown includes Case-1 R2 ≈ 0.5398 |
| phase5_md_contains_ar_ninja_r2 | PASS | markdown includes AR Case-1 R2 ≈ 0.4921 |
| phase5_md_nsrdb_pcnn_mae | PASS | table=2.2947 recalc=2.29473 |
| phase5_md_nsrdb_ar_mae | PASS | table=0.0001 recalc=7.28434e-05 |
| phase5_md_nsrdb_pcnn_rmse | PASS | table=3.7612 recalc=3.76123 |
| phase5_md_nsrdb_ar_rmse | PASS | table=0.0001 recalc=7.37106e-05 |
| phase5_md_nsrdb_pcnn_r2 | PASS | table=0.9976 recalc=0.997553 |
| phase5_md_nsrdb_ar_r2 | PASS | table=1.0 recalc=1 |
| phase5_md_nsrdb_pcnn_mape | PASS | table=10.452 recalc=10.4521 |
| phase5_md_nsrdb_ar_mape | PASS | table=0.0 recalc=0.000302044 |
| phase5_md_ninja_pcnn_mae | PASS | table=0.8736 recalc=0.873558 |
| phase5_md_ninja_ar_mae | PASS | table=0.9125 recalc=0.912467 |
| phase5_md_ninja_pcnn_rmse | PASS | table=1.5429 recalc=1.54292 |
| phase5_md_ninja_ar_rmse | PASS | table=1.6209 recalc=1.62086 |
| phase5_md_ninja_pcnn_r2 | PASS | table=0.5398 recalc=0.539788 |
| phase5_md_ninja_ar_r2 | PASS | table=0.4921 recalc=0.492123 |
| phase6_md_contains_mt_ninja_r2 | PASS | markdown includes multitask ninja R2 ≈ 0.8639 |
| phase7_md_A_pcnn_r2 | PASS | table=0.9976 source=0.9975527853018523 |
| phase7_md_A_pcnn_rmse | PASS | table=3.7612 source=3.761234036969936 |
| phase7_md_A_pcnn_mae | PASS | table=2.2947 source=2.2947349548339844 |
| phase7_md_A_pcnn_mape | PASS | table=10.452 source=10.45208090460635 |
| phase7_md_A_pcnn_picp | PASS | table=0.996 source=0.9962371721778791 |
| phase7_md_A_pcnn_mpiw | PASS | table=45.72 source=45.7204475402832 |
| phase7_md_A_pcnn_ninja_r2 | PASS | table=0.5398 source=0.5397878947281368 |
| phase7_md_B_residual_only_r2 | PASS | table=1.0 source=0.9999999999999807 |
| phase7_md_B_residual_only_rmse | PASS | table=0.0 source=1.0555793703741656e-05 |
| phase7_md_B_residual_only_mae | PASS | table=0.0 source=6.748289251845563e-06 |
| phase7_md_B_residual_only_mape | PASS | table=0.0 source=6.826278971217283e-05 |
| phase7_md_B_residual_only_picp | PASS | table=0.98 source=0.9795895096921323 |
| phase7_md_B_residual_only_mpiw | PASS | table=0.0 source=0.00012047339987475425 |
| phase7_md_B_residual_only_ninja_r2 | PASS | table=0.4921 source=0.49212416200062625 |
| phase7_md_C_residual_gate_r2 | PASS | table=1.0 source=0.9999999999990601 |
| phase7_md_C_residual_gate_rmse | PASS | table=0.0001 source=7.371064774159155e-05 |
| phase7_md_C_residual_gate_mae | PASS | table=0.0001 source=7.284335879376158e-05 |
| phase7_md_C_residual_gate_mape | PASS | table=0.0 source=0.0003020439393663318 |
| phase7_md_C_residual_gate_picp | PASS | table=0.683 source=0.6832383124287343 |
| phase7_md_C_residual_gate_mpiw | PASS | table=0.0 source=0.002570176962763071 |
| phase7_md_C_residual_gate_ninja_r2 | PASS | table=0.4921 source=0.4921232773293678 |
| phase7_md_D_residual_gate_uq_r2 | PASS | table=1.0 source=0.9999999999990601 |
| phase7_md_D_residual_gate_uq_rmse | PASS | table=0.0001 source=7.371064774159155e-05 |
| phase7_md_D_residual_gate_uq_mae | PASS | table=0.0001 source=7.284335879376158e-05 |
| phase7_md_D_residual_gate_uq_mape | PASS | table=0.0 source=0.0003020439393663318 |
| phase7_md_D_residual_gate_uq_picp | PASS | table=0.942 source=0.9417331812998859 |
| phase7_md_D_residual_gate_uq_mpiw | PASS | table=0.01 source=0.014136007986962795 |
| phase7_md_D_residual_gate_uq_ninja_r2 | PASS | table=0.4921 source=0.4921232773293678 |
| phase7_md_E_multitask_r2 | PASS | table=1.0 source=0.9999999991387867 |
| phase7_md_E_multitask_rmse | PASS | table=0.0022 source=0.0022312574555796265 |
| phase7_md_E_multitask_mae | PASS | table=0.0003 source=0.00025450039538554847 |
| phase7_md_E_multitask_mape | PASS | table=0.002 source=0.0018516639166036345 |
| phase7_md_E_multitask_picp | PASS | table=0.968 source=0.9678449258836944 |
| phase7_md_E_multitask_mpiw | PASS | table=0.01 source=0.014767532236874104 |
| phase7_md_E_multitask_ninja_r2 | PASS | table=0.8639 source=0.8639030248960824 |

## 4. Figures vs raw arrays

| Check | Pass | Detail |
|-------|------|--------|
| figure_exists:interpretability_C.png | PASS | data\figures\phase7_publication\interpretability_C.png |
| figure_exists:interpretability_E.png | PASS | data\figures\phase7_publication\interpretability_E.png |
| figure_exists:error_analysis_comparison.png | PASS | data\figures\phase7_publication\error_analysis_comparison.png |
| figure_exists:error_by_regime.png | PASS | data\figures\phase7_publication\error_by_regime.png |
| figure_exists:ablation_comparison_bars.png | PASS | data\figures\phase7_publication\ablation_comparison_bars.png |
| figure_exists:uncertainty_calibration_D.png | PASS | data\figures\phase7_publication\uncertainty_calibration_D.png |
| figure_exists:architecture_schematic.png | PASS | data\figures\phase7_publication\architecture_schematic.png |
| figure_exists:workflow_schematic.png | PASS | data\figures\phase7_publication\workflow_schematic.png |
| figure_exists:ar_pcnn_case1_validation.png | PASS | data\figures\ar_pcnn_case1_validation.png |
| figure_exists:ar_pcnn_uncertainty_calibrated.png | PASS | data\figures\ar_pcnn_uncertainty_calibrated.png |
| figure_exists:case1_ninja_validation.png | PASS | data\figures\case1_ninja_validation.png |
| figure_array_align_C_residual_gate | PASS | n_ghi=8770 alpha=8770 resid=8770 y=8770 |
| figure_source_r2_C_residual_gate | PASS | array_r2=1.00000000 (figures built from these arrays) |
| figure_alpha_range_C_residual_gate | PASS | min=0.7531 max=1.0000 |
| figure_array_align_E_multitask | PASS | n_ghi=8770 alpha=8770 resid=8770 y=8770 |
| figure_source_r2_E_multitask | PASS | array_r2=1.00000000 (figures built from these arrays) |
| figure_alpha_range_E_multitask | PASS | min=0.9944 max=1.0000 |

Publication figures under `data/figures/phase7_publication/` are generated from `phase7_arrays.npz` (aligned α, residual, GHI, hour, month). Architecture/workflow schematics are illustrative diagrams, not data plots.

## 5. Multitask Ninja label masking

| Check | Pass | Detail |
|-------|------|--------|
| train_external_only_2019 | PASS | years=[2019] n_ext=1753 |
| val_external_within_ninja_calendar | PASS | years=[2019, 2020] n_ext=7007 max_ts=2020-01-01 04:00:00+05:00 |
| cd_loss_masks_external | PASS | CrossDomainLoss uses has_external mask |
| train_cd_uses_masked_loss | PASS | train_ar_pcnn_cd imports/uses masked CD loss |
| ninja_not_in_features | PASS | features=['GHI', 'DNI', 'DHI', 'Temperature', 'Relative Humidity', 'Pressure', 'Surface Albedo', 'Wind Speed', 'Wind Direction', 'Dew Point', 'Precipitable Water', 'Solar Zenith Angle', 'hour_sin', 'hour_cos'] |
| train_external_within_train_window | PASS | ext=[2019-01-01 05:00:00+05:00 .. 2019-03-15 05:00:00+05:00] |

**Intended use:** `CrossDomainLoss` applies MSE on `external_kw` only where `has_external==1` (aligned 2019 hours). NSRDB test (2020) has **zero** ninja labels. Features never include ninja electricity.

## 6. Train / val / test leakage

| Check | Pass | Detail |
|-------|------|--------|
| no_timestamp_overlap | PASS | pairwise intersections empty |
| chronological_order | PASS | train_max=2019-03-15 05:00:00+05:00 val=[2019-03-15 06:00:00+05:00..2020-01-01 13:00:00+05:00] test_min=2020-01-01 14:00:00+05:00 |
| scaler_train_only | PASS | mean/scale match train empirical |

## 7. Phase 7 warnings — scientific impact

| Warning | Affects validity? | Explanation |
|---------|-------------------|-------------|
| InconsistentVersionWarning (sklearn StandardScaler 1.6.1 vs 1.9.0) | **NO** | Scaler was pickled with scikit-learn 1.6.1 during Phase 2; Phase 7 venv has sklearn 1.9.0. Audit re-verified scaler mean/scale against train-raw empirical statistics (allclose). Transformed features remain consistent; warning is packaging, not leakage. |
| UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names | **NO** | Scaler was fit on a pandas DataFrame (named columns); Case-1 / ninja eval passes a numpy array to transform(). Column order is fixed by cfg['features'] matching training order. Numeric transform is identical without names. |
| RuntimeWarning: Mean of empty slice (nanmean on alpha/residual for Model A) | **NO** | Baseline PCNN has no gate α or residual tensors; Phase 7 filled them with NaN, then called nanmean. Only Model A triggered this. Point metrics for A come from y_true/y_pred, not α. Fixed in later Phase 7 runner to avoid the warning. |

**Conclusion:** None of the three Phase 7 warnings invalidate reported metrics.

## 8. Metric changes after audit

No material metric changes. Recalculated values match saved JSON with magnitude-aware tolerance: absolute 1e−3 for O(1) metrics; for |value|<1e−2 use max(5e−4 floor, 5% relative) so tiny AR/CD MAE cannot pass order-of-magnitude mismatches while display-rounded tables still match. PICP/MPIW use 5e−2 for MC stochasticity.


## 9. Confidence scores by experiment

| Experiment | Confidence (0–1) | Checks passed |
|------------|------------------|---------------|
| phase5_nsrdb_pcnn_vs_ar | **1.0** | 16/16 |
| phase5_case1_ninja | **1.0** | 14/14 |
| phase5_uncertainty | **1.0** | 7/7 |
| phase6_multitask | **1.0** | 16/16 |
| phase6_finetune | **1.0** | 8/8 |
| phase7_ablation_table | **1.0** | 62/62 |
| leakage_and_masks | **1.0** | 8/8 |
| figures | **1.0** | 17/17 |

## 10. Reviewer concerns (disclose)

1. Lead with Ninja multitask transfer; treat NSRDB near-unity R² as a physics-label consistency check only (Phase 4 circular labels).
2. Full-2019 ninja Case-1 for multitask overlaps early-2019 train hours that saw masked external loss; prefer val-window holdout R² (~0.81) as the cleaner transfer estimate.
3. Site coordinates differ slightly from the paper VITCC location (~25 km). Do not align numeric tables to published paper MAE/R²: our NSRDB MAE is Watts (1× TS250) and Case-1 is kW (10 kW system); the paper’s absolute MAE (e.g. 0.0213) does not specify capacity/unit clearly enough to convert — not reconcilable; never guess 0.0213 kW ↔ 21.3 W.
4. PICP/MPIW for MC Dropout are stochastic across runs; Phase 7 re-runs can differ at the second decimal — report calibrated PICP with seed/n_mc.
5. Model D shares weights with C; only UQ post-processing differs — state this clearly in the ablation narrative.
6. Feasible fraction = 1.0 is guaranteed by the hard capacity clamp (architecture), not a learned achievement.
7. Prefer sMAPE / nRMSE over MAPE for external relative error; MAPE inflates at near-zero irradiance.
8. Sklearn scaler version skew generates a warning; empirically verified harmless but reviewers may ask for re-serialization.
9. Phase 9 λ-sensitivity: λ=0 is an untrained external-head control (Ninja R² negative); prefer λ≈0.25–1.0 for transfer claims.

## 11. Full check ledger

| Check | Pass | Severity | Detail |
|-------|------|----------|--------|
| metrics_sourced_from_code:evaluate.py | PASS | critical | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:evaluate_ar_pcnn.py | PASS | critical | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:case1_ninja.py | PASS | critical | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:case1_ar_pcnn.py | PASS | critical | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:evaluate_ar_pcnn_cd.py | PASS | critical | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:uncertainty.py | PASS | critical | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:uncertainty_ar_pcnn.py | PASS | critical | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:phase7_run.py | PASS | critical | summarize/regression=True load/save_arrays=True |
| metrics_sourced_from_code:compare_phase5.py | PASS | critical | summarize/regression=False load/save_arrays=True |
| metrics_sourced_from_code:compare_phase6.py | PASS | critical | summarize/regression=False load/save_arrays=True |
| phase5_7_docs_are_derived_artifacts | PASS | critical | comparison JSON artifacts exist alongside markdown tables |
| phase3_pcnn_nsrdb_test_mae_matches_json | PASS | critical | recalc=2.29473 reported=2.2947349637050176 Δ=8.87e-09 |
| phase3_pcnn_nsrdb_test_rmse_matches_json | PASS | critical | recalc=3.76123 reported=3.761233935951727 Δ=0 |
| phase3_pcnn_nsrdb_test_r2_matches_json | PASS | critical | recalc=0.997553 reported=0.9975527854333056 Δ=0 |
| phase3_pcnn_nsrdb_test_mape_matches_json | PASS | critical | recalc=10.4521 reported=10.45208070811051 Δ=0 |
| phase5_ar_pcnn_nsrdb_test_mae_matches_json | PASS | critical | recalc=7.28434e-05 reported=7.284336416112849e-05 Δ=5.37e-12 |
| phase5_ar_pcnn_nsrdb_test_rmse_matches_json | PASS | critical | recalc=7.37106e-05 reported=7.371064790110818e-05 Δ=0 |
| phase5_ar_pcnn_nsrdb_test_r2_matches_json | PASS | critical | recalc=1 reported=0.9999999999990601 Δ=0 |
| phase5_ar_pcnn_nsrdb_test_mape_matches_json | PASS | critical | recalc=0.000302044 reported=0.0003020439393663318 Δ=0 |
| phase5_case1_pcnn_mae_matches_json | PASS | critical | recalc=0.873558 reported=0.8735581319551258 Δ=0 |
| phase5_case1_pcnn_rmse_matches_json | PASS | critical | recalc=1.54292 reported=1.5429226314288524 Δ=0 |
| phase5_case1_pcnn_r2_matches_json | PASS | critical | recalc=0.539788 reported=0.5397878981046011 Δ=0 |
| phase5_case1_pcnn_mape_matches_json | PASS | critical | recalc=166.622 reported=166.62194856458473 Δ=0 |
| phase5_case1_ar_pcnn_mae_matches_json | PASS | critical | recalc=0.912467 reported=0.9124674422228375 Δ=0 |
| phase5_case1_ar_pcnn_rmse_matches_json | PASS | critical | recalc=1.62086 reported=1.6208554516340703 Δ=0 |
| phase5_case1_ar_pcnn_r2_matches_json | PASS | critical | recalc=0.492123 reported=0.4921232773293678 Δ=0 |
| phase5_case1_ar_pcnn_mape_matches_json | PASS | critical | recalc=171.763 reported=171.76252667871975 Δ=0 |
| phase7_B_residual_mae_matches_json | PASS | critical | recalc=6.74829e-06 reported=6.748289039768406e-06 Δ=2.12e-13 |
| phase7_B_residual_rmse_matches_json | PASS | critical | recalc=1.05558e-05 reported=1.0555793703741656e-05 Δ=0 |
| phase7_B_residual_r2_matches_json | PASS | critical | recalc=1 reported=0.9999999999999807 Δ=0 |
| phase7_B_residual_mape_matches_json | PASS | critical | recalc=6.82628e-05 reported=6.826278971217283e-05 Δ=0 |
| cd_multitask_nsrdb_mae | PASS | critical | recalc=0.0002545 reported=0.00025450040243999176 |
| cd_multitask_nsrdb_rmse | PASS | critical | recalc=0.00223126 reported=0.0022312574555698526 |
| cd_multitask_nsrdb_r2 | PASS | critical | recalc=1 reported=0.9999999991387867 |
| cd_multitask_nsrdb_mape | PASS | critical | recalc=0.00185166 reported=0.0018516639166036345 |
| cd_multitask_ninja_mae | PASS | critical | recalc=0.504367 reported=0.5043669195172209 |
| cd_multitask_ninja_rmse | PASS | critical | recalc=0.839053 reported=0.8390529478745805 |
| cd_multitask_ninja_r2 | PASS | critical | recalc=0.863903 reported=0.8639030252917119 |
| cd_finetune_nsrdb_mae | PASS | critical | recalc=7.28434e-05 reported=7.284336416112849e-05 |
| cd_finetune_nsrdb_rmse | PASS | critical | recalc=7.37106e-05 reported=7.371064790110818e-05 |
| cd_finetune_nsrdb_r2 | PASS | critical | recalc=1 reported=0.9999999999990601 |
| cd_finetune_nsrdb_mape | PASS | critical | recalc=0.000302044 reported=0.0003020439393663318 |
| cd_finetune_ninja_mae | PASS | critical | recalc=1.632 reported=1.6320034266133638 |
| cd_finetune_ninja_rmse | PASS | critical | recalc=2.79934 reported=2.7993373669297537 |
| cd_finetune_ninja_r2 | PASS | critical | recalc=-0.514887 reported=-0.5148869313126689 |
| phase7_A_pcnn_mae_vs_report_json | PASS | critical | recalc=2.29473 report=2.294734990928097 |
| phase7_A_pcnn_rmse_vs_report_json | PASS | critical | recalc=3.76123 report=3.761234036969936 |
| phase7_A_pcnn_r2_vs_report_json | PASS | critical | recalc=0.997553 report=0.9975527853018523 |
| phase7_A_pcnn_mape_vs_report_json | PASS | critical | recalc=10.4521 report=10.45208090460635 |
| phase7_B_residual_only_mae_vs_report_json | PASS | critical | recalc=6.74829e-06 report=6.748289039768406e-06 |
| phase7_B_residual_only_rmse_vs_report_json | PASS | critical | recalc=1.05558e-05 report=1.0555793703741656e-05 |
| phase7_B_residual_only_r2_vs_report_json | PASS | critical | recalc=1 report=0.9999999999999807 |
| phase7_B_residual_only_mape_vs_report_json | PASS | critical | recalc=6.82628e-05 report=6.826278971217283e-05 |
| phase7_C_residual_gate_mae_vs_report_json | PASS | critical | recalc=7.28434e-05 report=7.284336397736285e-05 |
| phase7_C_residual_gate_rmse_vs_report_json | PASS | critical | recalc=7.37106e-05 report=7.371064774159155e-05 |
| phase7_C_residual_gate_r2_vs_report_json | PASS | critical | recalc=1 report=0.9999999999990601 |
| phase7_C_residual_gate_mape_vs_report_json | PASS | critical | recalc=0.000302044 report=0.0003020439393663318 |
| phase7_D_residual_gate_uq_mae_vs_report_json | PASS | critical | recalc=7.28434e-05 report=7.284336397736285e-05 |
| phase7_D_residual_gate_uq_rmse_vs_report_json | PASS | critical | recalc=7.37106e-05 report=7.371064774159155e-05 |
| phase7_D_residual_gate_uq_r2_vs_report_json | PASS | critical | recalc=1 report=0.9999999999990601 |
| phase7_D_residual_gate_uq_mape_vs_report_json | PASS | critical | recalc=0.000302044 report=0.0003020439393663318 |
| phase7_E_multitask_mae_vs_report_json | PASS | critical | recalc=0.0002545 report=0.0002545004027270478 |
| phase7_E_multitask_rmse_vs_report_json | PASS | critical | recalc=0.00223126 report=0.0022312574555796265 |
| phase7_E_multitask_r2_vs_report_json | PASS | critical | recalc=1 report=0.9999999991387867 |
| phase7_E_multitask_mape_vs_report_json | PASS | critical | recalc=0.00185166 report=0.0018516639166036345 |
| phase5_comparison_json_baseline_pcnn_mae | PASS | critical | array=2.29473 json=2.2947349637050176 |
| phase5_comparison_json_baseline_pcnn_rmse | PASS | critical | array=3.76123 json=3.761233935951727 |
| phase5_comparison_json_baseline_pcnn_r2 | PASS | critical | array=0.997553 json=0.9975527854333056 |
| phase5_comparison_json_baseline_pcnn_mape | PASS | critical | array=10.4521 json=10.45208070811051 |
| phase5_comparison_json_ar_pcnn_mae | PASS | critical | array=7.28434e-05 json=7.284336416112849e-05 |
| phase5_comparison_json_ar_pcnn_rmse | PASS | critical | array=7.37106e-05 json=7.371064790110818e-05 |
| phase5_comparison_json_ar_pcnn_r2 | PASS | critical | array=1 json=0.9999999999990601 |
| phase5_comparison_json_ar_pcnn_mape | PASS | critical | array=0.000302044 json=0.0003020439393663318 |
| phase5_comparison_json_case1_baseline_pcnn_mae | PASS | critical | array=0.873558 json=0.8735581319551258 |
| phase5_comparison_json_case1_baseline_pcnn_rmse | PASS | critical | array=1.54292 json=1.5429226314288524 |
| phase5_comparison_json_case1_baseline_pcnn_r2 | PASS | critical | array=0.539788 json=0.5397878981046011 |
| phase5_comparison_json_case1_ar_pcnn_mae | PASS | critical | array=0.912467 json=0.9124674422228375 |
| phase5_comparison_json_case1_ar_pcnn_rmse | PASS | critical | array=1.62086 json=1.6208554516340703 |
| phase5_comparison_json_case1_ar_pcnn_r2 | PASS | critical | array=0.492123 json=0.4921232773293678 |
| phase6_comparison_json_mt_nsrdb_mae | PASS | critical | array=0.0002545 json=0.00025450040243999176 |
| phase6_comparison_json_mt_nsrdb_rmse | PASS | critical | array=0.00223126 json=0.0022312574555698526 |
| phase6_comparison_json_mt_nsrdb_r2 | PASS | critical | array=1 json=0.9999999991387867 |
| phase6_comparison_json_mt_nsrdb_mape | PASS | critical | array=0.00185166 json=0.0018516639166036345 |
| phase6_comparison_json_mt_ninja_r2 | PASS | critical | array=0.863903 json=0.8639030252917119 |
| uq_pcnn_picp | PASS | major | recalc=0.998632 reported=0.9986316989737742 |
| uq_pcnn_mpiw | PASS | major | recalc=46.1935 reported=46.19350051879883 |
| uq_ar_pcnn_picp_raw | PASS | major | recalc=0.742075 reported=0.7459521094640821 |
| uq_ar_pcnn_mpiw_raw | PASS | major | recalc=0.00345678 reported=0.0034567820839583874 |
| uq_ar_pcnn_picp_cal | PASS | major | recalc=0.944356 reported=0.9443557582668187 scale=4.392307692307693 |
| uq_ar_pcnn_mpiw_cal | PASS | major | recalc=0.0151832 reported=0.015183172188699245 |
| uq_ar_pcnn_saved_bands_coverage | PASS | major | band_coverage=0.944356 reported_picp=0.9443557582668187 |
| uq_cd_multitask_json_exists | PASS | critical | present=True |
| uq_cd_finetune_json_exists | PASS | critical | present=True |
| phase5_comparison_json_exists | PASS | critical | phase5_comparison.json |
| phase6_comparison_json_exists | PASS | critical | phase6_comparison.json |
| phase5_md_contains_nsrdb_r2 | PASS | critical | markdown includes NSRDB R2 0.9976 matching arrays |
| phase5_md_contains_ninja_r2 | PASS | critical | markdown includes Case-1 R2 ≈ 0.5398 |
| phase5_md_contains_ar_ninja_r2 | PASS | critical | markdown includes AR Case-1 R2 ≈ 0.4921 |
| phase5_md_nsrdb_pcnn_mae | PASS | critical | table=2.2947 recalc=2.29473 |
| phase5_md_nsrdb_ar_mae | PASS | critical | table=0.0001 recalc=7.28434e-05 |
| phase5_md_nsrdb_pcnn_rmse | PASS | critical | table=3.7612 recalc=3.76123 |
| phase5_md_nsrdb_ar_rmse | PASS | critical | table=0.0001 recalc=7.37106e-05 |
| phase5_md_nsrdb_pcnn_r2 | PASS | critical | table=0.9976 recalc=0.997553 |
| phase5_md_nsrdb_ar_r2 | PASS | critical | table=1.0 recalc=1 |
| phase5_md_nsrdb_pcnn_mape | PASS | critical | table=10.452 recalc=10.4521 |
| phase5_md_nsrdb_ar_mape | PASS | critical | table=0.0 recalc=0.000302044 |
| phase5_md_ninja_pcnn_mae | PASS | critical | table=0.8736 recalc=0.873558 |
| phase5_md_ninja_ar_mae | PASS | critical | table=0.9125 recalc=0.912467 |
| phase5_md_ninja_pcnn_rmse | PASS | critical | table=1.5429 recalc=1.54292 |
| phase5_md_ninja_ar_rmse | PASS | critical | table=1.6209 recalc=1.62086 |
| phase5_md_ninja_pcnn_r2 | PASS | critical | table=0.5398 recalc=0.539788 |
| phase5_md_ninja_ar_r2 | PASS | critical | table=0.4921 recalc=0.492123 |
| phase6_md_contains_mt_ninja_r2 | PASS | critical | markdown includes multitask ninja R2 ≈ 0.8639 |
| phase7_md_A_pcnn_r2 | PASS | critical | table=0.9976 source=0.9975527853018523 |
| phase7_md_A_pcnn_rmse | PASS | critical | table=3.7612 source=3.761234036969936 |
| phase7_md_A_pcnn_mae | PASS | critical | table=2.2947 source=2.2947349548339844 |
| phase7_md_A_pcnn_mape | PASS | critical | table=10.452 source=10.45208090460635 |
| phase7_md_A_pcnn_picp | PASS | major | table=0.996 source=0.9962371721778791 |
| phase7_md_A_pcnn_mpiw | PASS | major | table=45.72 source=45.7204475402832 |
| phase7_md_A_pcnn_ninja_r2 | PASS | critical | table=0.5398 source=0.5397878947281368 |
| phase7_md_B_residual_only_r2 | PASS | critical | table=1.0 source=0.9999999999999807 |
| phase7_md_B_residual_only_rmse | PASS | critical | table=0.0 source=1.0555793703741656e-05 |
| phase7_md_B_residual_only_mae | PASS | critical | table=0.0 source=6.748289251845563e-06 |
| phase7_md_B_residual_only_mape | PASS | critical | table=0.0 source=6.826278971217283e-05 |
| phase7_md_B_residual_only_picp | PASS | major | table=0.98 source=0.9795895096921323 |
| phase7_md_B_residual_only_mpiw | PASS | major | table=0.0 source=0.00012047339987475425 |
| phase7_md_B_residual_only_ninja_r2 | PASS | critical | table=0.4921 source=0.49212416200062625 |
| phase7_md_C_residual_gate_r2 | PASS | critical | table=1.0 source=0.9999999999990601 |
| phase7_md_C_residual_gate_rmse | PASS | critical | table=0.0001 source=7.371064774159155e-05 |
| phase7_md_C_residual_gate_mae | PASS | critical | table=0.0001 source=7.284335879376158e-05 |
| phase7_md_C_residual_gate_mape | PASS | critical | table=0.0 source=0.0003020439393663318 |
| phase7_md_C_residual_gate_picp | PASS | major | table=0.683 source=0.6832383124287343 |
| phase7_md_C_residual_gate_mpiw | PASS | major | table=0.0 source=0.002570176962763071 |
| phase7_md_C_residual_gate_ninja_r2 | PASS | critical | table=0.4921 source=0.4921232773293678 |
| phase7_md_D_residual_gate_uq_r2 | PASS | critical | table=1.0 source=0.9999999999990601 |
| phase7_md_D_residual_gate_uq_rmse | PASS | critical | table=0.0001 source=7.371064774159155e-05 |
| phase7_md_D_residual_gate_uq_mae | PASS | critical | table=0.0001 source=7.284335879376158e-05 |
| phase7_md_D_residual_gate_uq_mape | PASS | critical | table=0.0 source=0.0003020439393663318 |
| phase7_md_D_residual_gate_uq_picp | PASS | major | table=0.942 source=0.9417331812998859 |
| phase7_md_D_residual_gate_uq_mpiw | PASS | major | table=0.01 source=0.014136007986962795 |
| phase7_md_D_residual_gate_uq_ninja_r2 | PASS | critical | table=0.4921 source=0.4921232773293678 |
| phase7_md_E_multitask_r2 | PASS | critical | table=1.0 source=0.9999999991387867 |
| phase7_md_E_multitask_rmse | PASS | critical | table=0.0022 source=0.0022312574555796265 |
| phase7_md_E_multitask_mae | PASS | critical | table=0.0003 source=0.00025450039538554847 |
| phase7_md_E_multitask_mape | PASS | critical | table=0.002 source=0.0018516639166036345 |
| phase7_md_E_multitask_picp | PASS | major | table=0.968 source=0.9678449258836944 |
| phase7_md_E_multitask_mpiw | PASS | major | table=0.01 source=0.014767532236874104 |
| phase7_md_E_multitask_ninja_r2 | PASS | critical | table=0.8639 source=0.8639030248960824 |
| phase7_A_ninja_matches_case1_array | PASS | critical | case1=0.539788 p7=0.5397878947281368 |
| phase7_E_ninja_matches_cd_array | PASS | critical | cd=0.863903 p7=0.8639030248960824 |
| figure_exists:interpretability_C.png | PASS | critical | data\figures\phase7_publication\interpretability_C.png |
| figure_exists:interpretability_E.png | PASS | critical | data\figures\phase7_publication\interpretability_E.png |
| figure_exists:error_analysis_comparison.png | PASS | critical | data\figures\phase7_publication\error_analysis_comparison.png |
| figure_exists:error_by_regime.png | PASS | critical | data\figures\phase7_publication\error_by_regime.png |
| figure_exists:ablation_comparison_bars.png | PASS | critical | data\figures\phase7_publication\ablation_comparison_bars.png |
| figure_exists:uncertainty_calibration_D.png | PASS | critical | data\figures\phase7_publication\uncertainty_calibration_D.png |
| figure_exists:architecture_schematic.png | PASS | critical | data\figures\phase7_publication\architecture_schematic.png |
| figure_exists:workflow_schematic.png | PASS | critical | data\figures\phase7_publication\workflow_schematic.png |
| figure_exists:ar_pcnn_case1_validation.png | PASS | critical | data\figures\ar_pcnn_case1_validation.png |
| figure_exists:ar_pcnn_uncertainty_calibrated.png | PASS | critical | data\figures\ar_pcnn_uncertainty_calibrated.png |
| figure_exists:case1_ninja_validation.png | PASS | critical | data\figures\case1_ninja_validation.png |
| figure_array_align_C_residual_gate | PASS | critical | n_ghi=8770 alpha=8770 resid=8770 y=8770 |
| figure_source_r2_C_residual_gate | PASS | critical | array_r2=1.00000000 (figures built from these arrays) |
| figure_alpha_range_C_residual_gate | PASS | critical | min=0.7531 max=1.0000 |
| figure_array_align_E_multitask | PASS | critical | n_ghi=8770 alpha=8770 resid=8770 y=8770 |
| figure_source_r2_E_multitask | PASS | critical | array_r2=1.00000000 (figures built from these arrays) |
| figure_alpha_range_E_multitask | PASS | critical | min=0.9944 max=1.0000 |
| ablation_bars_ninja_A_matches_case1_array | PASS | critical | case1_pcnn_r2=0.539788 |
| no_timestamp_overlap | PASS | critical | pairwise intersections empty |
| chronological_order | PASS | critical | train_max=2019-03-15 05:00:00+05:00 val=[2019-03-15 06:00:00+05:00..2020-01-01 13:00:00+05:00] test_min=2020-01-01 14:00:00+05:00 |
| scaler_train_only | PASS | critical | mean/scale match train empirical |
| train_external_only_2019 | PASS | critical | years=[2019] n_ext=1753 |
| val_external_within_ninja_calendar | PASS | critical | years=[2019, 2020] n_ext=7007 max_ts=2020-01-01 04:00:00+05:00 |
| test_has_zero_ninja_labels | PASS | critical | test_external_count=0 |
| cd_loss_masks_external | PASS | critical | CrossDomainLoss uses has_external mask |
| train_cd_uses_masked_loss | PASS | critical | train_ar_pcnn_cd imports/uses masked CD loss |
| ninja_not_in_features | PASS | critical | features=['GHI', 'DNI', 'DHI', 'Temperature', 'Relative Humidity', 'Pressure', 'Surface Albedo', 'Wind Speed', 'Wind Direction', 'Dew Point', 'Precipitable Water', 'Solar Zenith Angle', 'hour_sin', 'hour_cos'] |
| train_external_within_train_window | PASS | critical | ext=[2019-01-01 05:00:00+05:00 .. 2019-03-15 05:00:00+05:00] |

## How to reproduce this audit

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.analysis.phase8_audit
```
