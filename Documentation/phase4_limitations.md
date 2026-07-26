# Phase 4 — Limitations of the reproduced PCNN

*Auto-generated from analysis. Use this to drive Phase 5 design.*

## 1. Target definition makes NSRDB scores look too good

- Test R² ≈ **0.9976**, MAE ≈ **2.29 W**.
- **Critical:** `target_power` is exactly `P_bounded` (analytical). Physics-vs-target R² ≈ **1.0000** (≈1 by construction).
- PCNN vs physics R² ≈ **0.9976**; mean |pred − physics| ≈ **2.29 W**.
- **Limitation:** the network is mostly reproducing an equation it is already fed — not learning hard residual structure from real PV measurements.

## 2. Errors concentrate in high-irradiance / midday conditions

- Worst hour (mean |error|): **12**
- Worst month (mean |error|): **3**
- Daytime MAE (>50 W/m² GHI): **4.29 W** vs night MAE: **0.71 W**
- **Limitation:** capacity-sensitive midday peaks still dominate residual error.

## 3. Weak external generalization (Renewables.ninja Case 1)

- Case 1 R² ≈ **0.540**, MAE ≈ **0.87 kW**, bias ≈ **0.60 kW**.
- We **do not** cite published Case-1 R² as a benchmark: units, site (~25 km from VITCC), and label construction differ, so numeric paper alignment is not claimed.
- Likely drivers of weak transfer: NSRDB weather vs MERRA-2 weather underlying ninja, Ninja system model (tilt 35°, losses, tracking=0) vs simple TS250 physics target, Site coordinate / timezone alignment differences.
- **Limitation:** model trained on NSRDB physics targets does not transfer cleanly to ninja’s MERRA-2 PV series.

## 4. Uncertainty is over-conservative

- Test PICP @ 95% ≈ **0.999** (over-wide vs a nominal 95% target).
- Calibration curve shows coverage above the diagonal at all nominal levels.
- **Limitation:** MC Dropout bands are too wide for decision-making; need calibration (temperature scaling, conformal, better aleatoric head, etc.).

## 5. Architectural / training limitations (from reproduction)

- Physics enters mainly as a scaled prior + soft loss penalties — not a hard structural constraint.
- No dedicated temporal sequence model (hourly i.i.d.-style MLP on engineered features).
- Single-panel Watts training vs system-level kW evaluation — do not mix scales when reading tables.
- Feature set for Case 1 still relies on NSRDB meteorology aligned in time, not ninja-native full feature vectors.

## Implications for Phase 5 (novelty directions)

1. **Train / evaluate on targets closer to real or external PV** (or multi-task: physics + ninja/system).
2. **Reduce physics copying** — residual learning (predict correction to physics) or gated fusion with learned trust.
3. **Better temporal modeling** under distribution shift (attention / TCN / informed LSTM) without abandoning constraints.
4. **Calibrated uncertainty** — conformal prediction or explicit variance head + proper scoring rules.
5. **Adaptive constraint weights** when physics prior conflicts with external observations.
