# How to check & interpret final results

Plain-language guide: run the commands yourself, then read what each number means.

**Always activate the project venv first:**

```powershell
cd E:\Solar-PCNN-Research
.\.venv\Scripts\Activate.ps1
```

---

## Quick command list (copy–paste)

### 0. Sanity — are artifacts present?

```powershell
dir artifacts\checkpoints\*.pt
dir artifacts\metrics\phase7_report.json, artifacts\metrics\phase8_audit.json
dir artifacts\phase9\csv
dir data\figures\phase7_publication
dir data\figures\phase9
```

### 1. Phase 5 / 6 comparison tables (open as text)

```powershell
type phase5_comparison.md
type phase6_comparison.md
type phase7_validation.md
```

### 2. Re-run scientific audit (Phase 8)

```powershell
python -m src.analysis.phase8_audit
```

Expect: `Ready: READY with caveats  178/178 passed`

### 3. Recalculate NSRDB R² / MAE from saved arrays

```powershell
python -c "import numpy as np
pairs=[('PCNN','artifacts/metrics/predictions_test.npz'),
('AR-PCNN','artifacts/metrics/ar_pcnn_predictions_test.npz'),
('CD-MT','artifacts/metrics/ar_pcnn_cd_multitask_test_arrays.npz')]
for name,path in pairs:
 z=np.load(path); y,p=z['y_true'],z['y_pred']; e=p-y
 print(name, 'R2=%.12f' % (1-np.sum(e**2)/np.sum((y-y.mean())**2)),
       'MAE=%.6e' % np.mean(np.abs(e)))"
```

### 4. Ablation A–E from Phase 7 arrays

```powershell
python -c "import numpy as np
z=np.load('artifacts/metrics/phase7_arrays.npz')
for m in ['A_pcnn','B_residual_only','C_residual_gate','D_residual_gate_uq','E_multitask']:
 y,p=z[f'{m}_y_true'],z[f'{m}_y_pred']; e=p-y
 print(m, 'R2=%.12f' % (1-np.sum(e**2)/np.sum((y-y.mean())**2)),
       'MAE=%.6e' % np.mean(np.abs(e)))"
```

### 5. Ninja Case-1 R² (external domain)

```powershell
python -c "import numpy as np, json
for f,label in [
 ('artifacts/metrics/case1_ninja_metrics.json','PCNN Case1'),
 ('artifacts/metrics/ar_pcnn_case1_metrics.json','AR-PCNN Case1'),
 ('artifacts/metrics/ar_pcnn_cd_multitask_eval.json','CD Multitask')]:
 d=json.load(open(f,encoding='utf-8'))
 if 'ninja_full_2019_external_head' in d:
  print(label, 'full R2=', round(d['ninja_full_2019_external_head']['r2'],4),
        'val_holdout R2=', round(d['ninja_val_holdout']['r2'],4))
 else:
  print(label, 'R2=', round(d['r2'],4), 'MAE_kW=', round(d['mae'],4))"
```

### 6. Phase 9 result tables (CSV)

```powershell
# Multi-seed stability (mean ± std)
type artifacts\phase9\csv\exp04_multi_seed_summary.csv
type artifacts\phase9\csv\exp04_multi_seed_runs.csv

# Gate α
type artifacts\phase9\csv\exp02_gate_summary.csv

# Residuals
type artifacts\phase9\csv\exp03_residual_stats.csv

# Runtime / params
type artifacts\phase9\csv\exp05_runtime_complexity.csv

# Feature importance (top of AR physics path)
python -c "import pandas as pd; print(pd.read_csv('artifacts/phase9/csv/exp06_importance_physics_ar.csv').head(5))"

# Regime MAE
python -c "import pandas as pd; df=pd.read_csv('artifacts/phase9/csv/exp08_regime_errors.csv'); print(df[df.regime.isin(['afternoon','high_ghi','night','clear_sky'])][['regime','model','mae','r2']].to_string(index=False))"

# Noise: GHI 20%
python -c "import pandas as pd; df=pd.read_csv('artifacts/phase9/csv/exp09_noise_robustness.csv'); print(df[(df.feature.isin(['none','GHI'])) & (df.noise_level.isin([0.0,0.2]))][['model','feature','noise_level','r2','mae']].to_string(index=False))"
```

### 7. Open key figures (Windows)

```powershell
start data\figures\phase7_publication\ablation_comparison_bars.png
start data\figures\phase7_publication\uncertainty_calibration_D.png
start data\figures\phase7_publication\interpretability_C.png
start data\figures\phase6_multitask_ninja_diag.png
start data\figures\phase9\exp04_multi_seed_stability.png
start data\figures\phase9\exp08_regime_errors.png
start data\figures\phase9\exp09_noise_robustness.png
```

### 8. Re-run Phase 7 (optional, ~1–2 min)

```powershell
python -m src.analysis.phase7_run --n-mc 30
```

### 9. Phase 9 analysis only (no retraining)

```powershell
python -m src.analysis.phase9.run_all --skip-train
```

### 10. One-shot “headline metrics” dump

```powershell
python -c "
import json
from pathlib import Path
print('=== PHASE 5 ===')
print(Path('phase5_comparison.md').read_text(encoding='utf-8')[:800])
print('\n=== PHASE 6 NINJA ===')
d=json.load(open('artifacts/metrics/ar_pcnn_cd_multitask_eval.json',encoding='utf-8'))
print('full2019 R2', d['ninja_full_2019_external_head']['r2'])
print('val_holdout R2', d['ninja_val_holdout']['r2'])
print('nsrdb MAE', d['nsrdb_test']['mae'])
print('\n=== PHASE 8 ===')
a=json.load(open('artifacts/metrics/phase8_audit.json',encoding='utf-8'))
print(a['publication_readiness'])
print('\n=== PHASE 9 MULTI-SEED ===')
import pandas as pd
print(pd.read_csv('artifacts/phase9/csv/exp04_multi_seed_summary.csv').to_string(index=False))
"
```

---

## How to interpret the numbers

### A. NSRDB metrics (physics panel, **Watts**) — consistency check only

Target: `target_power` ≈ `P_bounded` for **one** TS250 panel (`p_rated_w = 250`). Absolute errors are **W**, not kW.

| Metric | Good look | Your typical values | Meaning |
|--------|-----------|---------------------|---------|
| **R² (—)** | close to 1 | PCNN ~0.9976; AR/CD ≈1.0 | Fit to physics-derived labels. **Not** a forecasting win — use ≈1.0 + MAE. |
| **MAE (W)** | smaller better | PCNN ~2.3 W; AR ~10⁻⁴ W; CD ~10⁻⁴ W | Average absolute error in Watts |
| **RMSE (W)** | smaller better | PCNN ~3.8 W; CD can be higher than AR in some regimes | Multitask cost possible |
| **Feasible (—)** | 1.0 | 1.0 | **By clamp** — architectural, not learned |

**Caveat:** NSRDB training labels are built from the physics formula. Near-perfect R² for residual models is **by design** (Phase 4). Do not claim “we solved solar forecasting” from NSRDB alone. Do **not** convert source-paper MAE≈0.0213 into Watts/kW — capacity/unit is unspecified → **not reconcilable**.

### B. Ninja metrics (external **10 kW** system) — primary claim

Target: Renewables.ninja `electricity` for **40 × 250 W = 10 kW**. Absolute errors are **kW**.

| Metric | PCNN | AR-PCNN | CD Multitask | Meaning |
|--------|------|---------|--------------|---------|
| Full-2019 R² (—) | ~0.54 | ~0.49 | **~0.86** | Same Case-1 protocol |
| Val-holdout R² (—) | — | — | **~0.81** | **Cleaner** transfer estimate (prefer this in paper) |
| MAE (kW) | ~0.87 | ~0.91 | **~0.50** | Lower is better |
| **sMAPE (%)** | ~133.5 | ~133.1 | **~108.2** | Preferred relative error (bounded ≤200%) |
| **nRMSE (%)** | ~94.5 | ~99.3 | **~51.4** | RMSE / mean(|y|) × 100 |

**Interpretation:** Architecture tricks (gate alone) did **not** fix domain shift. **Multi-task learning** with ninja labels did. Do **not** compare these numbers to published paper Case-1 tables (unit/protocol mismatch).

Standard MAPE is susceptible to extreme inflation during low-irradiance periods due to near-zero denominators; therefore, sMAPE (or nRMSE) is reported to provide a more stable relative error metric across all generation regimes. (Raw Case-1 MAPE can read ~170% even when R²≈0.5 — that is the pathology, not the model.)

### C. Uncertainty (PICP / MPIW)

| Setting | PICP | Meaning |
|---------|------|---------|
| PCNN raw MC | ~0.999 | Too wide — over-confident coverage |
| AR raw MC | ~0.68–0.75 | Too narrow |
| AR / CD **calibrated** | **~0.94–0.96** | Near the 95% target — use these for claims |

PICP = fraction of truths inside the prediction interval.  
MPIW = average interval width (narrower is better **if** PICP is OK).

### D. Gate α (interpretability)

From `exp02_gate_summary.csv`:

- AR mean α ≈ **0.94** (day ≈ **1.0**, night ≈ **0.89**)  
  → trusts physics by day; adjusts more at night.
- CD mean α ≈ **1.00** on the physics path  
  → multitask did not break physics trust; ninja learning lives in the **external head**.

### E. Ablation A–E (what each piece buys)

| Model | Buys you |
|-------|----------|
| A PCNN | Baseline |
| B Residual | Near-zero NSRDB MAE (physics + r) |
| C + Gate | Interpretable α; still weak on Ninja |
| D + Calibrated UQ | Same preds as C; PICP → ~0.95 |
| E Multitask | **Only arm that lifts Ninja R²** |

### F. Multi-seed (Exp 4)

Open `exp04_multi_seed_runs.csv` **and** `exp04_multi_seed_summary.csv`.

**Current runs (reconciled):** seeds 42, 123, 456, 999, 2026 all succeed on Ninja  
(R² ≈ **0.86–0.90**; mean ± std ≈ **0.877 ± 0.014**). Seed **456 is not a failure** in the current CSV.

**How to read:**

1. Check **each seed’s Ninja R²** in the runs CSV first.
2. If a future re-run produces a negative R² outlier, say so explicitly — do not trust mean±std alone.
3. Do not confuse the **frozen fine-tune** failure (R² ≈ −0.51) with a multi-seed failure.

### G. Feature importance (Exp 6)

- **Physics-path AR** permutation ΔMAE is noise-scale (~10⁻⁶ W) because residuals are already ~10⁻⁴ W — **do not** narrate “top drivers” from that branch.
- **External CD head** shows real signal: GHI / DNI / Solar Zenith dominate — use that for interpretation.

### H. Noise / regimes (Phase 9)

- If **20% GHI noise** drops PCNN R² a lot but AR/CD barely move → physics prior helps on this label. Remember: still NSRDB physics target.
- Regime table: PCNN hurts most at **high GHI / afternoon / clear**; AR stays flat ~10⁻⁴ W MAE; CD can show higher RMSE in afternoon / medium_ghi (multitask cost).

---

## What “final success” looks like (checklist)

Tick these yourself:

- [ ] Phase 8 audit: **178/178 passed**
- [ ] Headline is **Ninja multitask** full R² ≈ **0.86**, val-holdout ≈ **0.81**
- [ ] NSRDB framed as physics-fidelity consistency (not a forecasting claim)
- [ ] Calibrated PICP ≈ **0.94–0.96** (not 0.999 raw PCNN)
- [ ] Ablation: only **E** improves Ninja; **D** fixes UQ
- [ ] No fabricated λ-sweep claims unless CSV has the full grid
- [ ] Multi-seed narrative matches `exp04_multi_seed_runs.csv`
- [ ] No unaligned paper MAE/R² reference numbers

---

## Files to read in order

1. `summary.md` — big picture  
2. `phase5_comparison.md` / `phase6_comparison.md` — main tables  
3. `phase7_validation.md` — ablation  
4. `phase8_audit.md` — “are numbers real?”  
5. `phase9_report.md` — extra analyses (claim-audited)  
6. `README.md` — architecture + figure gallery  
7. **This file** — commands + interpretation  

---

## One command to print a final scorecard

```powershell
python -c "import json,pandas as pd; p8=json.load(open('artifacts/metrics/phase8_audit.json',encoding='utf-8')); d=json.load(open('artifacts/metrics/ar_pcnn_cd_multitask_eval.json',encoding='utf-8')); print('SCORECARD'); print('Audit:', p8['publication_readiness']); print('CD Ninja full R2:', round(d['ninja_full_2019_external_head']['r2'],4)); print('CD Ninja holdout:', round(d['ninja_val_holdout']['r2'],4)); print('CD NSRDB MAE:', d['nsrdb_test']['mae']); print('Multi-seed runs:'); print(pd.read_csv('artifacts/phase9/csv/exp04_multi_seed_runs.csv').to_string(index=False)); print('Multi-seed summary:'); print(pd.read_csv('artifacts/phase9/csv/exp04_multi_seed_summary.csv').to_string(index=False))"
```
