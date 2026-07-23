# Solar-PCNN Research

Reproduction and extension of *Physics-constrained neural networks for reliable solar power forecasting with uncertainty quantification* (Results in Engineering, 2026).

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset collection (NSRDB 2016–2020 + Renewables.ninja) | Done |
| 2 | Data preparation (merge, clean, EDA, normalize, split) | Done |
| 3 | Baseline PCNN reproduction | Next |
| 4 | Limitation analysis | Pending |
| 5 | Novel physics-aware extension | Pending |

## Quick start (Phase 2)

```bash
pip install -r requirements.txt
python -m src.data.pipeline
```

Outputs land in `data/processed/` and `data/figures/`.

## Data

- **NSRDB** (train/test): hourly meteorology at 13.06°N, 80.25°E (Location ID 326432), 2016–2020 → **43,848** hours.
- **Renewables.ninja** (external validation only): 2019 MERRA-2 PV series; not used for training.
- Paper cites VITCC (12.8438°N, 80.1533°E); the NSRDB point is ~25 km NE — documented in `phase2_summary.json`.

## Phase 2 pipeline (paper-aligned)

1. Merge yearly NSRDB CSVs chronologically; build local timestamps (UTC+5).
2. QA / light cleaning (duplicate timestamps, clip impossible negatives; keep Fill Flag rows).
3. Select 12 features from Table 4 / Eq. (6).
4. Compute physics targets (Eqs. 2–5) for Tata Power Solar **TS250** (`P_rated=250 W`, `γ=-0.004`).
5. Add cyclical hour encodings (`hour_sin`, `hour_cos`) → **14** model inputs (Table 6).
6. EDA figures (correlation heatmap, diurnal/seasonal profiles).
7. Chronological split: earliest 80% train window / latest 20% test; within train window, 80/20 train/val (Table 7 `val_ratio=0.20`).
8. Z-score fit on **train only**; apply to val/test.

### Model-ready artifacts

| File | Role |
|------|------|
| `data/processed/train_normalized.csv` | Training features (Z-scored) + targets |
| `data/processed/val_normalized.csv` | Validation |
| `data/processed/test_normalized.csv` | Held-out test |
| `data/processed/*_raw.csv` | Same splits before scaling |
| `data/processed/scaler.joblib` | Train-fitted `StandardScaler` |
| `data/processed/ninja_2019_prepared.csv` | External validation (Case 1) |
| `data/processed/phase2_summary.json` | Full run metadata |

Target column: `target_power` (= `P_bounded`).

## Next phase

Reproduce the original dual-branch PCNN (Section 4), train on the normalized splits, and match reported metrics before any novel extensions.
