"""Phase 2 end-to-end preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean import clean_nsrdb, quality_report
from src.data.config import load_config
from src.data.features import add_temporal_features, model_feature_columns, select_features
from src.data.merge import merge_nsrdb_years
from src.data.physics import add_physics_targets
from src.data.split_normalize import chronological_split, fit_normalize, save_splits
from src.eda.generate_eda import run_eda


def prepare_ninja(cfg: dict) -> dict:
    """Load Renewables.ninja for later external validation (not used in training)."""
    import pandas as pd

    path = cfg["paths"]["ninja_path"]
    ninja = pd.read_csv(path, skiprows=3)
    ninja["timestamp"] = pd.to_datetime(ninja["time"], utc=True)
    # Align to local site time (+5) for any future joint analysis
    ninja["timestamp_local"] = ninja["timestamp"].dt.tz_convert("Asia/Kolkata")
    out = cfg["paths"]["processed_dir"] / "ninja_2019_prepared.csv"
    ninja.to_csv(out, index=False)
    return {
        "rows": len(ninja),
        "path": str(out),
        "time_start_utc": str(ninja["timestamp"].iloc[0]),
        "time_end_utc": str(ninja["timestamp"].iloc[-1]),
    }


def run_pipeline(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = cfg["features"]["columns"]
    site = cfg["site"]

    # 1. Merge
    merged = merge_nsrdb_years(
        nsrdb_dir=cfg["paths"]["nsrdb_dir"],
        years=cfg["years"],
        location_id=site["location_id"],
        latitude=site["latitude"],
        longitude=site["longitude"],
        tz_offset_hours=site["timezone_offset_hours"],
    )
    merged_path = processed_dir / "nsrdb_2016_2020_merged.csv"
    merged.to_csv(merged_path, index=False)

    # 2. Clean + QA
    cleaned, qa = clean_nsrdb(merged, feature_cols)
    selected = select_features(cleaned, feature_cols)

    # 3. Physics targets
    with_physics = add_physics_targets(selected, cfg["pv"])

    # 4. Feature engineering
    if cfg.get("temporal_features", True):
        with_physics = add_temporal_features(with_physics)
    model_cols = model_feature_columns(feature_cols, include_temporal=cfg.get("temporal_features", True))

    full_path = processed_dir / "nsrdb_model_ready_raw.csv"
    with_physics.to_csv(full_path, index=False)

    # 5. EDA on raw (unnormalized) features
    eda_summary = run_eda(with_physics, feature_cols, figures_dir)

    # 6. Chronological split + train-only Z-score
    train_raw, val_raw, test_raw = chronological_split(
        with_physics,
        test_ratio=cfg["split"]["test_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
    )
    train_n, val_n, test_n, scaler, split_meta = fit_normalize(
        train_raw, val_raw, test_raw, model_cols
    )
    save_splits(
        processed_dir,
        train_n,
        val_n,
        test_n,
        train_raw,
        val_raw,
        test_raw,
        scaler,
        split_meta,
    )

    ninja_meta = prepare_ninja(cfg)

    summary = {
        "phase": 2,
        "status": "complete",
        "merged_rows": len(merged),
        "model_ready_rows": len(with_physics),
        "feature_cols_12": feature_cols,
        "model_input_cols": model_cols,
        "qa": qa,
        "split": split_meta,
        "eda": eda_summary,
        "ninja": ninja_meta,
        "artifacts": {
            "merged": str(merged_path),
            "model_ready_raw": str(full_path),
            "train_normalized": str(processed_dir / "train_normalized.csv"),
            "val_normalized": str(processed_dir / "val_normalized.csv"),
            "test_normalized": str(processed_dir / "test_normalized.csv"),
            "scaler": str(processed_dir / "scaler.joblib"),
        },
        "notes": {
            "site_vs_paper": (
                f"NSRDB point ({site['latitude']}, {site['longitude']}) vs "
                f"paper VITCC ({site['paper_latitude']}, {site['paper_longitude']})"
            ),
            "precipitable_water_units": "NSRDB provides cm; paper Table 4 lists mm — values kept as NSRDB",
            "pressure_units": "NSRDB mbar ≈ hPa",
            "target": "P_bounded from Eqs. (2)–(5); γ=-0.004 as in Eq. (3)",
        },
    }

    summary_path = processed_dir / "phase2_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    qa_path = processed_dir / "qa_report.json"
    with qa_path.open("w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2, default=str)

    print(json.dumps({k: summary[k] for k in ("status", "merged_rows", "split", "artifacts")}, indent=2, default=str))
    print(f"\nWrote summary -> {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: prepare model-ready NSRDB dataset")
    parser.add_argument("--config", default=None, help="Path to preprocess.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
