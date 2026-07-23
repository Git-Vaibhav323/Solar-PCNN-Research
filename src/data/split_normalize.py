"""Chronological splitting and train-only Z-score normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def chronological_split(
    df: pd.DataFrame,
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Forward hold-out (Section 3.4):
      - earliest (1 - test_ratio) → train window
      - latest test_ratio → test
      - within train window, earliest (1 - val_ratio) → train, rest → validation
    No shuffling.
    """
    n = len(df)
    test_start = int(np.floor(n * (1.0 - test_ratio)))
    train_window = df.iloc[:test_start].copy()
    test = df.iloc[test_start:].copy()

    n_tw = len(train_window)
    val_start = int(np.floor(n_tw * (1.0 - val_ratio)))
    train = train_window.iloc[:val_start].copy()
    val = train_window.iloc[val_start:].copy()
    return train, val, test


def fit_normalize(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, dict[str, Any]]:
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[feature_cols] = scaler.transform(frame[feature_cols])
        return out

    meta = {
        "feature_cols": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_start": str(train["timestamp"].iloc[0]),
        "train_end": str(train["timestamp"].iloc[-1]),
        "val_start": str(val["timestamp"].iloc[0]),
        "val_end": str(val["timestamp"].iloc[-1]),
        "test_start": str(test["timestamp"].iloc[0]),
        "test_end": str(test["timestamp"].iloc[-1]),
    }
    return _apply(train), _apply(val), _apply(test), scaler, meta


def save_splits(
    processed_dir: Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    scaler: StandardScaler,
    meta: dict[str, Any],
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train_normalized.csv", index=False)
    val.to_csv(processed_dir / "val_normalized.csv", index=False)
    test.to_csv(processed_dir / "test_normalized.csv", index=False)
    train_raw.to_csv(processed_dir / "train_raw.csv", index=False)
    val_raw.to_csv(processed_dir / "val_raw.csv", index=False)
    test_raw.to_csv(processed_dir / "test_raw.csv", index=False)
    joblib.dump(scaler, processed_dir / "scaler.joblib")
    pd.Series(meta).to_json(processed_dir / "split_meta.json")
