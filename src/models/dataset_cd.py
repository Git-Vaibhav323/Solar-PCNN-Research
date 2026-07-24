"""Datasets that attach optional Renewables.ninja labels to Phase-2 splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_ninja_lookup(ninja_path: Path) -> pd.DataFrame:
    ninja = pd.read_csv(ninja_path)
    ts = pd.to_datetime(ninja["time"], utc=True).dt.tz_convert("Etc/GMT-5").dt.floor("h")
    out = pd.DataFrame({"align_ts": ts, "ninja_kw": ninja["electricity"].astype(float)})
    return out.drop_duplicates("align_ts").set_index("align_ts")


def attach_ninja_labels(split_csv: Path, ninja_path: Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Etc/GMT-5").dt.floor("h")
    df = df.copy()
    df["align_ts"] = ts
    lookup = load_ninja_lookup(ninja_path)
    df["ninja_kw"] = df["align_ts"].map(lookup["ninja_kw"])
    df["has_external"] = df["ninja_kw"].notna().astype(np.float32)
    df["ninja_kw"] = df["ninja_kw"].fillna(0.0)
    return df


class CrossDomainDataset(Dataset):
    """
    Same feature/physics/target contract as SolarPCNNDataset, plus optional ninja kW.

    has_external == 1 only for aligned 2019 hours; external loss should be masked.
    """

    def __init__(
        self,
        split_csv: Path | str,
        ninja_path: Path | str,
        feature_cols: list[str],
        target_col: str = "target_power",
        physics_col: str = "P_theoretical",
        target_scale: float = 1.0,
        physics_rated: float = 250.0,
    ) -> None:
        df = attach_ninja_labels(Path(split_csv), Path(ninja_path))
        missing = [c for c in feature_cols + [target_col, physics_col] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {split_csv}: {missing}")

        self.features = torch.tensor(df[feature_cols].to_numpy(dtype="float32"), dtype=torch.float32)
        self.targets = torch.tensor(
            (df[target_col].to_numpy(dtype="float32") / target_scale),
            dtype=torch.float32,
        ).unsqueeze(1)
        self.physics = torch.tensor(
            (df[physics_col].to_numpy(dtype="float32") / physics_rated),
            dtype=torch.float32,
        ).unsqueeze(1)
        self.ninja_kw = torch.tensor(df["ninja_kw"].to_numpy(dtype="float32"), dtype=torch.float32).unsqueeze(1)
        self.has_external = torch.tensor(
            df["has_external"].to_numpy(dtype="float32"), dtype=torch.float32
        ).unsqueeze(1)
        self.timestamps = df["align_ts"].astype(str).tolist()
        self.n_external = int(df["has_external"].sum())

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.features[idx],
            self.physics[idx],
            self.targets[idx],
            self.ninja_kw[idx],
            self.has_external[idx],
        )
