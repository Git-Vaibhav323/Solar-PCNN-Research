"""PyTorch dataset for Phase-2 normalized NSRDB splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class SolarPCNNDataset(Dataset):
    def __init__(
        self,
        csv_path: Path | str,
        feature_cols: list[str],
        target_col: str = "target_power",
        physics_col: str = "P_theoretical",
        target_scale: float = 1.0,
        physics_rated: float = 250.0,
    ) -> None:
        df = pd.read_csv(csv_path)
        missing = [c for c in feature_cols + [target_col, physics_col] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {csv_path}: {missing}")

        self.features = torch.tensor(df[feature_cols].to_numpy(dtype="float32"), dtype=torch.float32)
        self.targets = torch.tensor(
            (df[target_col].to_numpy(dtype="float32") / target_scale),
            dtype=torch.float32,
        ).unsqueeze(1)
        # Physics prior scaled by rated power (dimensionless-ish ~[0, 1.1])
        self.physics = torch.tensor(
            (df[physics_col].to_numpy(dtype="float32") / physics_rated),
            dtype=torch.float32,
        ).unsqueeze(1)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.physics[idx], self.targets[idx]
