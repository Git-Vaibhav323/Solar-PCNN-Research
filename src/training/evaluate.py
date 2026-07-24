"""Evaluate a trained PCNN checkpoint on val/test (and optional ninja later)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.dataset import SolarPCNNDataset
from src.models.loss import PhysicsConstrainedLoss
from src.models.metrics import summarize_predictions
from src.models.pcnn import PCNN
from src.training.train import load_train_config, resolve_device, target_scale_and_rated


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, p_phys, y in loader:
        y_hat = model(x.to(device), p_phys.to(device))
        ys.append(y.numpy())
        preds.append(y_hat.cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(preds, axis=0)


def run_eval(checkpoint: str, config_path: str | None = None, split: str = "test") -> dict:
    cfg = load_train_config(config_path)
    device = resolve_device(cfg["train"]["device"])
    target_scale, p_rated = target_scale_and_rated(cfg)
    capacity_margin = float(cfg["pv"]["capacity_margin"])

    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    model = PCNN(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        fuse_dim=int(cfg["model"]["fuse_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    csv_name = {
        "train": "train_normalized.csv",
        "val": "val_normalized.csv",
        "test": "test_normalized.csv",
    }[split]
    ds = SolarPCNNDataset(
        cfg["paths"]["processed_dir"] / csv_name,
        cfg["features"],
        target_col=cfg["target"],
        physics_col=cfg["physics_col"],
        target_scale=target_scale,
        physics_rated=float(cfg["pv"]["p_rated_w"]),
    )
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    y_true, y_pred = predict(model, loader, device)
    metrics = summarize_predictions(y_true, y_pred, p_rated, capacity_margin)

    out_dir = cfg["paths"]["metrics_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{split}.npz"
    np.savez_compressed(pred_path, y_true=y_true, y_pred=y_pred)
    metrics_path = out_dir / f"metrics_{split}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions -> {pred_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline PCNN")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts" / "checkpoints" / "pcnn_best.pt"),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    run_eval(args.checkpoint, args.config, args.split)


if __name__ == "__main__":
    main()
