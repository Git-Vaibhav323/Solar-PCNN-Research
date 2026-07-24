"""Evaluate AR-PCNN checkpoint on a chronological split."""

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

from src.models.ar_pcnn import ARPCNN
from src.models.dataset import SolarPCNNDataset
from src.models.metrics import summarize_predictions
from src.training.train import load_train_config, resolve_device, target_scale_and_rated


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds, alphas, residuals = [], [], [], []
    for x, p_phys, y in loader:
        out = model(x.to(device), p_phys.to(device), return_details=True)
        ys.append(y.numpy())
        preds.append(out.y_hat.cpu().numpy())
        alphas.append(out.alpha.cpu().numpy())
        residuals.append(out.residual.cpu().numpy())
    return (
        np.concatenate(ys, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(alphas, axis=0),
        np.concatenate(residuals, axis=0),
    )


def run_eval(checkpoint: str | None = None, config_path: str | None = None, split: str = "test") -> dict:
    cfg = load_train_config(config_path or str(ROOT / "configs" / "train_ar_pcnn.yaml"))
    device = resolve_device(cfg["train"]["device"])
    target_scale, p_rated = target_scale_and_rated(cfg)
    ckpt_path = Path(checkpoint) if checkpoint else cfg["paths"]["checkpoint_dir"] / "ar_pcnn_best.pt"

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    model = ARPCNN(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        dropout=float(cfg["model"]["dropout"]),
        p_rated=p_rated,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    csv_name = {"train": "train_normalized.csv", "val": "val_normalized.csv", "test": "test_normalized.csv"}[split]
    ds = SolarPCNNDataset(
        cfg["paths"]["processed_dir"] / csv_name,
        cfg["features"],
        target_col=cfg["target"],
        physics_col=cfg["physics_col"],
        target_scale=target_scale,
        physics_rated=float(cfg["pv"]["p_rated_w"]),
    )
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    y_true, y_pred, alpha, residual = predict(model, loader, device)
    metrics = summarize_predictions(
        y_true, y_pred, p_rated, float(cfg["pv"]["capacity_margin"])
    )
    metrics["mean_alpha"] = float(np.mean(alpha))
    metrics["mean_abs_residual"] = float(np.mean(np.abs(residual)))

    out_dir = cfg["paths"]["metrics_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"ar_pcnn_predictions_{split}.npz",
        y_true=y_true,
        y_pred=y_pred,
        alpha=alpha,
        residual=residual,
    )
    with (out_dir / f"ar_pcnn_metrics_{split}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AR-PCNN")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_ar_pcnn.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    run_eval(args.checkpoint, args.config, args.split)


if __name__ == "__main__":
    main()
