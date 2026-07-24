"""Train Phase-7 Model B: residual-only PCNN (alpha fixed = 1)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ar_loss import AdaptiveResidualLoss
from src.models.dataset import SolarPCNNDataset
from src.models.metrics import summarize_predictions
from src.models.residual_only_pcnn import ResidualOnlyPCNN
from src.training.train import (
    load_train_config,
    resolve_device,
    set_seed,
    target_scale_and_rated,
)


def run(config_path: str | None = None) -> dict:
    # Reuse AR-PCNN train hyperparams
    cfg = load_train_config(config_path or str(ROOT / "configs" / "train_ar_pcnn.yaml"))
    set_seed(42)
    device = resolve_device(cfg["train"]["device"])
    cfg["paths"]["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    cfg["paths"]["metrics_dir"].mkdir(parents=True, exist_ok=True)

    target_scale, p_rated = target_scale_and_rated(cfg)
    capacity_margin = float(cfg["pv"]["capacity_margin"])

    def ds(name):
        return SolarPCNNDataset(
            cfg["paths"]["processed_dir"] / name,
            cfg["features"],
            target_col=cfg["target"],
            physics_col=cfg["physics_col"],
            target_scale=target_scale,
            physics_rated=float(cfg["pv"]["p_rated_w"]),
        )

    train_ds, val_ds, test_ds = ds("train_normalized.csv"), ds("val_normalized.csv"), ds("test_normalized.csv")
    bs = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    model = ResidualOnlyPCNN(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        dropout=float(cfg["model"]["dropout"]),
        p_rated=p_rated,
    ).to(device)
    criterion = AdaptiveResidualLoss(
        p_rated=p_rated,
        capacity_margin=capacity_margin,
        lambda_capacity=float(cfg["loss"]["lambda_capacity"]),
        lambda_nonneg=float(cfg["loss"]["lambda_nonneg"]),
        lambda_residual=float(cfg["loss"]["lambda_residual"]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["learning_rate"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=7, min_lr=1e-6)

    best = float("inf")
    stale = 0
    best_path = cfg["paths"]["checkpoint_dir"] / "ablation_residual_only_best.pt"
    t0 = time.time()

    print(f"Model B residual-only | device={device}")
    for epoch in range(1, int(cfg["train"]["max_epochs"]) + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        for x, p, y in train_loader:
            x, p, y = x.to(device), p.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, p, return_details=True)
            br = criterion(out.y_hat, y, out.residual, out.alpha)
            br.total.backward()
            opt.step()
            tr_loss += br.total.item() * x.size(0)
            n += x.size(0)

        model.eval()
        vals, preds = [], []
        vloss = 0.0
        vn = 0
        with torch.no_grad():
            for x, p, y in val_loader:
                x, p, y = x.to(device), p.to(device), y.to(device)
                out = model(x, p, return_details=True)
                br = criterion(out.y_hat, y, out.residual, out.alpha)
                vloss += br.total.item() * x.size(0)
                vn += x.size(0)
                vals.append(y.cpu().numpy())
                preds.append(out.y_hat.cpu().numpy())
        vloss /= max(vn, 1)
        vm = summarize_predictions(np.concatenate(vals), np.concatenate(preds), p_rated, capacity_margin)
        sched.step(vloss)
        print(f"Epoch {epoch:03d} | train={tr_loss/max(n,1):.5f} val={vloss:.5f} MAE={vm['mae']:.4f} R2={vm['r2']:.5f}")
        payload = {"epoch": epoch, "model_state": model.state_dict(), "val_metrics": vm, "model_name": "residual_only"}
        if vloss < best - 1e-8:
            best = vloss
            stale = 0
            torch.save(payload, best_path)
        else:
            stale += 1
            if stale >= int(cfg["train"]["early_stopping_patience"]):
                print(f"Early stopping at {epoch}")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, p, y in test_loader:
            ys.append(y.numpy())
            ps.append(model(x.to(device), p.to(device)).cpu().numpy())
    test = summarize_predictions(np.concatenate(ys), np.concatenate(ps), p_rated, capacity_margin)
    np.savez_compressed(
        cfg["paths"]["metrics_dir"] / "ablation_residual_only_predictions_test.npz",
        y_true=np.concatenate(ys),
        y_pred=np.concatenate(ps),
    )
    summary = {
        "model": "B_residual_only",
        "best_epoch": ckpt["epoch"],
        "elapsed_sec": time.time() - t0,
        "test": test,
        "checkpoint": str(best_path),
    }
    with (cfg["paths"]["metrics_dir"] / "ablation_residual_only_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
