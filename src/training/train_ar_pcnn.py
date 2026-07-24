"""Train Phase-5 AR-PCNN (baseline PCNN training code left untouched)."""

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
from src.models.ar_pcnn import ARPCNN
from src.models.dataset import SolarPCNNDataset
from src.models.metrics import summarize_predictions
from src.training.train import (
    load_train_config,
    resolve_device,
    set_seed,
    target_scale_and_rated,
)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, p_rated, capacity_margin) -> dict:
    model.eval()
    losses, ys, preds, alphas, residuals = [], [], [], [], []
    for x, p_phys, y in loader:
        x, p_phys, y = x.to(device), p_phys.to(device), y.to(device)
        out = model(x, p_phys, return_details=True)
        br = criterion(out.y_hat, y, out.residual, out.alpha)
        losses.append(br.total.item())
        ys.append(y.cpu().numpy())
        preds.append(out.y_hat.cpu().numpy())
        alphas.append(out.alpha.cpu().numpy())
        residuals.append(out.residual.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    metrics = summarize_predictions(y_true, y_pred, p_rated, capacity_margin)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["mean_alpha"] = float(np.mean(np.concatenate(alphas)))
    metrics["mean_abs_residual"] = float(np.mean(np.abs(np.concatenate(residuals))))
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "residual_reg": 0.0, "mean_alpha": 0.0}
    n = 0
    for x, p_phys, y in loader:
        x, p_phys, y = x.to(device), p_phys.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x, p_phys, return_details=True)
        br = criterion(out.y_hat, y, out.residual, out.alpha)
        br.total.backward()
        optimizer.step()
        bs = x.size(0)
        n += bs
        totals["loss"] += br.total.item() * bs
        totals["mse"] += br.mse.item() * bs
        totals["residual_reg"] += br.residual_reg.item() * bs
        totals["mean_alpha"] += br.mean_alpha.item() * bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def run_training(config_path: str | None = None) -> dict:
    cfg_path = config_path or str(ROOT / "configs" / "train_ar_pcnn.yaml")
    cfg = load_train_config(cfg_path)
    set_seed(int(cfg["train"]["seed"]))
    device = resolve_device(cfg["train"]["device"])

    processed = cfg["paths"]["processed_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_scale, p_rated = target_scale_and_rated(cfg)
    capacity_margin = float(cfg["pv"]["capacity_margin"])
    feature_cols = cfg["features"]

    def make_ds(name: str) -> SolarPCNNDataset:
        return SolarPCNNDataset(
            processed / name,
            feature_cols,
            target_col=cfg["target"],
            physics_col=cfg["physics_col"],
            target_scale=target_scale,
            physics_rated=float(cfg["pv"]["p_rated_w"]),
        )

    train_ds, val_ds, test_ds = (
        make_ds("train_normalized.csv"),
        make_ds("val_normalized.csv"),
        make_ds("test_normalized.csv"),
    )
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    model = ARPCNN(
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        betas=tuple(cfg["train"]["adam_betas"]),
        eps=float(cfg["train"]["adam_eps"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["train"]["lr_scheduler_factor"]),
        patience=int(cfg["train"]["lr_scheduler_patience"]),
        min_lr=float(cfg["train"]["min_lr"]),
    )

    best_path = ckpt_dir / cfg["train"].get("checkpoint_name", "ar_pcnn_best.pt")
    last_path = ckpt_dir / "ar_pcnn_last.pt"
    history = []
    best_val = float("inf")
    stale = 0
    patience = int(cfg["train"]["early_stopping_patience"])
    t0 = time.time()

    print(f"Model: AR-PCNN | Device: {device}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    for epoch in range(1, int(cfg["train"]["max_epochs"]) + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate_loader(
            model, val_loader, criterion, device, p_rated, capacity_margin
        )
        scheduler.step(val_stats["loss"])
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_mean_alpha": train_stats["mean_alpha"],
            "val_loss": val_stats["loss"],
            "val_mae": val_stats["mae"],
            "val_rmse": val_stats["rmse"],
            "val_r2": val_stats["r2"],
            "val_mape": val_stats.get("mape"),
            "val_mean_alpha": val_stats["mean_alpha"],
            "val_mean_abs_residual": val_stats["mean_abs_residual"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_stats['loss']:.5f} | "
            f"val_loss={val_stats['loss']:.5f} | MAE={val_stats['mae']:.4f} | "
            f"R2={val_stats['r2']:.5f} | alpha={val_stats['mean_alpha']:.3f} | "
            f"|r|={val_stats['mean_abs_residual']:.3f} | lr={lr:.2e}"
        )

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": json.loads(
                json.dumps(
                    {
                        **{k: cfg[k] for k in cfg if k != "paths"},
                        "paths": {k: str(v) for k, v in cfg["paths"].items()},
                    },
                    default=str,
                )
            ),
            "val_metrics": val_stats,
            "model_name": "ar_pcnn",
        }
        torch.save(payload, last_path)

        if val_stats["loss"] < best_val - 1e-8:
            best_val = val_stats["loss"]
            stale = 0
            torch.save(payload, best_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_best = evaluate_loader(model, val_loader, criterion, device, p_rated, capacity_margin)
    test_stats = evaluate_loader(model, test_loader, criterion, device, p_rated, capacity_margin)

    summary = {
        "phase": 5,
        "model": "ar_pcnn",
        "status": "complete",
        "device": str(device),
        "elapsed_sec": time.time() - t0,
        "best_epoch": int(ckpt["epoch"]),
        "best_checkpoint": str(best_path),
        "val": val_best,
        "test": test_stats,
    }
    with (metrics_dir / "ar_pcnn_train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (metrics_dir / "ar_pcnn_phase5_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== AR-PCNN best ===")
    print(f"Val  MAE={val_best['mae']:.4f} RMSE={val_best['rmse']:.4f} R2={val_best['r2']:.5f} MAPE={val_best.get('mape', float('nan')):.3f}")
    print(f"Test MAE={test_stats['mae']:.4f} RMSE={test_stats['rmse']:.4f} R2={test_stats['r2']:.5f} MAPE={test_stats.get('mape', float('nan')):.3f}")
    print(f"Test mean alpha={test_stats['mean_alpha']:.3f} | mean |r|={test_stats['mean_abs_residual']:.3f} W")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase-5 AR-PCNN")
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_ar_pcnn.yaml"))
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
