"""Train the baseline PCNN on Phase-2 chronological splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.dataset import SolarPCNNDataset
from src.models.loss import PhysicsConstrainedLoss
from src.models.metrics import summarize_predictions
from src.models.pcnn import PCNN


def load_train_config(path: Path | str | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "configs" / "train_pcnn.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    path_keys = (
        "processed_dir",
        "checkpoint_dir",
        "metrics_dir",
        "figures_dir",
        "scaler_path",
        "ninja_path",
        "pretrained_ar_pcnn",
    )
    for key in path_keys:
        if key not in cfg.get("paths", {}):
            continue
        raw = cfg["paths"][key]
        p = Path(raw)
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def target_scale_and_rated(cfg: dict) -> tuple[float, float]:
    """Return (divide_targets_by, p_rated_in_target_units)."""
    unit = cfg["train"].get("target_unit", "W")
    p_rated_w = float(cfg["pv"]["p_rated_w"])
    if unit == "kW":
        return 1000.0, p_rated_w / 1000.0
    return 1.0, p_rated_w


@torch.no_grad()
def evaluate_loader(
    model: PCNN,
    loader: DataLoader,
    criterion: PhysicsConstrainedLoss,
    device: torch.device,
    p_rated: float,
    capacity_margin: float,
) -> dict:
    model.eval()
    losses = []
    ys, preds = [], []
    for x, p_phys, y in loader:
        x = x.to(device)
        p_phys = p_phys.to(device)
        y = y.to(device)
        y_hat = model(x, p_phys)
        br = criterion(y_hat, y)
        losses.append(br.total.item())
        ys.append(y.cpu().numpy())
        preds.append(y_hat.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    metrics = summarize_predictions(y_true, y_pred, p_rated, capacity_margin)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def train_one_epoch(
    model: PCNN,
    loader: DataLoader,
    criterion: PhysicsConstrainedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "capacity": 0.0, "nonneg": 0.0}
    n = 0
    for x, p_phys, y in loader:
        x = x.to(device)
        p_phys = p_phys.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x, p_phys)
        br = criterion(y_hat, y)
        br.total.backward()
        optimizer.step()
        bs = x.size(0)
        n += bs
        totals["loss"] += br.total.item() * bs
        totals["mse"] += br.mse.item() * bs
        totals["capacity"] += br.capacity.item() * bs
        totals["nonneg"] += br.nonneg.item() * bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def run_training(config_path: str | None = None) -> dict:
    cfg = load_train_config(config_path)
    set_seed(int(cfg["train"]["seed"]))
    device = resolve_device(cfg["train"]["device"])

    processed = cfg["paths"]["processed_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_scale, p_rated = target_scale_and_rated(cfg)
    feature_cols = cfg["features"]
    capacity_margin = float(cfg["pv"]["capacity_margin"])

    train_ds = SolarPCNNDataset(
        processed / "train_normalized.csv",
        feature_cols,
        target_col=cfg["target"],
        physics_col=cfg["physics_col"],
        target_scale=target_scale,
        physics_rated=float(cfg["pv"]["p_rated_w"]),
    )
    val_ds = SolarPCNNDataset(
        processed / "val_normalized.csv",
        feature_cols,
        target_col=cfg["target"],
        physics_col=cfg["physics_col"],
        target_scale=target_scale,
        physics_rated=float(cfg["pv"]["p_rated_w"]),
    )
    test_ds = SolarPCNNDataset(
        processed / "test_normalized.csv",
        feature_cols,
        target_col=cfg["target"],
        physics_col=cfg["physics_col"],
        target_scale=target_scale,
        physics_rated=float(cfg["pv"]["p_rated_w"]),
    )

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    model = PCNN(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        fuse_dim=int(cfg["model"]["fuse_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    criterion = PhysicsConstrainedLoss(
        p_rated=p_rated,
        capacity_margin=capacity_margin,
        lambda_capacity=float(cfg["loss"]["lambda_capacity"]),
        lambda_nonneg=float(cfg["loss"]["lambda_nonneg"]),
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

    history = []
    best_val = float("inf")
    best_path = ckpt_dir / "pcnn_best.pt"
    last_path = ckpt_dir / "pcnn_last.pt"
    patience = int(cfg["train"]["early_stopping_patience"])
    stale = 0
    t0 = time.time()

    print(f"Device: {device}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Target unit: {cfg['train'].get('target_unit', 'W')} | P_rated={p_rated}")

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
            "train_mse": train_stats["mse"],
            "val_loss": val_stats["loss"],
            "val_mae": val_stats["mae"],
            "val_rmse": val_stats["rmse"],
            "val_r2": val_stats["r2"],
            "val_feasible": val_stats["feasible_fraction"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{cfg['train']['max_epochs']} | "
            f"train_loss={train_stats['loss']:.5f} | "
            f"val_loss={val_stats['loss']:.5f} | "
            f"val_MAE={val_stats['mae']:.5f} | "
            f"val_RMSE={val_stats['rmse']:.5f} | "
            f"val_R2={val_stats['r2']:.5f} | lr={lr:.2e}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "val_metrics": val_stats,
            },
            last_path,
        )

        if val_stats["loss"] < best_val - 1e-8:
            best_val = val_stats["loss"]
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "val_metrics": val_stats,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Load best and evaluate test
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_stats = evaluate_loader(
        model, test_loader, criterion, device, p_rated, capacity_margin
    )
    val_best = evaluate_loader(
        model, val_loader, criterion, device, p_rated, capacity_margin
    )

    elapsed = time.time() - t0
    summary = {
        "phase": 3,
        "status": "complete",
        "device": str(device),
        "elapsed_sec": elapsed,
        "best_epoch": int(ckpt["epoch"]),
        "best_checkpoint": str(best_path),
        "target_unit": cfg["train"].get("target_unit", "W"),
        "val": val_best,
        "test": test_stats,
        "paper_metric_alignment": "not_reconcilable",
        "target_scale": {
            "nsrdb": "single TS250 panel Watts (p_rated_w=250)",
            "case1_ninja": "10 kW system = 40 × 250 W panels (electricity in kW)",
        },
        "note": (
            "Do not align our NSRDB MAE/RMSE to published paper MAE figures (e.g. 0.0213). "
            "Our NSRDB target is single-panel Watts; Case-1/Ninja is a 10 kW system in kW. "
            "The source paper does not specify system capacity / unit for its reported MAE in a "
            "way that permits safe conversion — precluding unit reconciliation. "
            "Report our protocol metrics only; never present a guessed W↔kW conversion as fact."
        ),
    }

    hist_path = metrics_dir / "train_history.json"
    sum_path = metrics_dir / "phase3_summary.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Best model ===")
    print(f"Best epoch: {ckpt['epoch']}")
    print(f"Val  MAE={val_best['mae']:.5f} RMSE={val_best['rmse']:.5f} R2={val_best['r2']:.5f}")
    print(f"Test MAE={test_stats['mae']:.5f} RMSE={test_stats['rmse']:.5f} R2={test_stats['r2']:.5f}")
    print(f"Physics feasible (test): {test_stats['feasible_fraction']:.4f}")
    print(f"Wrote {sum_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline PCNN (Phase 3)")
    parser.add_argument("--config", default=None, help="Path to train_pcnn.yaml")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
