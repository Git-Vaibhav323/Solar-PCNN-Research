"""Phase 6 trainers: Strategy A (multitask) and Strategy B (finetune)."""

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

from src.models.ar_pcnn_cd import ARPCNNCrossDomain
from src.models.cd_loss import CrossDomainLoss
from src.models.dataset_cd import CrossDomainDataset
from src.models.metrics import summarize_predictions
from src.training.train import (
    load_train_config,
    resolve_device,
    set_seed,
    target_scale_and_rated,
)


def _cfg(path: str | None) -> dict:
    return load_train_config(path or str(ROOT / "configs" / "train_ar_pcnn_cd.yaml"))


def _serialize_cfg(cfg: dict) -> dict:
    return json.loads(
        json.dumps(
            {
                **{k: cfg[k] for k in cfg if k != "paths"},
                "paths": {k: str(v) for k, v in cfg["paths"].items()},
            },
            default=str,
        )
    )


def make_loaders(cfg: dict):
    processed = cfg["paths"]["processed_dir"]
    ninja = cfg["paths"]["ninja_path"]
    target_scale, p_rated = target_scale_and_rated(cfg)
    feature_cols = cfg["features"]

    def ds(name: str) -> CrossDomainDataset:
        return CrossDomainDataset(
            processed / name,
            ninja,
            feature_cols,
            target_col=cfg["target"],
            physics_col=cfg["physics_col"],
            target_scale=target_scale,
            physics_rated=float(cfg["pv"]["p_rated_w"]),
        )

    train_ds, val_ds, test_ds = (
        ds("train_normalized.csv"),
        ds("val_normalized.csv"),
        ds("test_normalized.csv"),
    )
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    return (
        train_ds,
        val_ds,
        test_ds,
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw),
        p_rated,
    )


def build_model(cfg: dict, device: torch.device, strategy: str) -> ARPCNNCrossDomain:
    model = ARPCNNCrossDomain(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        dropout=float(cfg["model"]["dropout"]),
        p_rated=float(cfg["pv"]["p_rated_w"]),
        n_panels=int(cfg["pv"]["n_panels"]),
        system_capacity_kw=float(cfg["pv"]["system_capacity_kw"]),
    ).to(device)

    pretrained = Path(cfg["paths"].get("pretrained_ar_pcnn", ""))
    if cfg["train"].get("warm_start", True) and pretrained.exists():
        try:
            ckpt = torch.load(pretrained, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(pretrained, map_location=device)
        model.load_physics_pretrained(ckpt["model_state"])
        print(f"Warm-started physics branch from {pretrained}")

    if strategy == "finetune":
        if cfg["train"].get("freeze_encoder", True):
            model.freeze_encoder()
            print("Frozen encoder")
        if cfg["train"].get("freeze_physics_heads", True):
            model.freeze_physics_heads()
            print("Frozen physics residual/gate heads")
    return model


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, p_rated, capacity_margin) -> dict:
    model.eval()
    losses, phys_losses, ext_losses = [], [], []
    ys, preds, alphas, residuals = [], [], [], []
    ninja_t, ninja_p, masks = [], [], []
    for x, p_phys, y, ninja_kw, has_ext in loader:
        x = x.to(device)
        p_phys = p_phys.to(device)
        y = y.to(device)
        ninja_kw = ninja_kw.to(device)
        has_ext = has_ext.to(device)
        out = model(x, p_phys, return_details=True)
        br = criterion(
            out.physics.y_hat,
            y,
            out.physics.residual,
            out.physics.alpha,
            out.external_kw,
            ninja_kw,
            has_ext,
        )
        losses.append(br.total.item())
        phys_losses.append(br.physics_total.item())
        ext_losses.append(br.external_mse.item())
        ys.append(y.cpu().numpy())
        preds.append(out.physics.y_hat.cpu().numpy())
        alphas.append(out.physics.alpha.cpu().numpy())
        residuals.append(out.physics.residual.cpu().numpy())
        ninja_t.append(ninja_kw.cpu().numpy())
        ninja_p.append(out.external_kw.cpu().numpy())
        masks.append(has_ext.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    metrics = summarize_predictions(y_true, y_pred, p_rated, capacity_margin)
    metrics["loss"] = float(np.mean(losses))
    metrics["physics_loss"] = float(np.mean(phys_losses))
    metrics["external_mse"] = float(np.mean(ext_losses))
    metrics["mean_alpha"] = float(np.mean(np.concatenate(alphas)))
    metrics["mean_abs_residual"] = float(np.mean(np.abs(np.concatenate(residuals))))

    mask = np.concatenate(masks).reshape(-1) > 0.5
    if mask.any():
        nt = np.concatenate(ninja_t).reshape(-1)[mask]
        np_ = np.concatenate(ninja_p).reshape(-1)[mask]
        ext = summarize_predictions(
            nt, np_, p_rated=float(10.0), capacity_margin=capacity_margin
        )
        metrics["ninja_in_split"] = {f"ext_{k}": v for k, v in ext.items()}
        metrics["ninja_in_split"]["n"] = int(mask.sum())
    else:
        metrics["ninja_in_split"] = {"n": 0}
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device, strategy: str = "multitask") -> dict:
    model.train()
    # Keep frozen BatchNorm in eval if encoder frozen
    if not any(p.requires_grad for p in model.physics_branch.encoder.parameters()):
        model.physics_branch.encoder.eval()

    totals = {
        "loss": 0.0,
        "physics_loss": 0.0,
        "external_mse": 0.0,
        "mean_alpha": 0.0,
        "n_external": 0.0,
    }
    n = 0
    for x, p_phys, y, ninja_kw, has_ext in loader:
        x = x.to(device)
        p_phys = p_phys.to(device)
        y = y.to(device)
        ninja_kw = ninja_kw.to(device)
        has_ext = has_ext.to(device)

        # Strategy B: only external supervision; skip batches without ninja labels
        if strategy == "finetune" and (has_ext > 0.5).sum().item() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        out = model(x, p_phys, return_details=True)
        br = criterion(
            out.physics.y_hat,
            y,
            out.physics.residual,
            out.physics.alpha,
            out.external_kw,
            ninja_kw,
            has_ext,
        )
        if strategy == "finetune":
            # Physics path is frozen; backprop external head only
            loss = br.external_mse
            if not loss.requires_grad:
                continue
        else:
            loss = br.total
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        n += bs
        totals["loss"] += float(loss.detach().item()) * bs
        totals["physics_loss"] += br.physics_total.item() * bs
        totals["external_mse"] += br.external_mse.item() * bs
        totals["mean_alpha"] += br.mean_alpha.item() * bs
        totals["n_external"] += br.n_external.item()
    if n == 0:
        return {k: 0.0 for k in totals}
    out = {k: v / max(n, 1) for k, v in totals.items() if k != "n_external"}
    out["n_external"] = totals["n_external"]
    return out


def run_training(strategy: str, config_path: str | None = None) -> dict:
    cfg = _cfg(config_path)
    set_seed(int(cfg["train"]["seed"]))
    device = resolve_device(cfg["train"]["device"])
    cfg["paths"]["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    cfg["paths"]["metrics_dir"].mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, p_rated = make_loaders(cfg)
    capacity_margin = float(cfg["pv"]["capacity_margin"])
    model = build_model(cfg, device, strategy)

    criterion = CrossDomainLoss(
        p_rated=p_rated,
        capacity_margin=capacity_margin,
        lambda_capacity=float(cfg["loss"]["lambda_capacity"]),
        lambda_nonneg=float(cfg["loss"]["lambda_nonneg"]),
        lambda_residual=float(cfg["loss"]["lambda_residual"]),
        lambda_external=float(cfg["loss"]["lambda_external"]),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg["train"]["finetune_lr"] if strategy == "finetune" else cfg["train"]["learning_rate"])
    optimizer = torch.optim.Adam(
        params,
        lr=lr,
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

    ckpt_key = (
        "checkpoint_name_finetune" if strategy == "finetune" else "checkpoint_name_multitask"
    )
    best_path = cfg["paths"]["checkpoint_dir"] / cfg["train"][ckpt_key]
    last_path = cfg["paths"]["checkpoint_dir"] / best_path.name.replace("best", "last")

    print(f"Strategy: {strategy} | Device: {device}")
    print(
        f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)} | "
        f"external labels train/val: {train_ds.n_external}/{val_ds.n_external}"
    )

    history = []
    best_val = float("inf")
    stale = 0
    patience = int(cfg["train"]["early_stopping_patience"])
    t0 = time.time()

    for epoch in range(1, int(cfg["train"]["max_epochs"]) + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, strategy=strategy)
        va = evaluate_loader(model, val_loader, criterion, device, p_rated, capacity_margin)
        # Early stop on combined val loss; also track ninja-in-val if present
        monitor = va["loss"]
        if va.get("ninja_in_split", {}).get("n", 0) > 0:
            # Prefer improving external MAE when available
            monitor = 0.5 * va["loss"] + 0.5 * float(va["ninja_in_split"].get("ext_mae", va["loss"]))
        scheduler.step(va["loss"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": tr,
            "val_physics_mae": va["mae"],
            "val_physics_r2": va["r2"],
            "val_mean_alpha": va["mean_alpha"],
            "val_ninja": va.get("ninja_in_split", {}),
            "val_monitor": monitor,
        }
        history.append(row)
        ninja_r2 = va.get("ninja_in_split", {}).get("ext_r2", float("nan"))
        print(
            f"Epoch {epoch:03d} | loss={tr['loss']:.4f} phys={tr['physics_loss']:.4f} "
            f"ext_mse={tr['external_mse']:.4f} | val_MAE={va['mae']:.4f} val_R2={va['r2']:.5f} "
            f"alpha={va['mean_alpha']:.3f} | val_ninja_R2={ninja_r2 if ninja_r2==ninja_r2 else float('nan'):.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": _serialize_cfg(cfg),
            "strategy": strategy,
            "val_metrics": va,
            "model_name": f"ar_pcnn_cd_{strategy}",
        }
        torch.save(payload, last_path)
        if monitor < best_val - 1e-8:
            best_val = monitor
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
        "phase": 6,
        "strategy": strategy,
        "status": "complete",
        "device": str(device),
        "elapsed_sec": time.time() - t0,
        "best_epoch": int(ckpt["epoch"]),
        "best_checkpoint": str(best_path),
        "external_counts": {
            "train": train_ds.n_external,
            "val": val_ds.n_external,
            "test": test_ds.n_external,
        },
        "val": val_best,
        "test": test_stats,
    }
    tag = strategy
    with (cfg["paths"]["metrics_dir"] / f"ar_pcnn_cd_{tag}_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (cfg["paths"]["metrics_dir"] / f"ar_pcnn_cd_{tag}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"strategy": strategy, "val": val_best, "test_physics": {
        "mae": test_stats["mae"], "r2": test_stats["r2"], "mean_alpha": test_stats["mean_alpha"]
    }}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 cross-domain AR-PCNN training")
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_ar_pcnn_cd.yaml"))
    parser.add_argument(
        "--strategy",
        choices=["multitask", "finetune", "both"],
        default="both",
    )
    args = parser.parse_args()
    if args.strategy == "both":
        run_training("multitask", args.config)
        run_training("finetune", args.config)
    else:
        run_training(args.strategy, args.config)


if __name__ == "__main__":
    main()
