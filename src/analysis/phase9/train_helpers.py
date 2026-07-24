"""Phase 9 training helper — writes only to artifacts/phase9/."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ar_pcnn_cd import ARPCNNCrossDomain
from src.models.cd_loss import CrossDomainLoss
from src.training.train import resolve_device, set_seed, target_scale_and_rated
from src.training.train_ar_pcnn_cd import (
    build_model,
    evaluate_loader,
    make_loaders,
    train_one_epoch,
)

from .common import CKPT_DIR, METRICS_DIR, phase9_cfg, save_cfg_yaml, save_json


def _serialize_cfg(cfg: dict) -> dict:
    return json.loads(
        json.dumps(
            {k: v for k, v in cfg.items() if k != "_phase9_tag"},
            default=lambda o: str(o) if isinstance(o, Path) else o,
        )
    )


def run_multitask_training(
    *,
    lambda_external: float | None = None,
    seed: int | None = None,
    tag: str = "",
    max_epochs: int | None = None,
) -> dict:
    """
    Train multitask CD model into Phase 9 checkpoint/metrics dirs only.
    Does not touch Phase 6 checkpoints.
    """
    cfg = phase9_cfg(lambda_external=lambda_external, seed=seed, tag=tag)
    tag = cfg["_phase9_tag"]
    save_cfg_yaml(cfg, f"train_{tag}.yaml")

    set_seed(int(cfg["train"]["seed"]))
    device = resolve_device(cfg["train"]["device"])
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, p_rated = make_loaders(cfg)
    capacity_margin = float(cfg["pv"]["capacity_margin"])
    model = build_model(cfg, device, "multitask")

    criterion = CrossDomainLoss(
        p_rated=p_rated,
        capacity_margin=capacity_margin,
        lambda_capacity=float(cfg["loss"]["lambda_capacity"]),
        lambda_nonneg=float(cfg["loss"]["lambda_nonneg"]),
        lambda_residual=float(cfg["loss"]["lambda_residual"]),
        lambda_external=float(cfg["loss"]["lambda_external"]),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
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

    best_path = CKPT_DIR / cfg["train"]["checkpoint_name_multitask"]
    last_path = CKPT_DIR / best_path.name.replace("best", "last")
    epochs = int(max_epochs or cfg["train"]["max_epochs"])
    patience = int(cfg["train"]["early_stopping_patience"])

    history = []
    best_val = float("inf")
    stale = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, strategy="multitask")
        va = evaluate_loader(model, val_loader, criterion, device, p_rated, capacity_margin)
        monitor = va["loss"]
        if va.get("ninja_in_split", {}).get("n", 0) > 0:
            monitor = 0.5 * va["loss"] + 0.5 * float(va.get("ninja_in_split", {}).get("ext_mae", va["loss"]))
        scheduler.step(va["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_physics_loss": tr["physics_loss"],
                "train_external_mse": tr["external_mse"],
                "val_loss": va["loss"],
                "val_physics_mae": va["mae"],
                "val_physics_r2": va["r2"],
                "val_ninja_r2": va.get("ninja_in_split", {}).get("ext_r2"),
            }
        )
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": _serialize_cfg(cfg),
            "strategy": "multitask",
            "val_metrics": va,
            "tag": tag,
        }
        torch.save(payload, last_path)
        if monitor < best_val - 1e-8:
            best_val = monitor
            stale = 0
            torch.save(payload, best_path)
        else:
            stale += 1
            if stale >= patience:
                break

    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_best = evaluate_loader(model, val_loader, criterion, device, p_rated, capacity_margin)
    test_stats = evaluate_loader(model, test_loader, criterion, device, p_rated, capacity_margin)

    summary = {
        "tag": tag,
        "lambda_external": float(cfg["loss"]["lambda_external"]),
        "seed": int(cfg["train"]["seed"]),
        "elapsed_sec": time.time() - t0,
        "best_epoch": int(ckpt["epoch"]),
        "best_checkpoint": str(best_path),
        "train_loss_final": history[-1]["train_loss"] if history else None,
        "val_loss_final": val_best["loss"],
        "val": val_best,
        "test": test_stats,
    }
    save_json(history, f"history_{tag}.json")
    save_json(summary, f"summary_{tag}.json")
    return summary
