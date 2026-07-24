"""
Evaluate Phase-6 cross-domain models on:
  - NSRDB physics test split
  - Ninja holdout (2019 hours inside val split only)
  - Ninja full 2019 (Phase-5 Case-1 protocol, for apples-to-apples)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ar_pcnn_cd import ARPCNNCrossDomain
from src.models.dataset_cd import CrossDomainDataset, attach_ninja_labels
from src.models.metrics import summarize_predictions
from src.training.case1_ninja import build_case1_frame
from src.training.train import load_train_config, resolve_device, target_scale_and_rated


def load_cd_model(cfg, ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model = ARPCNNCrossDomain(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        dropout=float(cfg["model"]["dropout"]),
        p_rated=float(cfg["pv"]["p_rated_w"]),
        n_panels=int(cfg["pv"]["n_panels"]),
        system_capacity_kw=float(cfg["pv"]["system_capacity_kw"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def eval_physics_split(model, cfg, split: str, device):
    target_scale, p_rated = target_scale_and_rated(cfg)
    ds = CrossDomainDataset(
        cfg["paths"]["processed_dir"] / f"{split}_normalized.csv",
        cfg["paths"]["ninja_path"],
        cfg["features"],
        target_col=cfg["target"],
        physics_col=cfg["physics_col"],
        target_scale=target_scale,
        physics_rated=float(cfg["pv"]["p_rated_w"]),
    )
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    ys, preds, alphas, residuals, ext_t, ext_p, masks = [], [], [], [], [], [], []
    for x, p_phys, y, ninja_kw, has_ext in loader:
        out = model(x.to(device), p_phys.to(device), return_details=True)
        ys.append(y.numpy())
        preds.append(out.physics.y_hat.cpu().numpy())
        alphas.append(out.physics.alpha.cpu().numpy())
        residuals.append(out.physics.residual.cpu().numpy())
        ext_t.append(ninja_kw.numpy())
        ext_p.append(out.external_kw.cpu().numpy())
        masks.append(has_ext.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    metrics = summarize_predictions(
        y_true, y_pred, p_rated, float(cfg["pv"]["capacity_margin"])
    )
    metrics["mean_alpha"] = float(np.mean(np.concatenate(alphas)))
    metrics["mean_abs_residual"] = float(np.mean(np.abs(np.concatenate(residuals))))
    mask = np.concatenate(masks).reshape(-1) > 0.5
    if mask.any():
        ext = summarize_predictions(
            np.concatenate(ext_t).reshape(-1)[mask],
            np.concatenate(ext_p).reshape(-1)[mask],
            p_rated=float(cfg["pv"]["system_capacity_kw"]),
            capacity_margin=float(cfg["pv"]["capacity_margin"]),
        )
        metrics["ninja_in_split"] = ext
        metrics["ninja_in_split"]["n"] = int(mask.sum())
    return metrics, {
        "y_true": y_true,
        "y_pred": y_pred,
        "alpha": np.concatenate(alphas),
        "residual": np.concatenate(residuals),
    }


@torch.no_grad()
def eval_ninja_full(model, cfg, device, use_external_head: bool = True):
    frame = build_case1_frame(cfg)
    scaler = joblib.load(cfg["paths"]["scaler_path"])
    x = scaler.transform(frame[cfg["features"]].to_numpy(dtype=np.float64)).astype(np.float32)
    p_phys = (
        frame[cfg["physics_col"]].to_numpy(dtype=np.float32) / float(cfg["pv"]["p_rated_w"])
    ).reshape(-1, 1)
    true_kw = frame["electricity"].to_numpy(dtype=np.float64)

    ds = TensorDataset(torch.tensor(x), torch.tensor(p_phys))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    preds = []
    alphas = []
    for xb, pb in loader:
        out = model(xb.to(device), pb.to(device), return_details=True)
        if use_external_head:
            preds.append(out.external_kw.cpu().numpy())
        else:
            # physics path scaled to 10 kW system
            preds.append((out.physics.y_hat.cpu().numpy() * cfg["pv"]["n_panels"] / 1000.0))
        alphas.append(out.physics.alpha.cpu().numpy())
    pred_kw = np.concatenate(preds).reshape(-1)
    metrics = summarize_predictions(
        true_kw,
        pred_kw,
        p_rated=float(cfg["pv"]["system_capacity_kw"]),
        capacity_margin=float(cfg["pv"]["capacity_margin"]),
    )
    metrics["n"] = int(len(true_kw))
    metrics["mean_alpha"] = float(np.mean(np.concatenate(alphas)))
    metrics["head"] = "external" if use_external_head else "physics_scaled"
    return metrics, {"y_true_kw": true_kw, "y_pred_kw": pred_kw, "alpha": np.concatenate(alphas)}


def run_eval(strategy: str, config_path: str | None = None) -> dict:
    cfg = load_train_config(config_path or str(ROOT / "configs" / "train_ar_pcnn_cd.yaml"))
    device = resolve_device(cfg["train"]["device"])
    ckpt_name = (
        cfg["train"]["checkpoint_name_finetune"]
        if strategy == "finetune"
        else cfg["train"]["checkpoint_name_multitask"]
    )
    ckpt_path = cfg["paths"]["checkpoint_dir"] / ckpt_name
    model, _ = load_cd_model(cfg, ckpt_path, device)

    phys_metrics, phys_arrays = eval_physics_split(model, cfg, "test", device)
    ninja_holdout = phys_metrics.get("ninja_in_split")  # test has almost no 2019
    # Holdout = ninja labels inside VAL split
    val_metrics, _ = eval_physics_split(model, cfg, "val", device)
    ninja_full_ext, full_arr_ext = eval_ninja_full(model, cfg, device, use_external_head=True)
    ninja_full_phys, _ = eval_ninja_full(model, cfg, device, use_external_head=False)

    summary = {
        "strategy": strategy,
        "nsrdb_test": phys_metrics,
        "ninja_val_holdout": val_metrics.get("ninja_in_split", {}),
        "ninja_full_2019_external_head": ninja_full_ext,
        "ninja_full_2019_physics_scaled": ninja_full_phys,
    }
    out = cfg["paths"]["metrics_dir"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"ar_pcnn_cd_{strategy}_eval.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out / f"ar_pcnn_cd_{strategy}_test_arrays.npz",
        **phys_arrays,
    )
    np.savez_compressed(
        out / f"ar_pcnn_cd_{strategy}_ninja_full_external.npz",
        **full_arr_ext,
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_ar_pcnn_cd.yaml"))
    parser.add_argument("--strategy", choices=["multitask", "finetune", "both"], default="both")
    args = parser.parse_args()
    if args.strategy == "both":
        run_eval("multitask", args.config)
        run_eval("finetune", args.config)
    else:
        run_eval(args.strategy, args.config)


if __name__ == "__main__":
    main()
