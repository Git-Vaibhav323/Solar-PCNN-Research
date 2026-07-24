"""Monte Carlo Dropout uncertainty quantification (paper Section 4.5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.dataset import SolarPCNNDataset
from src.models.metrics import regression_metrics
from src.models.pcnn import PCNN
from src.training.train import load_train_config, resolve_device, target_scale_and_rated


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep BatchNorm in eval mode; enable Dropout stochasticity only."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _load_model(ckpt_path: Path, cfg: dict, device: torch.device) -> PCNN:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model = PCNN(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        fuse_dim=int(cfg["model"]["fuse_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


@torch.no_grad()
def mc_dropout_predict(
    model: PCNN,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns y_true, predictive mean, predictive std (epistemic via MC Dropout).
    """
    enable_mc_dropout(model)
    all_true = []
    # Collect per-pass predictions then stack
    passes = []
    for s in range(n_samples):
        preds = []
        truths = []
        for x, p_phys, y in loader:
            y_hat = model(x.to(device), p_phys.to(device))
            preds.append(y_hat.cpu().numpy())
            if s == 0:
                truths.append(y.numpy())
        passes.append(np.concatenate(preds, axis=0))
        if s == 0:
            all_true = np.concatenate(truths, axis=0)

    stack = np.stack(passes, axis=0)  # [T, N, 1]
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    return all_true.reshape(-1), mean.reshape(-1), std.reshape(-1)


def prediction_interval_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    z: float = 1.96,
) -> dict:
    """PICP / MPIW style metrics for mean ± z*std intervals."""
    lower = mean - z * std
    upper = mean + z * std
    covered = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(upper - lower))
    # PINW: width normalised by target range
    rng = float(np.max(y_true) - np.min(y_true))
    pinw = float(mpiw / rng) if rng > 0 else float("nan")
    return {
        "picp_95": picp,
        "mpiw": mpiw,
        "pinw": pinw,
        "mean_predictive_std": float(np.mean(std)),
        "paper_picp_reference": 0.9472,
    }


def run_uncertainty(
    checkpoint: str | None = None,
    config_path: str | None = None,
    split: str = "test",
    n_samples: int = 50,
) -> dict:
    cfg = load_train_config(config_path)
    device = resolve_device(cfg["train"]["device"])
    ckpt_path = Path(checkpoint) if checkpoint else cfg["paths"]["checkpoint_dir"] / "pcnn_best.pt"
    model = _load_model(ckpt_path, cfg, device)

    target_scale, p_rated = target_scale_and_rated(cfg)
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

    y_true, mean, std = mc_dropout_predict(model, loader, device, n_samples=n_samples)
    reg = regression_metrics(y_true, mean)
    uq = prediction_interval_metrics(y_true, mean, std)
    summary = {
        "split": split,
        "n_mc_samples": n_samples,
        "target_unit": cfg["train"].get("target_unit", "W"),
        "p_rated": p_rated,
        "regression_on_mc_mean": reg,
        "uncertainty": uq,
    }

    out_dir = cfg["paths"]["metrics_dir"]
    fig_dir = cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / f"uncertainty_{split}.npz",
        y_true=y_true,
        y_mean=mean,
        y_std=std,
    )
    with (out_dir / f"uncertainty_{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot a short window with ±1.96σ band
    n_show = min(24 * 7, len(y_true))
    idx = np.arange(n_show)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx, y_true[:n_show], label="True", lw=1)
    ax.plot(idx, mean[:n_show], label="MC mean", lw=1)
    ax.fill_between(
        idx,
        mean[:n_show] - 1.96 * std[:n_show],
        mean[:n_show] + 1.96 * std[:n_show],
        alpha=0.25,
        label="95% PI",
    )
    ax.set_title(f"MC Dropout uncertainty ({split}) | PICP={uq['picp_95']:.3f}")
    ax.set_xlabel("Hour index")
    ax.set_ylabel(cfg["train"].get("target_unit", "W"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = fig_dir / f"uncertainty_{split}_mc_dropout.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Figure -> {fig_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty on a data split")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()
    run_uncertainty(args.checkpoint, args.config, args.split, args.n_samples)


if __name__ == "__main__":
    main()
