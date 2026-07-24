"""
MC Dropout uncertainty for AR-PCNN + post-hoc conformal std scaling.

Fits calibration scale on the validation split; reports calibrated PICP/MPIW on test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ar_pcnn import ARPCNN
from src.models.conformal import ConformalCalibrator
from src.models.dataset import SolarPCNNDataset
from src.models.metrics import regression_metrics
from src.training.train import load_train_config, resolve_device, target_scale_and_rated
from src.training.uncertainty import enable_mc_dropout


def _load_model(cfg, ckpt_path, device, p_rated):
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
    return model


@torch.no_grad()
def mc_predict(model, loader, device, n_samples: int):
    enable_mc_dropout(model)
    passes, truths = [], None
    for s in range(n_samples):
        preds, ys = [], []
        for x, p_phys, y in loader:
            y_hat = model(x.to(device), p_phys.to(device))
            preds.append(y_hat.cpu().numpy())
            if s == 0:
                ys.append(y.numpy())
        passes.append(np.concatenate(preds, axis=0))
        if s == 0:
            truths = np.concatenate(ys, axis=0)
    stack = np.stack(passes, axis=0)
    return truths.reshape(-1), stack.mean(axis=0).reshape(-1), stack.std(axis=0).reshape(-1)


def _loader(cfg, split, target_scale, p_rated_w):
    name = {"train": "train_normalized.csv", "val": "val_normalized.csv", "test": "test_normalized.csv"}[split]
    ds = SolarPCNNDataset(
        cfg["paths"]["processed_dir"] / name,
        cfg["features"],
        target_col=cfg["target"],
        physics_col=cfg["physics_col"],
        target_scale=target_scale,
        physics_rated=p_rated_w,
    )
    return DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)


def run_uncertainty(
    checkpoint: str | None = None,
    config_path: str | None = None,
    n_samples: int | None = None,
) -> dict:
    cfg = load_train_config(config_path or str(ROOT / "configs" / "train_ar_pcnn.yaml"))
    device = resolve_device(cfg["train"]["device"])
    target_scale, p_rated = target_scale_and_rated(cfg)
    n_mc = int(n_samples or cfg.get("uncertainty", {}).get("n_mc_samples", 50))
    target_picp = float(cfg.get("uncertainty", {}).get("target_picp", 0.95))
    z = float(cfg.get("uncertainty", {}).get("z", 1.96))

    ckpt_path = Path(checkpoint) if checkpoint else cfg["paths"]["checkpoint_dir"] / "ar_pcnn_best.pt"
    model = _load_model(cfg, ckpt_path, device, p_rated)

    val_loader = _loader(cfg, "val", target_scale, float(cfg["pv"]["p_rated_w"]))
    test_loader = _loader(cfg, "test", target_scale, float(cfg["pv"]["p_rated_w"]))

    y_val, mean_val, std_val = mc_predict(model, val_loader, device, n_mc)
    calibrator = ConformalCalibrator().fit_std_scale(
        y_val, mean_val, std_val, target_picp=target_picp, z=z
    )
    val_cal = calibrator.metrics(y_val, mean_val, std_val)

    y_test, mean_test, std_test = mc_predict(model, test_loader, device, n_mc)
    raw_picp = float(
        np.mean(
            (y_test >= mean_test - z * std_test) & (y_test <= mean_test + z * std_test)
        )
    )
    raw_mpiw = float(np.mean(2 * z * std_test))
    cal = calibrator.metrics(y_test, mean_test, std_test)
    reg = regression_metrics(y_test, mean_test)

    summary = {
        "model": "ar_pcnn",
        "n_mc_samples": n_mc,
        "regression_on_mc_mean": reg,
        "uncalibrated": {"picp_95": raw_picp, "mpiw": raw_mpiw},
        "calibrated_on_val_applied_to_test": cal,
        "val_calibration_fit": val_cal,
        "calibrator": calibrator.to_dict(),
    }

    out_dir = cfg["paths"]["metrics_dir"]
    fig_dir = cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    lo, hi = calibrator.intervals(mean_test, std_test)
    np.savez_compressed(
        out_dir / "ar_pcnn_uncertainty_test.npz",
        y_true=y_test,
        y_mean=mean_test,
        y_std=std_test,
        y_lo=lo,
        y_hi=hi,
    )
    with (out_dir / "ar_pcnn_uncertainty_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    n_show = min(24 * 7, len(y_test))
    idx = np.arange(n_show)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx, y_test[:n_show], label="True", lw=1)
    ax.plot(idx, mean_test[:n_show], label="MC mean", lw=1)
    ax.fill_between(idx, lo[:n_show], hi[:n_show], alpha=0.25, label="Calibrated 95% PI")
    ax.set_title(
        f"AR-PCNN calibrated UQ | PICP={cal['picp']:.3f} MPIW={cal['mpiw']:.2f} (scale={calibrator.scale:.3f})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "ar_pcnn_uncertainty_calibrated.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_ar_pcnn.yaml"))
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()
    run_uncertainty(args.checkpoint, args.config, args.n_samples)


if __name__ == "__main__":
    main()
