"""MC Dropout + conformal calibration for Phase-6 cross-domain models (physics path)."""

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

from src.models.conformal import ConformalCalibrator
from src.models.dataset_cd import CrossDomainDataset
from src.models.metrics import regression_metrics
from src.training.evaluate_ar_pcnn_cd import load_cd_model
from src.training.train import load_train_config, resolve_device, target_scale_and_rated
from src.training.uncertainty import enable_mc_dropout


@torch.no_grad()
def mc_predict(model, loader, device, n_samples: int):
    enable_mc_dropout(model)
    passes, truths = [], None
    for s in range(n_samples):
        preds, ys = [], []
        for batch in loader:
            x, p_phys, y = batch[0], batch[1], batch[2]
            out = model(x.to(device), p_phys.to(device), return_details=True)
            preds.append(out.physics.y_hat.cpu().numpy())
            if s == 0:
                ys.append(y.numpy())
        passes.append(np.concatenate(preds, axis=0))
        if s == 0:
            truths = np.concatenate(ys, axis=0)
    stack = np.stack(passes, axis=0)
    return truths.reshape(-1), stack.mean(0).reshape(-1), stack.std(0).reshape(-1)


def run_uq(strategy: str, config_path: str | None = None, n_samples: int | None = None) -> dict:
    cfg = load_train_config(config_path or str(ROOT / "configs" / "train_ar_pcnn_cd.yaml"))
    device = resolve_device(cfg["train"]["device"])
    target_scale, p_rated = target_scale_and_rated(cfg)
    n_mc = int(n_samples or cfg.get("uncertainty", {}).get("n_mc_samples", 50))
    target_picp = float(cfg.get("uncertainty", {}).get("target_picp", 0.95))
    z = float(cfg.get("uncertainty", {}).get("z", 1.96))

    ckpt_name = (
        cfg["train"]["checkpoint_name_finetune"]
        if strategy == "finetune"
        else cfg["train"]["checkpoint_name_multitask"]
    )
    model, _ = load_cd_model(cfg, cfg["paths"]["checkpoint_dir"] / ckpt_name, device)

    def loader(split: str):
        ds = CrossDomainDataset(
            cfg["paths"]["processed_dir"] / f"{split}_normalized.csv",
            cfg["paths"]["ninja_path"],
            cfg["features"],
            target_col=cfg["target"],
            physics_col=cfg["physics_col"],
            target_scale=target_scale,
            physics_rated=float(cfg["pv"]["p_rated_w"]),
        )
        return DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    y_val, mean_val, std_val = mc_predict(model, loader("val"), device, n_mc)
    calibrator = ConformalCalibrator().fit_std_scale(
        y_val, mean_val, std_val, target_picp=target_picp, z=z
    )
    y_test, mean_test, std_test = mc_predict(model, loader("test"), device, n_mc)
    raw_picp = float(np.mean((y_test >= mean_test - z * std_test) & (y_test <= mean_test + z * std_test)))
    raw_mpiw = float(np.mean(2 * z * np.maximum(std_test, 1e-8)))
    cal = calibrator.metrics(y_test, mean_test, std_test)
    summary = {
        "strategy": strategy,
        "n_mc_samples": n_mc,
        "regression_on_mc_mean": regression_metrics(y_test, mean_test),
        "uncalibrated": {"picp_95": raw_picp, "mpiw": raw_mpiw},
        "calibrated_on_val_applied_to_test": cal,
        "calibrator": calibrator.to_dict(),
    }
    out = cfg["paths"]["metrics_dir"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"ar_pcnn_cd_{strategy}_uncertainty.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lo, hi = calibrator.intervals(mean_test, std_test)
    n_show = min(24 * 7, len(y_test))
    fig, ax = plt.subplots(figsize=(11, 4))
    idx = np.arange(n_show)
    ax.plot(idx, y_test[:n_show], label="True", lw=1)
    ax.plot(idx, mean_test[:n_show], label="MC mean", lw=1)
    ax.fill_between(idx, lo[:n_show], hi[:n_show], alpha=0.25, label="Calibrated 95% PI")
    ax.set_title(f"CD-{strategy} calibrated UQ | PICP={cal['picp']:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg["paths"]["figures_dir"] / f"phase6_{strategy}_uncertainty.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_ar_pcnn_cd.yaml"))
    parser.add_argument("--strategy", choices=["multitask", "finetune", "both"], default="both")
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()
    if args.strategy == "both":
        run_uq("multitask", args.config, args.n_samples)
        run_uq("finetune", args.config, args.n_samples)
    else:
        run_uq(args.strategy, args.config, args.n_samples)


if __name__ == "__main__":
    main()
