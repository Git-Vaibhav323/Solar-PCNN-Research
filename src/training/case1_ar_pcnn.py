"""Case 1 external validation for AR-PCNN (same protocol as baseline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ar_pcnn import ARPCNN
from src.models.metrics import summarize_predictions
from src.training.case1_ninja import build_case1_frame
from src.training.train import load_train_config, resolve_device, target_scale_and_rated


@torch.no_grad()
def predict_watts(model, x, p_phys, device, batch_size=512):
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(p_phys, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs = []
    for xb, pb in loader:
        outs.append(model(xb.to(device), pb.to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0).reshape(-1)


def run_case1(checkpoint: str | None = None, config_path: str | None = None, n_panels: int = 40) -> dict:
    cfg = load_train_config(config_path or str(ROOT / "configs" / "train_ar_pcnn.yaml"))
    device = resolve_device(cfg["train"]["device"])
    _, p_rated = target_scale_and_rated(cfg)
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
    model.eval()

    frame = build_case1_frame(cfg)
    scaler = joblib.load(cfg["paths"]["scaler_path"])
    x = scaler.transform(frame[cfg["features"]].to_numpy(dtype=np.float64)).astype(np.float32)
    p_phys = (frame[cfg["physics_col"]].to_numpy(dtype=np.float32) / float(cfg["pv"]["p_rated_w"])).reshape(-1, 1)

    pred_w = predict_watts(model, x, p_phys, device)
    pred_kw = pred_w * n_panels / 1000.0
    true_kw = frame["electricity"].to_numpy(dtype=np.float64)
    metrics = summarize_predictions(
        true_kw,
        pred_kw,
        p_rated=p_rated * n_panels / 1000.0,
        capacity_margin=float(cfg["pv"]["capacity_margin"]),
    )
    metrics.update(
        {
            "n_samples": int(len(frame)),
            "n_panels": n_panels,
            "system_capacity_kw": p_rated * n_panels / 1000.0,
            "model": "ar_pcnn",
        }
    )

    out_dir = cfg["paths"]["metrics_dir"]
    fig_dir = cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "ar_pcnn_case1_predictions.npz",
        y_true_kw=true_kw,
        y_pred_kw=pred_kw,
    )
    with (out_dir / "ar_pcnn_case1_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(true_kw, pred_kw, s=3, alpha=0.2)
    lim = max(true_kw.max(), pred_kw.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("Ninja (kW)")
    ax.set_ylabel("AR-PCNN (kW)")
    ax.set_title(f"AR-PCNN Case 1 (R²={metrics['r2']:.4f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "ar_pcnn_case1_validation.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_ar_pcnn.yaml"))
    parser.add_argument("--n-panels", type=int, default=40)
    args = parser.parse_args()
    run_case1(args.checkpoint, args.config, args.n_panels)


if __name__ == "__main__":
    main()
