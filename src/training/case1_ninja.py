"""Case 1 — external validation on Renewables.ninja (2019), paper Section 5.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.metrics import summarize_predictions
from src.models.pcnn import PCNN
from src.training.train import load_train_config, resolve_device


def _load_checkpoint(path: Path, cfg: dict, device: torch.device) -> PCNN:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model = PCNN(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        fuse_dim=int(cfg["model"]["fuse_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def build_case1_frame(cfg: dict) -> pd.DataFrame:
    """
    Align NSRDB meteorological inputs with ninja PV output on UTC+5 hourly grid.

    Model inputs: NSRDB features (scaled with train-only scaler).
    External target: ninja `electricity` for a 10 kW system (40 × 250 W panels).
    """
    processed = cfg["paths"]["processed_dir"]
    nsrdb = pd.read_csv(processed / "nsrdb_model_ready_raw.csv", parse_dates=["timestamp"])
    ninja = pd.read_csv(processed / "ninja_2019_prepared.csv")

    nsrdb["timestamp"] = pd.to_datetime(nsrdb["timestamp"], utc=True).dt.tz_convert("Etc/GMT-5")
    ninja_ts = pd.to_datetime(ninja["time"], utc=True).dt.tz_convert("Etc/GMT-5").dt.floor("h")
    ninja = ninja.copy()
    ninja["align_ts"] = ninja_ts
    nsrdb = nsrdb.copy()
    nsrdb["align_ts"] = nsrdb["timestamp"].dt.floor("h")

    # Overlap year 2019 only (ninja coverage)
    nsrdb = nsrdb[nsrdb["align_ts"].dt.year == 2019]
    merged = nsrdb.merge(
        ninja[["align_ts", "electricity", "irradiance_direct", "irradiance_diffuse", "temperature"]],
        on="align_ts",
        how="inner",
        suffixes=("", "_ninja"),
    )
    if merged.empty:
        raise RuntimeError("No overlapping timestamps between NSRDB and ninja after alignment.")
    return merged.sort_values("align_ts").reset_index(drop=True)


@torch.no_grad()
def predict_panel_watts(
    model: PCNN,
    x: np.ndarray,
    p_phys: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(p_phys, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs = []
    for xb, pb in loader:
        outs.append(model(xb.to(device), pb.to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def run_case1(
    checkpoint: str | None = None,
    config_path: str | None = None,
    n_panels: int = 40,
) -> dict:
    cfg = load_train_config(config_path)
    device = resolve_device(cfg["train"]["device"])
    ckpt_path = Path(checkpoint) if checkpoint else cfg["paths"]["checkpoint_dir"] / "pcnn_best.pt"
    model = _load_checkpoint(ckpt_path, cfg, device)

    frame = build_case1_frame(cfg)
    feature_cols = cfg["features"]
    scaler = joblib.load(cfg["paths"]["scaler_path"])
    x = scaler.transform(frame[feature_cols].to_numpy(dtype=np.float64)).astype(np.float32)
    p_rated = float(cfg["pv"]["p_rated_w"])
    p_phys = (frame[cfg["physics_col"]].to_numpy(dtype=np.float32) / p_rated).reshape(-1, 1)

    pred_w = predict_panel_watts(model, x, p_phys, device).reshape(-1)
    # Scale single-panel watts → system kW (paper Case 1: 10 kW = 40 × 250 W)
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
            "align_start": str(frame["align_ts"].iloc[0]),
            "align_end": str(frame["align_ts"].iloc[-1]),
            "paper_metric_alignment": "not_reconcilable",
            "note": (
                "Published Case-1 absolute-error figures are not used as a reference: "
                "our Case-1 is a 10 kW system in kW; the source paper does not specify "
                "system capacity/unit clearly enough for safe conversion — not reconcilable. "
                "Do not guess conversions such as 0.0213 kW ↔ 21.3 W."
            ),
        }
    )

    out_dir = cfg["paths"]["metrics_dir"]
    fig_dir = cfg["paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "case1_ninja_predictions.npz",
        y_true_kw=true_kw,
        y_pred_kw=pred_kw,
        timestamps=frame["align_ts"].astype(str).to_numpy(),
    )
    with (out_dir / "case1_ninja_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Scatter + short time window plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(true_kw, pred_kw, s=3, alpha=0.2)
    lim = max(true_kw.max(), pred_kw.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", lw=1)
    axes[0].set_xlabel("Ninja electricity (kW)")
    axes[0].set_ylabel("PCNN prediction (kW)")
    axes[0].set_title(f"Case 1 scatter (R²={metrics['r2']:.4f})")

    n_show = min(24 * 14, len(true_kw))  # 2 weeks
    axes[1].plot(true_kw[:n_show], label="Ninja", lw=1)
    axes[1].plot(pred_kw[:n_show], label="PCNN", lw=1, alpha=0.8)
    axes[1].set_title("First 2 weeks (kW)")
    axes[1].set_xlabel("Hour index")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig_path = fig_dir / "case1_ninja_validation.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Figure -> {fig_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Case 1: Renewables.ninja external validation")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-panels", type=int, default=40, help="Panels in 10 kW system (paper=40)")
    args = parser.parse_args()
    run_case1(args.checkpoint, args.config, args.n_panels)


if __name__ == "__main__":
    main()
