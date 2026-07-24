"""Shared utilities for Phase 9 experiments."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ar_pcnn import ARPCNN
from src.models.ar_pcnn_cd import ARPCNNCrossDomain
from src.models.conformal import ConformalCalibrator
from src.models.dataset import SolarPCNNDataset
from src.models.dataset_cd import CrossDomainDataset
from src.models.metrics import regression_metrics, summarize_predictions
from src.models.pcnn import PCNN
from src.training.case1_ninja import build_case1_frame
from src.training.evaluate_ar_pcnn_cd import eval_ninja_full, eval_physics_split, load_cd_model
from src.training.train import load_train_config, resolve_device, set_seed, target_scale_and_rated
from src.training.uncertainty import enable_mc_dropout

# Phase 9 output roots (never overwrite Phase 3–8 artifacts)
PHASE9_ROOT = ROOT / "artifacts" / "phase9"
METRICS_DIR = PHASE9_ROOT / "metrics"
CKPT_DIR = PHASE9_ROOT / "checkpoints"
CSV_DIR = PHASE9_ROOT / "csv"
FIG_DIR = ROOT / "data" / "figures" / "phase9"
CONFIG_DIR = PHASE9_ROOT / "configs"
REPORT_DIR = PHASE9_ROOT / "reports"

BASE_PCNN = ROOT / "artifacts" / "checkpoints" / "pcnn_best.pt"
BASE_AR = ROOT / "artifacts" / "checkpoints" / "ar_pcnn_best.pt"
BASE_CD = ROOT / "artifacts" / "checkpoints" / "ar_pcnn_cd_multitask_best.pt"

for d in (METRICS_DIR, CKPT_DIR, CSV_DIR, FIG_DIR, CONFIG_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
    }
)


def ensure_dirs() -> None:
    for d in (METRICS_DIR, CKPT_DIR, CSV_DIR, FIG_DIR, CONFIG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_base_cfg() -> dict:
    return load_train_config(str(ROOT / "configs" / "train_ar_pcnn_cd.yaml"))


def phase9_cfg(
    *,
    lambda_external: float | None = None,
    seed: int | None = None,
    tag: str = "",
) -> dict:
    """Clone base config with Phase 9 output paths and optional overrides."""
    cfg = copy.deepcopy(load_base_cfg())
    cfg["paths"]["checkpoint_dir"] = CKPT_DIR
    cfg["paths"]["metrics_dir"] = METRICS_DIR
    cfg["paths"]["figures_dir"] = FIG_DIR
    if lambda_external is not None:
        cfg["loss"]["lambda_external"] = float(lambda_external)
        tag = tag or f"le{lambda_external:g}".replace(".", "p")
    if seed is not None:
        cfg["train"]["seed"] = int(seed)
        tag = tag or f"seed{seed}"
    cfg["train"]["checkpoint_name_multitask"] = f"multitask_{tag}_best.pt"
    cfg["_phase9_tag"] = tag
    return cfg


def save_cfg_yaml(cfg: dict, name: str) -> Path:
    out = CONFIG_DIR / name
    serial = copy.deepcopy(cfg)
    serial["paths"] = {k: str(v) for k, v in cfg["paths"].items()}
    serial.pop("_phase9_tag", None)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serial, f, sort_keys=False)
    return out


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = CSV_DIR / name
    df.to_csv(path, index=False)
    return path


def save_json(obj: dict, name: str) -> Path:
    path = METRICS_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def save_fig(fig, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_pcnn(device: torch.device | None = None) -> PCNN:
    device = device or resolve_device("auto")
    cfg = load_train_config(str(ROOT / "configs" / "train_pcnn.yaml"))
    ckpt = _torch_load(BASE_PCNN, device)
    model = PCNN(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        fuse_dim=int(cfg["model"]["fuse_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def load_ar_pcnn(device: torch.device | None = None) -> tuple[ARPCNN, dict]:
    device = device or resolve_device("auto")
    cfg = load_train_config(str(ROOT / "configs" / "train_ar_pcnn.yaml"))
    _, p_rated = target_scale_and_rated(cfg)
    ckpt = _torch_load(BASE_AR, device)
    model = ARPCNN(
        input_dim=14,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        p_rated=p_rated,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def load_cd_multitask(
    device: torch.device | None = None,
    ckpt_path: Path | None = None,
) -> tuple[ARPCNNCrossDomain, dict, dict]:
    device = device or resolve_device("auto")
    cfg = load_base_cfg()
    path = ckpt_path or BASE_CD
    model, ckpt = load_cd_model(cfg, path, device)
    return model, cfg, ckpt


def get_test_context(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_base_cfg()
    raw = pd.read_csv(cfg["paths"]["processed_dir"] / "test_raw.csv", parse_dates=["timestamp"])
    raw["hour"] = raw["timestamp"].dt.hour
    raw["month"] = raw["timestamp"].dt.month
    raw["season"] = raw["month"].map(
        lambda m: "summer" if m in (4, 5, 6) else ("winter" if m in (10, 11, 12, 1, 2) else "other")
    )
    raw["daytime"] = raw["GHI"] > 50
    raw["cloudy"] = (raw["GHI"] > 50) & (raw["DHI"] / (raw["GHI"] + 1e-6) > 0.6)
    raw["clear"] = (raw["GHI"] > 200) & (raw["DHI"] / (raw["GHI"] + 1e-6) < 0.35)
    return raw


def make_solar_loader(cfg: dict, split: str = "test") -> DataLoader:
    target_scale, p_rated = target_scale_and_rated(cfg)
    ds = SolarPCNNDataset(
        cfg["paths"]["processed_dir"] / f"{split}_normalized.csv",
        cfg["features"],
        target_scale=target_scale,
        physics_rated=float(cfg["pv"]["p_rated_w"]),
    )
    return DataLoader(ds, batch_size=256, shuffle=False)


def make_cd_loader(cfg: dict, split: str = "test") -> DataLoader:
    target_scale, _ = target_scale_and_rated(cfg)
    ds = CrossDomainDataset(
        cfg["paths"]["processed_dir"] / f"{split}_normalized.csv",
        cfg["paths"]["ninja_path"],
        cfg["features"],
        target_col=cfg["target"],
        physics_col=cfg["physics_col"],
        target_scale=target_scale,
        physics_rated=float(cfg["pv"]["p_rated_w"]),
    )
    return DataLoader(ds, batch_size=256, shuffle=False)


@torch.no_grad()
def predict_ar_on_test(model: ARPCNN, cfg: dict, device: torch.device) -> dict:
    loader = make_solar_loader(cfg, "test")
    ys, preds, alphas, resids = [], [], [], []
    for x, p, y in loader:
        out = model(x.to(device), p.to(device), return_details=True)
        ys.append(y.numpy())
        preds.append(out.y_hat.cpu().numpy())
        alphas.append(out.alpha.cpu().numpy())
        resids.append(out.residual.cpu().numpy())
    return {
        "y_true": np.concatenate(ys).reshape(-1),
        "y_pred": np.concatenate(preds).reshape(-1),
        "alpha": np.concatenate(alphas).reshape(-1),
        "residual": np.concatenate(resids).reshape(-1),
    }


@torch.no_grad()
def predict_cd_physics_on_test(model, cfg: dict, device: torch.device) -> dict:
    _, arrays = eval_physics_split(model, cfg, "test", device)
    return arrays


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def mc_uq_physics(model, cfg: dict, device: torch.device, n_mc: int = 30) -> dict:
    """MC Dropout + conformal PICP/MPIW on physics path (val fit, test eval)."""

    def _mc(loader):
        enable_mc_dropout(model)
        passes, truths = [], None
        with torch.no_grad():
            for s in range(n_mc):
                preds, ys = [], []
                for batch in loader:
                    if len(batch) == 5:
                        x, p, y = batch[0], batch[1], batch[2]
                    else:
                        x, p, y = batch
                    if isinstance(model, ARPCNNCrossDomain):
                        out = model(x.to(device), p.to(device), return_details=True)
                        y_hat = out.physics.y_hat
                    elif isinstance(model, ARPCNN):
                        out = model(x.to(device), p.to(device), return_details=True)
                        y_hat = out.y_hat
                    else:
                        y_hat = model(x.to(device), p.to(device))
                    preds.append(y_hat.cpu().numpy())
                    if s == 0:
                        ys.append(y.numpy())
                passes.append(np.concatenate(preds).reshape(-1))
                if s == 0:
                    truths = np.concatenate(ys).reshape(-1)
        stack = np.stack(passes, 0)
        return truths, stack.mean(0), stack.std(0)

    yv, mv, sv = _mc(make_cd_loader(cfg, "val") if isinstance(model, ARPCNNCrossDomain) else make_solar_loader(cfg, "val"))
    cal = ConformalCalibrator().fit_std_scale(yv, mv, sv, target_picp=0.95)
    yt, mt, st = _mc(make_cd_loader(cfg, "test") if isinstance(model, ARPCNNCrossDomain) else make_solar_loader(cfg, "test"))
    m = cal.metrics(yt, mt, st)
    return {"picp": m["picp"], "mpiw": m["mpiw"], "scale": m["scale"]}


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, p_rated: float = 250.0) -> dict:
    m = summarize_predictions(y_true, y_pred, p_rated, 1.1)
    return {"experiment": name, **{k: m[k] for k in ("mae", "rmse", "r2", "mape")}}


def plain_summary(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
