"""
FastAPI inference server for Solar AR-PCNN-CD model.

Exposes:
    GET  /health        — liveness check
    GET  /model-info    — loaded model metadata
    POST /predict       — single-sample prediction
    POST /predict/batch — batch predictions (up to 256 samples)

Load order:
    1. Loads configs/train_ar_pcnn_cd.yaml (same config used during training)
    2. Loads data/processed/scaler.joblib   (train-fitted StandardScaler)
    3. Loads artifacts/checkpoints/ar_pcnn_cd_multitask_best.pt

All inputs are raw (un-normalised) meteorological values — the server
applies the scaler and computes P_theoretical internally.

Usage:
    python api/run_server.py            # recommended
    uvicorn api.predict:app --reload    # for development
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Project root on sys.path so src.* imports resolve regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ar_pcnn_cd import ARPCNNCrossDomain  # noqa: E402

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Raw (un-normalised) meteorological features for one time-step.

    Units match the NSRDB dataset exactly:
      GHI, DNI, DHI          — W/m²
      Temperature             — °C
      Relative Humidity       — %
      Pressure                — mbar
      Surface Albedo          — fraction (0–1)
      Wind Speed              — m/s
      Wind Direction          — degrees (0–360)
      Dew Point               — °C
      Precipitable Water      — cm
      Solar Zenith Angle      — degrees (0–90)
      hour_sin / hour_cos     — pre-computed cyclical hour encodings
                                (pass values OR omit and supply `hour`)
      hour                    — integer 0-23 (used only when hour_sin/hour_cos
                                are omitted)
      P_theoretical           — theoretical PV output in Watts (optional;
                                defaults to GHI-proportional estimate)
    """

    GHI: float = Field(..., ge=0.0, le=2000.0, description="Global Horizontal Irradiance W/m²")
    DNI: float = Field(..., ge=0.0, le=2000.0, description="Direct Normal Irradiance W/m²")
    DHI: float = Field(..., ge=0.0, le=2000.0, description="Diffuse Horizontal Irradiance W/m²")
    Temperature: float = Field(..., ge=-60.0, le=70.0, description="Air temperature °C")
    Relative_Humidity: float = Field(
        ..., ge=0.0, le=100.0, alias="Relative Humidity",
        description="Relative humidity %"
    )
    Pressure: float = Field(..., ge=800.0, le=1100.0, description="Atmospheric pressure mbar")
    Surface_Albedo: float = Field(
        ..., ge=0.0, le=1.0, alias="Surface Albedo",
        description="Surface albedo fraction"
    )
    Wind_Speed: float = Field(
        ..., ge=0.0, le=100.0, alias="Wind Speed",
        description="Wind speed m/s"
    )
    Wind_Direction: float = Field(
        ..., ge=0.0, le=360.0, alias="Wind Direction",
        description="Wind direction degrees"
    )
    Dew_Point: float = Field(
        ..., ge=-80.0, le=60.0, alias="Dew Point",
        description="Dew point temperature °C"
    )
    Precipitable_Water: float = Field(
        ..., ge=0.0, le=20.0, alias="Precipitable Water",
        description="Precipitable water cm"
    )
    Solar_Zenith_Angle: float = Field(
        ..., ge=0.0, le=180.0, alias="Solar Zenith Angle",
        description="Solar zenith angle degrees"
    )
    hour_sin: float | None = Field(
        None, ge=-1.0, le=1.0,
        description="sin(2π·hour/24) — computed from `hour` if omitted"
    )
    hour_cos: float | None = Field(
        None, ge=-1.0, le=1.0,
        description="cos(2π·hour/24) — computed from `hour` if omitted"
    )
    hour: int | None = Field(
        None, ge=0, le=23,
        description="Hour of day 0-23 (only used when hour_sin/hour_cos are omitted)"
    )
    P_theoretical: float | None = Field(
        None, ge=0.0,
        description="Theoretical PV output W (optional; estimated from GHI if omitted)"
    )

    model_config = {"populate_by_name": True}

    @field_validator("hour_sin", "hour_cos", mode="before")
    @classmethod
    def _allow_none(cls, v: Any) -> Any:
        return v


class PredictResponse(BaseModel):
    predicted_power_w: float = Field(..., description="Predicted PV output in Watts")
    predicted_power_kw: float = Field(..., description="Predicted PV output in kilowatts")
    alpha: float = Field(..., description="Physics gate α — how much the model trusts physics (0–1)")
    residual_w: float = Field(..., description="Learned residual correction in Watts")
    physics_w: float = Field(..., description="Raw theoretical physics estimate in Watts")
    external_kw: float = Field(..., description="Cross-domain system power head output kW (10 kW system)")
    latency_ms: float = Field(..., description="Server-side inference latency in milliseconds")


class BatchPredictRequest(BaseModel):
    samples: list[PredictRequest] = Field(..., min_length=1, max_length=256)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count: int
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

class _ModelState:
    model: ARPCNNCrossDomain | None = None
    scaler: Any = None
    cfg: dict = {}
    device: torch.device = torch.device("cpu")
    feature_cols: list[str] = []
    p_rated_w: float = 250.0
    checkpoint_path: str = ""
    loaded_at: str = ""


_state = _ModelState()

# The 14 feature columns in the exact order the scaler was fitted
_FEATURE_ORDER = [
    "GHI",
    "DNI",
    "DHI",
    "Temperature",
    "Relative Humidity",
    "Relative_Humidity",   # alias handled below
    "Pressure",
    "Surface Albedo",
    "Surface_Albedo",
    "Wind Speed",
    "Wind_Speed",
    "Wind Direction",
    "Wind_Direction",
    "Dew Point",
    "Dew_Point",
    "Precipitable Water",
    "Precipitable_Water",
    "Solar Zenith Angle",
    "Solar_Zenith_Angle",
    "hour_sin",
    "hour_cos",
]

# Canonical order expected by scaler and model (must match training)
_CANONICAL_FEATURES = [
    "GHI", "DNI", "DHI", "Temperature",
    "Relative Humidity", "Pressure", "Surface Albedo",
    "Wind Speed", "Wind Direction", "Dew Point",
    "Precipitable Water", "Solar Zenith Angle",
    "hour_sin", "hour_cos",
]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Solar AR-PCNN Inference API",
    description=(
        "REST inference endpoint for the Adaptive Residual Physics-Constrained Neural Network "
        "(AR-PCNN-CD) trained on NSRDB solar irradiance data. "
        "Designed for JMeter load testing."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def _load_model() -> None:
    """Load config, scaler, and model checkpoint once at server start."""
    cfg_path = ROOT / "configs" / "train_ar_pcnn_cd.yaml"
    if not cfg_path.exists():
        raise RuntimeError(f"Config not found: {cfg_path}")

    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against project root
    path_keys = (
        "processed_dir", "checkpoint_dir", "metrics_dir",
        "figures_dir", "scaler_path", "ninja_path", "pretrained_ar_pcnn",
    )
    for key in path_keys:
        if key not in cfg.get("paths", {}):
            continue
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p

    scaler_path = cfg["paths"]["scaler_path"]
    if not scaler_path.exists():
        raise RuntimeError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)

    ckpt_name = cfg["train"]["checkpoint_name_multitask"]
    ckpt_path = cfg["paths"]["checkpoint_dir"] / ckpt_name
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    device_str = cfg["train"].get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

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

    _state.model = model
    _state.scaler = scaler
    _state.cfg = cfg
    _state.device = device
    _state.feature_cols = list(cfg["features"])
    _state.p_rated_w = float(cfg["pv"]["p_rated_w"])
    _state.checkpoint_path = str(ckpt_path)
    _state.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print(f"[API] Model loaded from {ckpt_path} on {device}")
    print(f"[API] Features: {_state.feature_cols}")


# ---------------------------------------------------------------------------
# Helper: request → numpy feature vector
# ---------------------------------------------------------------------------

def _request_to_features(req: PredictRequest) -> tuple[np.ndarray, float]:
    """
    Returns (feature_row, p_theoretical_scaled) where:
      feature_row          — shape (14,) normalised float32
      p_theoretical_scaled — P_theoretical / p_rated_w
    """
    # Resolve cyclical time features
    if req.hour_sin is None or req.hour_cos is None:
        if req.hour is None:
            raise HTTPException(
                status_code=422,
                detail="Provide either hour_sin+hour_cos or hour (integer 0-23).",
            )
        angle = 2 * math.pi * req.hour / 24.0
        hour_sin = math.sin(angle)
        hour_cos = math.cos(angle)
    else:
        hour_sin = req.hour_sin
        hour_cos = req.hour_cos

    # Map Pydantic field aliases back to canonical column names
    raw = {
        "GHI": req.GHI,
        "DNI": req.DNI,
        "DHI": req.DHI,
        "Temperature": req.Temperature,
        "Relative Humidity": req.Relative_Humidity,
        "Pressure": req.Pressure,
        "Surface Albedo": req.Surface_Albedo,
        "Wind Speed": req.Wind_Speed,
        "Wind Direction": req.Wind_Direction,
        "Dew Point": req.Dew_Point,
        "Precipitable Water": req.Precipitable_Water,
        "Solar Zenith Angle": req.Solar_Zenith_Angle,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }

    # Build feature vector in canonical order
    row = np.array([[raw[c] for c in _CANONICAL_FEATURES]], dtype=np.float64)
    row_normalised = _state.scaler.transform(row).astype(np.float32)

    # P_theoretical — use provided value or estimate from GHI
    if req.P_theoretical is not None:
        p_theoretical = float(req.P_theoretical)
    else:
        # Simple GHI-proportional estimate at STC (1000 W/m²)
        p_theoretical = req.GHI / 1000.0 * _state.p_rated_w

    p_theoretical_scaled = float(p_theoretical / _state.p_rated_w)

    return row_normalised, p_theoretical_scaled


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness check")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _state.model is not None,
        "device": str(_state.device),
    }


@app.get("/model-info", summary="Model and checkpoint metadata")
def model_info() -> dict:
    return {
        "model_class": "ARPCNNCrossDomain",
        "checkpoint": _state.checkpoint_path,
        "loaded_at": _state.loaded_at,
        "input_dim": len(_state.feature_cols),
        "features": _state.feature_cols,
        "p_rated_w": _state.p_rated_w,
        "device": str(_state.device),
    }


@app.post("/predict", response_model=PredictResponse, summary="Single-sample prediction")
@torch.no_grad()
def predict(req: PredictRequest) -> PredictResponse:
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t_start = time.perf_counter()

    row_norm, p_scaled = _request_to_features(req)

    x_t = torch.tensor(row_norm, device=_state.device)          # (1, 14)
    p_t = torch.tensor([[p_scaled]], dtype=torch.float32, device=_state.device)  # (1, 1)

    out = _state.model(x_t, p_t, return_details=True)

    y_hat_w   = float(out.physics.y_hat.cpu().item())
    alpha     = float(out.physics.alpha.cpu().item())
    residual  = float(out.physics.residual.cpu().item())
    phys_w    = float(out.physics.physics_watts.cpu().item())
    ext_kw    = float(out.external_kw.cpu().item())

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    return PredictResponse(
        predicted_power_w=y_hat_w,
        predicted_power_kw=y_hat_w / 1000.0,
        alpha=alpha,
        residual_w=residual,
        physics_w=phys_w,
        external_kw=ext_kw,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="Batch predictions")
@torch.no_grad()
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t_start = time.perf_counter()

    rows, p_scaleds = [], []
    for sample in req.samples:
        row_norm, p_scaled = _request_to_features(sample)
        rows.append(row_norm)
        p_scaleds.append([[p_scaled]])

    x_t = torch.tensor(np.vstack(rows), device=_state.device)
    p_t = torch.tensor(np.array(p_scaleds, dtype=np.float32).reshape(-1, 1), device=_state.device)

    out = _state.model(x_t, p_t, return_details=True)

    y_hats   = out.physics.y_hat.cpu().numpy().flatten()
    alphas   = out.physics.alpha.cpu().numpy().flatten()
    residuals = out.physics.residual.cpu().numpy().flatten()
    phys_ws  = out.physics.physics_watts.cpu().numpy().flatten()
    ext_kws  = out.external_kw.cpu().numpy().flatten()

    total_ms = (time.perf_counter() - t_start) * 1000.0
    per_ms   = total_ms / len(req.samples)

    predictions = [
        PredictResponse(
            predicted_power_w=float(y_hats[i]),
            predicted_power_kw=float(y_hats[i]) / 1000.0,
            alpha=float(alphas[i]),
            residual_w=float(residuals[i]),
            physics_w=float(phys_ws[i]),
            external_kw=float(ext_kws[i]),
            latency_ms=round(per_ms, 3),
        )
        for i in range(len(req.samples))
    ]

    return BatchPredictResponse(
        predictions=predictions,
        count=len(predictions),
        total_latency_ms=round(total_ms, 3),
    )
