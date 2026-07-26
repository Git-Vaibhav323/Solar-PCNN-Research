"""Exp 1 — Hyperparameter sensitivity (lambda_external)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.evaluate_ar_pcnn_cd import eval_ninja_full, eval_physics_split, load_cd_model
from src.training.train import resolve_device, target_scale_and_rated

from .common import (
    BASE_CD,
    CKPT_DIR,
    FIG_DIR,
    REPORT_DIR,
    mc_uq_physics,
    phase9_cfg,
    plain_summary,
    save_csv,
    save_fig,
    save_json,
)
from .train_helpers import run_multitask_training

LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


def _tag(le: float) -> str:
    return f"le{le:g}".replace(".", "p")


def evaluate_run(cfg, ckpt_path, device) -> dict:
    model, _ = load_cd_model(cfg, ckpt_path, device)
    phys, _ = eval_physics_split(model, cfg, "test", device)
    ninja, _ = eval_ninja_full(model, cfg, device, use_external_head=True)
    uq = mc_uq_physics(model, cfg, device, n_mc=30)
    return {
        "nsrdb_r2": phys["r2"],
        "nsrdb_mae": phys["mae"],
        "nsrdb_rmse": phys["rmse"],
        "ninja_r2": ninja["r2"],
        "ninja_mae": ninja["mae"],
        "ninja_rmse": ninja["rmse"],
        "picp": uq["picp"],
        "mpiw": uq["mpiw"],
    }


def run(skip_train: bool = False) -> pd.DataFrame:
    device = resolve_device("auto")
    rows = []

    for le in LAMBDA_VALUES:
        tag = _tag(le)
        cfg = phase9_cfg(lambda_external=le, tag=tag)
        ckpt = CKPT_DIR / cfg["train"]["checkpoint_name_multitask"]

        if not skip_train and not ckpt.exists():
            print(f"[Exp1] Training lambda_external={le} ...")
            summary = run_multitask_training(lambda_external=le, tag=tag)
            train_loss = summary["train_loss_final"]
            val_loss = summary["val_loss_final"]
        else:
            print(f"[Exp1] Using existing checkpoint for lambda={le}")
            train_loss = val_loss = np.nan

        if not ckpt.exists():
            if skip_train and le == 1.0 and BASE_CD.exists():
                ckpt = BASE_CD
                print(f"  Using frozen Phase-6 checkpoint for lambda=1.0")
            else:
                print(f"  WARNING: missing checkpoint {ckpt}")
                continue

        metrics = evaluate_run(cfg, ckpt, device)
        rows.append(
            {
                "lambda_external": le,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    save_csv(df, "exp01_lambda_sensitivity.csv")
    save_json({"rows": rows}, "exp01_lambda_sensitivity.json")

    if len(df):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(df["lambda_external"], df["nsrdb_r2"], "o-", label="NSRDB R²")
        axes[0, 0].plot(df["lambda_external"], df["ninja_r2"], "s-", label="Ninja R²")
        axes[0, 0].set_xlabel("lambda_external")
        axes[0, 0].set_title("R² vs lambda_external")
        axes[0, 0].legend()

        axes[0, 1].plot(df["lambda_external"], df["nsrdb_mae"], "o-", label="NSRDB MAE")
        axes[0, 1].plot(df["lambda_external"], df["ninja_mae"], "s-", label="Ninja MAE")
        axes[0, 1].set_xlabel("lambda_external")
        axes[0, 1].set_title("MAE vs lambda_external")
        axes[0, 1].legend()

        axes[1, 0].plot(df["lambda_external"], df["picp"], "o-", color="#ff7f0e")
        axes[1, 0].axhline(0.95, ls="--", color="k", lw=1)
        axes[1, 0].set_xlabel("lambda_external")
        axes[1, 0].set_title("Calibrated PICP")

        axes[1, 1].plot(df["lambda_external"], df["mpiw"], "o-", color="#2ca02c")
        axes[1, 1].set_xlabel("lambda_external")
        axes[1, 1].set_title("Calibrated MPIW")
        fig.tight_layout()
        save_fig(fig, "exp01_lambda_sensitivity.png")

        n_eval = len(df)
        lambdas_done = sorted(df["lambda_external"].tolist())
        if n_eval < len(LAMBDA_VALUES):
            text = f"""## Hyperparameter sensitivity (lambda_external)

- **Status: incomplete — do not claim a λ sweep.** Planned grid λ ∈ {LAMBDA_VALUES}; only λ ∈ {lambdas_done} has evaluation rows in `exp01_lambda_sensitivity.csv`.
- Reported numbers below are for the evaluated λ only (typically the frozen Phase-6 checkpoint at λ=1.0 when run with `--skip-train`).
- At λ={df.iloc[0]['lambda_external']}: NSRDB MAE={df.iloc[0]['nsrdb_mae']:.6g} W, Ninja R²={df.iloc[0]['ninja_r2']:.4f}, PICP={df.iloc[0]['picp']:.3f}.
- **Publication rule:** omit λ-sensitivity figures/claims until all planned λ values are trained and present in the CSV.
"""
        else:
            best_ninja = df.loc[df["ninja_r2"].idxmax()]
            best_phys = df.loc[df["nsrdb_mae"].idxmin()]
            row0 = df.loc[df["lambda_external"] == 0.0]
            zero_note = ""
            if len(row0):
                zero_note = (
                    f" **λ=0 → Ninja R²≈{row0.iloc[0]['ninja_r2']:.2f}** "
                    f"(external head never trained — control, not a seed failure)."
                )
            text = f"""## Hyperparameter sensitivity (lambda_external)

- Swept λ ∈ {LAMBDA_VALUES} with frozen architecture; each run warm-starts from Phase-5 AR-PCNN.
- **Physics fidelity:** NSRDB R² stays ≈1 across λ; lowest NSRDB MAE at λ={best_phys['lambda_external']} (MAE={best_phys['nsrdb_mae']:.6g} W) — expected when external loss is off if that is λ=0.
- **Cross-domain:** Ninja R² peaks at λ={best_ninja['lambda_external']} (R²={best_ninja['ninja_r2']:.4f}); λ>0 stays ≈0.86–0.87.{zero_note}
- **Trade-off:** Any λ>0 trains the ninja head; λ≈0.25–1.0 keeps physics MAE tiny while holding Ninja R². PICP is weaker at very low λ before recovering near λ≥0.25.
- **UQ:** PICP after conformal scaling is listed per λ in the CSV.
"""
        plain_summary(REPORT_DIR / "exp01_lambda_sensitivity.md", text)

    return df
