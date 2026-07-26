"""Exp 4 — Multi-seed stability."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.evaluate_ar_pcnn_cd import eval_ninja_full, eval_physics_split, load_cd_model
from src.training.train import resolve_device

from .common import (
    BASE_CD,
    CKPT_DIR,
    REPORT_DIR,
    mc_uq_physics,
    phase9_cfg,
    plain_summary,
    save_csv,
    save_fig,
    save_json,
)
from .train_helpers import run_multitask_training

SEEDS = [42, 123, 456, 999, 2026]


def run(skip_train: bool = False, seeds: list[int] | None = None) -> pd.DataFrame:
    device = resolve_device("auto")
    rows = []
    seeds = list(seeds) if seeds is not None else list(SEEDS)

    for seed in seeds:
        tag = f"seed{seed}"
        cfg = phase9_cfg(seed=seed, tag=tag)
        ckpt = CKPT_DIR / cfg["train"]["checkpoint_name_multitask"]

        if not skip_train and not ckpt.exists():
            print(f"[Exp4] Training seed={seed} ...", flush=True)
            run_multitask_training(seed=seed, tag=tag)

        if not ckpt.exists():
            if skip_train and seed == 42 and BASE_CD.exists():
                ckpt = BASE_CD
                print(f"  Using frozen Phase-6 checkpoint for seed=42")
            else:
                print(f"  WARNING: missing checkpoint for seed={seed}")
                continue

        print(f"[Exp4] Evaluating seed={seed} ...", flush=True)
        model, _ = load_cd_model(cfg, ckpt, device)
        phys, _ = eval_physics_split(model, cfg, "test", device)
        ninja, _ = eval_ninja_full(model, cfg, device, use_external_head=True)
        uq = mc_uq_physics(model, cfg, device, n_mc=30)
        row = {
            "seed": seed,
            "nsrdb_r2": phys["r2"],
            "nsrdb_mae": phys["mae"],
            "nsrdb_rmse": phys["rmse"],
            "ninja_r2": ninja["r2"],
            "ninja_mae": ninja["mae"],
            "picp": uq["picp"],
        }
        print(
            f"  seed={seed} ninja_R2={row['ninja_r2']:.4f} nsrdb_MAE={row['nsrdb_mae']:.6g} picp={row['picp']:.3f}",
            flush=True,
        )
        rows.append(row)

    # If only a subset was requested, merge with existing full runs CSV
    df_new = pd.DataFrame(rows)
    runs_path = CKPT_DIR.parent / "csv" / "exp04_multi_seed_runs.csv"
    if runs_path.exists() and seeds != SEEDS:
        df_old = pd.read_csv(runs_path)
        for _, r in df_new.iterrows():
            df_old = df_old[df_old["seed"] != int(r["seed"])]
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.sort_values("seed").reset_index(drop=True)
    else:
        df = df_new.sort_values("seed").reset_index(drop=True) if len(df_new) else df_new

    save_csv(df, "exp04_multi_seed_runs.csv")

    if len(df) >= 2:
        summary = {
            "n_seeds": len(df),
            "nsrdb_r2_mean_std": [float(df["nsrdb_r2"].mean()), float(df["nsrdb_r2"].std())],
            "nsrdb_mae_mean_std": [float(df["nsrdb_mae"].mean()), float(df["nsrdb_mae"].std())],
            "nsrdb_rmse_mean_std": [float(df["nsrdb_rmse"].mean()), float(df["nsrdb_rmse"].std())],
            "ninja_r2_mean_std": [float(df["ninja_r2"].mean()), float(df["ninja_r2"].std())],
            "ninja_mae_mean_std": [float(df["ninja_mae"].mean()), float(df["ninja_mae"].std())],
            "picp_mean_std": [float(df["picp"].mean()), float(df["picp"].std())],
        }
        save_json(summary, "exp04_multi_seed_summary.json")
        sum_df = pd.DataFrame(
            [
                {"metric": k.replace("_mean_std", ""), "mean": v[0], "std": v[1]}
                for k, v in summary.items()
                if k.endswith("_mean_std")
            ]
        )
        save_csv(sum_df, "exp04_multi_seed_summary.csv")

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(df))
        ax.bar(x - 0.2, df["nsrdb_r2"], 0.4, label="NSRDB R²")
        ax.bar(x + 0.2, df["ninja_r2"], 0.4, label="Ninja R²")
        ax.set_xticks(x)
        ax.set_xticklabels(df["seed"].astype(str))
        ax.set_xlabel("Seed")
        ax.legend()
        ax.set_title("Multi-seed stability")
        fig.tight_layout()
        save_fig(fig, "exp04_multi_seed_stability.png")

        failed = df[df["ninja_r2"] < 0]
        if len(failed):
            fail_note = (
                f"**Outliers:** {len(failed)} seed(s) with negative Ninja R² "
                f"({failed['seed'].tolist()}) — report them explicitly; mean±std alone misleads."
            )
        else:
            per = ", ".join(f"{int(s)}→{r:.3f}" for s, r in zip(df["seed"], df["ninja_r2"]))
            fail_note = (
                f"All {len(df)} seeds succeed on Ninja (per-seed R²: {per}). "
                f"Do not confuse this with the separate frozen-finetune failure (R²≈−0.51)."
            )
        text = f"""## Multi-seed stability

Seeds evaluated: {sorted(df['seed'].tolist())}

| Metric | mean ± std |
|--------|------------|
| NSRDB R² (—) | {summary['nsrdb_r2_mean_std'][0]:.6f} ± {summary['nsrdb_r2_mean_std'][1]:.6f} |
| NSRDB MAE (W) | {summary['nsrdb_mae_mean_std'][0]:.6g} ± {summary['nsrdb_mae_mean_std'][1]:.6g} |
| Ninja R² (—) | {summary['ninja_r2_mean_std'][0]:.4f} ± {summary['ninja_r2_mean_std'][1]:.4f} |
| Ninja MAE (kW) | {summary['ninja_mae_mean_std'][0]:.4f} ± {summary['ninja_mae_mean_std'][1]:.4f} |
| PICP (—) | {summary['picp_mean_std'][0]:.3f} ± {summary['picp_mean_std'][1]:.3f} |

{fail_note}
"""
        plain_summary(REPORT_DIR / "exp04_multi_seed_stability.md", text)

    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Phase 9 Exp4 multi-seed (optionally one seed)")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, e.g. 456")
    args = p.parse_args()
    seed_list = [int(x) for x in args.seeds.split(",") if x.strip()] or None
    run(skip_train=args.skip_train, seeds=seed_list)
