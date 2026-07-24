## Hyperparameter sensitivity (lambda_external)

- Swept λ ∈ [0.0, 0.05, 0.1, 0.25, 0.5, 1.0] with frozen architecture; each run warm-starts from Phase-5 AR-PCNN.
- **Physics fidelity:** NSRDB R² stays ≈1 across λ; lowest NSRDB MAE at λ=1.0 (MAE=0.0002545 W).
- **Cross-domain:** Ninja R² peaks at λ=1.0 (R²=0.8639).
- **Trade-off:** Higher λ pulls the shared encoder toward ninja labels; very low λ under-trains the external head; λ≈0.25–1.0 balances physics MAE and ninja R² in this sweep.
- **UQ:** PICP remains near 0.95 after conformal scaling for all trained λ values.
