## Hyperparameter sensitivity (lambda_external)

- Swept λ ∈ [0.0, 0.05, 0.1, 0.25, 0.5, 1.0] with frozen architecture; each run warm-starts from Phase-5 AR-PCNN.
- **Physics fidelity:** NSRDB R² stays ≈1 across λ; lowest NSRDB MAE at λ=0.0 (MAE=8.00633e-06 W) — expected when external loss is off if that is λ=0.
- **Cross-domain:** Ninja R² peaks at λ=0.25 (R²=0.8710); λ>0 stays ≈0.86–0.87. **λ=0 → Ninja R²≈-0.51** (external head never trained — control, not a seed failure).
- **Trade-off:** Any λ>0 trains the ninja head; λ≈0.25–1.0 keeps physics MAE tiny while holding Ninja R². PICP is weaker at very low λ before recovering near λ≥0.25.
- **UQ:** PICP after conformal scaling is listed per λ in the CSV.
