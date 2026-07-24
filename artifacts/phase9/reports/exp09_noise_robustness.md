## Noise robustness

- Gaussian noise injected on normalized test features at 5%, 10%, 20% of column σ.
- **GHI/DNI** perturbations cause the largest R² drop for PCNN; AR-PCNN and CD degrade more gracefully due to physics prior + gate.
- **Temperature / wind** noise has smaller effect — consistent with lower feature importance.
- At 20% GHI noise, baseline PCNN loses more absolute R² than multitask CD, supporting robustness claims with caveats (still NSRDB physics target).
