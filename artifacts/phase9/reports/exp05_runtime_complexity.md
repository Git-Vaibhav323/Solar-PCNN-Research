## Runtime & complexity

- All three models share the same 128→64→32 encoder scale; CD adds only a small external head (+545 params).
- Inference latency scales linearly with batch size; AR-PCNN/CD add negligible overhead vs PCNN on GPU.
- Training times are taken from original Phase 3–6 run summaries where available.
