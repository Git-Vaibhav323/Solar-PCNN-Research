## Embedding visualization

- **Before multitask (AR-PCNN):** NSRDB and Ninja embeddings partially overlap but retain domain-specific clusters — limited alignment.
- **After multitask (CD):** Shared encoder pulls Ninja embeddings closer to NSRDB meteorological structure while preserving domain labels in t-SNE/UMAP.
- **Interpretation:** Multitask learning aligns feature space for cross-domain transfer without collapsing physics-specific structure.
