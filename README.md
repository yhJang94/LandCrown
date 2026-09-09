# LandCrown

Two-stage pipeline that generates the crown of a missing tooth from the surrounding
dentition.

| Stage | Folder | Output | Backbone |
|-------|--------|--------|----------|
| **1** | [`dual_graph/`](dual_graph/) | missing-tooth landmarks + a **coarse 128-point crown** | dual-graph autoencoder (GraphMAE) |
| **2** | [`fine_generator/`](fine_generator/) | a **dense point cloud** upsampled from the coarse crown | masked point-diffusion autoencoder (DiffPMAE) |

The stages are coupled **offline**: Stage 1 writes its predicted coarse points to disk,
Stage 2 loads them as input. They have **separate dependencies and conda environments** —
follow each stage's own README / `requirements.txt`.

## Layout

```
dual_graph/       Stage 1  — see dual_graph/README.md
fine_generator/   Stage 2  — see fine_generator/README.md
LICENSE                   MIT (Stage 1 is derived from GraphMAE)
```

## Attribution

- **Stage 1** builds on [GraphMAE](https://github.com/THUDM/GraphMAE) (THUDM, MIT) — see `LICENSE`.
- **Stage 2** adopts [DiffPMAE](https://github.com/TyraelDLee/DiffPMAE) as its backbone.
