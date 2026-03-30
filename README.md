# 4DGS to SBPM — Multiview face motion pipeline

End-to-end starter codebase: **LED wand temporal sync** → **synchronized calibrated dataset** → **canonical 3D Gaussians + time-conditioned deformation** → **position / velocity / acceleration export**.

| Area | Doc |
|------|-----|
| Sync (detection, offsets) | [docs/README_sync.md](docs/README_sync.md) |
| Training (dynamic GS) | [docs/README_train.md](docs/README_train.md) |
| Export & derivatives | [docs/README_export.md](docs/README_export.md) |
| Firmware (LED wand) | [firmware/wand_led/README.md](firmware/wand_led/README.md) |
| Dataset format | [docs/dataset_format.md](docs/dataset_format.md) |
| Browser MediaPipe demo | [web/](web/) — `cd web && python3 -m http.server 8080` |

## Install

```bash
cd 4dgs-to-sbpm
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# Optional full splat renderer:
# pip install -e ".[gsplat]"
```

## Pipeline (CLI)

```bash
# 1) Detect LED in each video → tracks NPZ
face-mc-run-led-detection --config configs/sync.yaml

# 2) Estimate per-camera time offsets → offsets.json + plots
face-mc-run-sync --config configs/sync.yaml

# 3) Build synchronized image dataset + manifest
face-mc-build-dataset --config configs/dataset.yaml

# 4) Train dynamic Gaussian model
face-mc-train --config configs/train.yaml

# 5) Export trajectories (FD + optional autograd derivatives)
face-mc-export --config configs/export.yaml

# 6) Compare finite differences vs autograd
face-mc-compare-derivatives --config configs/export.yaml

# 7) Plots / diagnostics
face-mc-visualize --config configs/export.yaml
```

## Design notes

- **Sync before training:** Global timestamps are required so all cameras contribute to one coherent dynamic model; wand gives observability of `t_k → t_global`.
- **Canonical Gaussians:** A fixed set of 3D primitives with identity over time; motion is a **deformation field** `Δx(x_canon, t)` so each index `i` is a dense tracked point.
- **Fixed neighbor graph:** `k`-NN in canonical space stabilizes relative motion regularization; neighborhoods do not change every frame (see `NeighborGraph`).

## Tests

```bash
pytest tests/ -q
```

## License

Apache-2.0 (align with common CV stack); adjust as needed.
