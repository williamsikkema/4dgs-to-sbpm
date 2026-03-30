# Dynamic Gaussian training

Canonical 3D Gaussians + **deformation MLP** on `(x_canon, t)` → `Δx`. Rendering uses a **simple** differentiable splat (alpha-composited 2D Gaussians) by default; optional `gsplat` extra for higher fidelity (see `pyproject.toml`).

```bash
face-mc-build-dataset --config configs/dataset.yaml
face-mc-train --config configs/train.yaml
```

Losses: RGB L1, optional neighbor rigidity (small time perturbation), scale/opacity regularization.

Checkpoints: `output_dir/checkpoint.pt` plus periodic `checkpoint_ep*.pt`.
