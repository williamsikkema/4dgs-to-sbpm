# Sync pipeline

1. **LED detection** (`face-mc-run-led-detection`): per-frame blob + centroid → `tracks.json`.
2. **Coarse alignment**: normalized cross-correlation of `x`, `y`, speed, brightness vs reference camera → integer frame lag per camera.
3. **Refinement** (`face-mc-run-sync`): optimize `b_k` (seconds) so `t_global = t_k + b_k` (reference `b_0=0`), minimizing two-view triangulation reprojection residuals on a temporal grid.

Outputs in `output_dir`: `tracks.json`, `coarse_lags.json`, `offsets.json`.

```bash
face-mc-run-led-detection --config configs/sync.yaml
face-mc-run-sync --config configs/sync.yaml
```

Point `calibration_path` and `videos` in `configs/sync.yaml` to your data.
