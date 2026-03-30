# Trajectory export

```bash
face-mc-export --config configs/export.yaml
```

Produces:

- `positions.npy` — `(T, G, 3)`
- `velocity_fd.npy`, `acceleration_fd.npy` — central finite differences vs `times_global.npy`
- `velocity_autograd.npy`, `acceleration_autograd.npy` — if `derivative_method: both`
- `summary.csv`, speed / acceleration magnitude arrays

**Finite differences** are robust to black-box time sampling; **autograd** gives `dx/dt` from the learned `F(x,t)` and is smoother if the network is well identified.

```bash
face-mc-compare-derivatives --config configs/export.yaml
```
