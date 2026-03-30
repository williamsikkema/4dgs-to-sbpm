"""Export positions, finite-diff and autograd velocity/acceleration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import savgol_filter

from face_mc_gs.dynamic_gs.deformation import DeformationNetwork
from face_mc_gs.dynamic_gs.gaussian_model import GaussianModel


def finite_diff_velocity(
    X: np.ndarray,
    dt: float,
    *,
    savgol_window: int = 0,
    savgol_poly: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    X: (T, N, 3)
    Returns v (T,N,3), a (T,N,3) with central differences.
    """
    T = X.shape[0]
    v = np.zeros_like(X)
    a = np.zeros_like(X)
    if T < 2:
        return v, a
    v[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
    v[0] = (X[1] - X[0]) / dt
    v[-1] = (X[-1] - X[-2]) / dt
    a[1:-1] = (X[2:] - 2 * X[1:-1] + X[:-2]) / (dt**2)
    a[0] = a[1]
    a[-1] = a[-2]
    if savgol_window >= 5 and savgol_window % 2 == 1 and T >= savgol_window:
        for j in range(X.shape[1]):
            for c in range(3):
                v[:, j, c] = savgol_filter(v[:, j, c], savgol_window, savgol_poly)
    return v, a


def autograd_vel_acc_simple(
    gauss: GaussianModel,
    deform: DeformationNetwork,
    t_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-point d x / dt and d^2 x / dt^2 at normalized time."""
    device = t_norm.device
    xyz = gauss.get_xyz()
    N = xyz.shape[0]
    t = t_norm.detach().clone().requires_grad_(True)
    tr = t.expand(N, 1)
    dxyz, _, _ = deform(xyz, tr)
    x = xyz + dxyz
    v = torch.zeros(N, 3, device=device, dtype=x.dtype)
    a = torch.zeros(N, 3, device=device, dtype=x.dtype)
    for i in range(N):
        for k in range(3):
            (gi,) = torch.autograd.grad(x[i, k], t, retain_graph=True, create_graph=True)
            v[i, k] = gi
            (ai,) = torch.autograd.grad(gi, t, retain_graph=True)
            a[i, k] = ai
    return v, a


class TrajectoryExporter:
    def __init__(self, checkpoint: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            data = torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            data = torch.load(checkpoint, map_location=self.device)
        cfg = data["cfg"]
        n = int(cfg["num_gaussians"])
        self.gauss = GaussianModel(n, sh_degree=int(cfg.get("sh_degree", 0))).to(self.device)
        dcfg = cfg.get("deformation", {})
        self.deform = DeformationNetwork(
            hidden_dim=int(dcfg.get("hidden_dim", 128)),
            num_layers=int(dcfg.get("num_layers", 4)),
            posenc_L=int(dcfg.get("posenc_L", 6)),
        ).to(self.device)
        self.gauss.load_state_dict(data["gauss"])
        self.deform.load_state_dict(data["deform"])
        self.t_min = float(data["t_min"])
        self.t_max = float(data["t_max"])
        self.gauss.eval()
        self.deform.eval()

    def positions_at(self, t_norm_list: list[float]) -> np.ndarray:
        """(T, N, 3)"""
        out = []
        with torch.no_grad():
            for tn in t_norm_list:
                t = torch.tensor(tn, device=self.device, dtype=torch.float32)
                xyz = self.gauss.get_xyz()
                N = xyz.shape[0]
                dxyz, _, _ = self.deform(xyz, t.expand(N, 1))
                out.append((xyz + dxyz).cpu().numpy())
        return np.stack(out, axis=0)


def export_all(cfg: dict[str, Any]) -> None:
    ck = Path(cfg["checkpoint"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    T = int(cfg.get("num_time_samples", 64))
    t_list = np.linspace(-1, 1, T).tolist()
    exp = TrajectoryExporter(ck)
    pos = exp.positions_at(t_list)
    t_real = np.linspace(exp.t_min, exp.t_max, T)
    dt = float(t_real[1] - t_real[0]) if T > 1 else 1.0
    v_fd, a_fd = finite_diff_velocity(
        pos,
        dt,
        savgol_window=int(cfg.get("savgol_window", 0)),
        savgol_poly=int(cfg.get("savgol_poly", 2)),
    )

    np.save(out / "positions.npy", pos)
    np.save(out / "velocity_fd.npy", v_fd)
    np.save(out / "acceleration_fd.npy", a_fd)
    np.save(out / "times_global.npy", t_real)

    speed = np.linalg.norm(v_fd, axis=-1)
    acc_mag = np.linalg.norm(a_fd, axis=-1)
    np.save(out / "speed_fd.npy", speed)
    np.save(out / "acceleration_magnitude_fd.npy", acc_mag)

    if cfg.get("derivative_method", "both") in ("both", "autograd"):
        v_ag = []
        a_ag = []
        for tn in t_list:
            t = torch.tensor(tn, device=exp.device, dtype=torch.float32)
            v, a = autograd_vel_acc_simple(exp.gauss, exp.deform, t)
            v_ag.append(v.detach().cpu().numpy())
            a_ag.append(a.detach().cpu().numpy())
        np.save(out / "velocity_autograd.npy", np.stack(v_ag, axis=0))
        np.save(out / "acceleration_autograd.npy", np.stack(a_ag, axis=0))

    with open(out / "summary.csv", "w", encoding="utf-8") as f:
        f.write("t_global,mean_speed_fd,mean_acc_fd\n")
        for i in range(T):
            f.write(f"{t_real[i]},{float(speed[i].mean())},{float(acc_mag[i].mean())}\n")

    if cfg.get("save_ply_snapshots"):
        for t_snap in cfg.get("ply_times", []):
            tn = 2.0 * (t_snap - exp.t_min) / (exp.t_max - exp.t_min + 1e-9) - 1.0
            p = exp.positions_at([tn])[0]
            _write_ply(out / f"cloud_{t_snap}.ply", p)


def _write_ply(path: Path, xyz: np.ndarray) -> None:
    n = xyz.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]}\n")
