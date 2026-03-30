"""Differentiable simplified splat: project Gaussians, accumulate 2D Gaussian blobs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def world_to_camera(
    X_w: torch.Tensor,
    R_cw: torch.Tensor,
    t_cw: torch.Tensor,
) -> torch.Tensor:
    """X_w (N,3), R_cw (3,3), t_cw (3,) -> X_c (N,3)."""
    return (R_cw @ X_w.T).T + t_cw


def project_points(X_c: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """X_c (N,3) in camera frame -> uv (N,2) pixel coords."""
    z = X_c[:, 2:3].clamp(min=1e-4)
    x = X_c[:, 0:1] / z
    y = X_c[:, 1:2] / z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x + cx
    v = fy * y + cy
    return torch.cat([u, v], dim=1)


def render_simple(
    means_w: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    scale: torch.Tensor,
    R_cw: torch.Tensor,
    t_cw: torch.Tensor,
    K: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """
    means_w (N,3), colors (N,3), opacity (N,1), scale (N,3) axis world (use as screen scale proxy),
    returns image (3,H,W) in [0,1].
    """
    device = means_w.device
    H, W = image_size
    X_c = world_to_camera(means_w, R_cw, t_cw)
    uv = project_points(X_c, K)
    z = X_c[:, 2].clamp(min=1e-2)
    # screen-space sigma ~ focal * world_scale / z
    f = (K[0, 0] + K[1, 1]) / 2.0
    sig = (f * scale.mean(dim=1) / z).clamp(min=0.5, max=float(max(H, W)))

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1)  # H,W,2

    acc = torch.zeros(H, W, 3, device=device)
    trans = torch.ones(H, W, device=device)

    order = torch.argsort(z, descending=True)
    for i in order:
        u, v = uv[i, 0], uv[i, 1]
        s = sig[i]
        o = opacity[i, 0]
        c = colors[i]
        d2 = (grid[..., 0] - u) ** 2 + (grid[..., 1] - v) ** 2
        g = torch.exp(-0.5 * d2 / (s * s + 1e-4))
        a = o * g
        acc = acc + trans[..., None] * a[..., None] * c[None, None, :]
        trans = trans * (1.0 - a).clamp(min=0.0, max=1.0)

    return acc.clamp(0, 1).permute(2, 0, 1)
