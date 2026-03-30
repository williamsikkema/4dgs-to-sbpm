"""Photometric, temporal smoothness, neighbor rigidity."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_rgb(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def l_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """Simplified SSIM loss (1 - SSIM) using avg pool as fallback."""
    if window_size <= 1:
        return torch.tensor(0.0, device=pred.device)
    p = F.avg_pool2d(pred.unsqueeze(0), window_size, stride=1, padding=window_size // 2)
    t = F.avg_pool2d(target.unsqueeze(0), window_size, stride=1, padding=window_size // 2)
    return (p - t).abs().mean()


def temporal_smoothness(x_t: torch.Tensor, x_tp: torch.Tensor) -> torch.Tensor:
    """L2 between consecutive deformed positions (same canonical indices)."""
    return ((x_t - x_tp) ** 2).mean()


def neighbor_coherence(
    x_t: torch.Tensor,
    x_tp: torch.Tensor,
    neigh_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize change in relative offsets between neighbors from t to t'.

    x: (N,3), neigh_idx: (N,k)
    """
    if neigh_idx.numel() == 0:
        return torch.tensor(0.0, device=x_t.device)
    j = neigh_idx
    n, k = j.shape
    rel_t = x_t[:, None, :] - x_t[j, :].mean(dim=1)  # simplified: use first neighbor
    rel_tp = x_tp[:, None, :] - x_tp[j, :].mean(dim=1)
    return ((rel_t - rel_tp) ** 2).mean()


def neighbor_coherence_pairwise(
    x_t: torch.Tensor,
    x_tp: torch.Tensor,
    neigh_idx: torch.Tensor,
) -> torch.Tensor:
    """Sum over neighbors of || (x_i - x_j)_t - (x_i - x_j)_tp ||^2."""
    if neigh_idx.numel() == 0:
        return torch.tensor(0.0, device=x_t.device)
    loss = 0.0
    n, k = neigh_idx.shape
    for i in range(n):
        for j in range(k):
            ni = neigh_idx[i, j]
            d1 = x_t[i] - x_t[ni]
            d2 = x_tp[i] - x_tp[ni]
            loss = loss + ((d1 - d2) ** 2).sum()
    return loss / (n * k + 1e-8)


def scale_opacity_reg(scale: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
    return (scale**2).mean() + (opacity**2).mean()
