"""k-nearest neighbors in canonical xyz for rigidity regularization."""

from __future__ import annotations

import torch


def build_neighbor_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    xyz: (N,3) tensor
    Returns (N, k) long indices (excluding self; use next k neighbors).
    """
    n = xyz.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return torch.zeros(n, 0, dtype=torch.long, device=xyz.device)
    d = torch.cdist(xyz, xyz)
    d.fill_diagonal_(float("inf"))
    _, idx = torch.topk(d, k, largest=False, dim=1)
    return idx
