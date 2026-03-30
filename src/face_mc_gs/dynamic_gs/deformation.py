"""MLP deformation field: canonical position + encoded time -> delta position."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def positional_encoding(x: torch.Tensor, L: int) -> torch.Tensor:
    """x (N,1) scalar time -> [x, sin(2^i pi x), cos(...)] for i=0..L-1 => dim 1+2L."""
    out = [x]
    for i in range(L):
        freq = 2.0**i * math.pi
        out.append(torch.sin(freq * x))
        out.append(torch.cos(freq * x))
    return torch.cat(out, dim=-1)


class DeformationNetwork(nn.Module):
    """
    Maps (x_canon, t) -> Δx. Time normalized to [-1,1] externally.

    Fixed canonical neighbors regularize relative motion; this network only outputs smooth deltas.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        posenc_L: int = 6,
        delta_rotation: bool = False,
        delta_scale: bool = False,
    ) -> None:
        super().__init__()
        self.posenc_L = posenc_L
        in_dim = 3 + (1 + 2 * posenc_L)

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 3)
        self.delta_rotation = delta_rotation
        self.delta_scale = delta_scale
        if delta_rotation or delta_scale:
            self.extra = nn.Linear(hidden_dim, (6 if delta_rotation else 0) + (3 if delta_scale else 0))
        else:
            self.extra = None

    def forward(self, xyz: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        xyz: (N,3), t: (N,1) or (N,) in [-1,1]
        Returns delta_xyz (N,3).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        te = positional_encoding(t, self.posenc_L)
        x = torch.cat([xyz, te], dim=-1)
        h = self.net(x)
        d = self.head(h)
        if self.extra is not None:
            raise NotImplementedError("delta_rotation/scale optional path not implemented in baseline")
        return d, None, None
