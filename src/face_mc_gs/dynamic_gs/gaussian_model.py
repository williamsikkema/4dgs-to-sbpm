"""Canonical 3D Gaussians: positions, axis-aligned scale, opacity, color."""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianModel(nn.Module):
    """
    Canonical Gaussians; deformation network moves centers through time.

    Axis-aligned covariance diag(s^2) for stability and fast training in the baseline renderer.
    """

    def __init__(self, num_points: int, sh_degree: int = 0) -> None:
        super().__init__()
        self.num_points = num_points
        self.sh_degree = sh_degree
        pts = torch.randn(num_points, 3) * 0.1
        self._xyz = nn.Parameter(pts)
        self._log_scale = nn.Parameter(torch.log(torch.abs(torch.randn(num_points, 3) * 0.05) + 0.02))
        self._opacity = nn.Parameter(torch.logit(torch.full((num_points, 1), 0.5)))
        self._features_dc = nn.Parameter(torch.rand(num_points, 3) * 0.5 + 0.25)

    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    def get_scale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._log_scale) + 1e-4

    def get_covariance(self) -> torch.Tensor:
        """(N,3,3) diagonal covariance in world frame."""
        s = self.get_scale()
        return torch.diag_embed(s * s)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    def get_color(self) -> torch.Tensor:
        return torch.sigmoid(self._features_dc)
