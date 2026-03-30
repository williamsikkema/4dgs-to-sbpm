"""Multi-view triangulation and ray-ray geometry."""

from __future__ import annotations

import cv2
import numpy as np

from face_mc_gs.calibration.load import Camera


def triangulate_two_view(
    uv0: np.ndarray,
    uv1: np.ndarray,
    cam0: Camera,
    cam1: Camera,
) -> np.ndarray:
    """Triangulate corresponding points (N,2) using DLT; returns Nx3 world points."""
    uv0 = np.asarray(uv0, dtype=np.float64).reshape(-1, 2)
    uv1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    P0 = cam0.P()
    P1 = cam1.P()
    n = len(uv0)
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        x = cv2.triangulatePoints(
            P0.astype(np.float64),
            P1.astype(np.float64),
            uv0[i].reshape(2, 1).astype(np.float64),
            uv1[i].reshape(2, 1).astype(np.float64),
        )
        x = x.reshape(4)
        x = x[:3] / (x[3] + 1e-12)
        out[i] = x
    return out


def ray_ray_closest(
    C0: np.ndarray,
    d0: np.ndarray,
    C1: np.ndarray,
    d1: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Closest points between two rays X=C0+s d0 and Y=C1+t d1.
    Returns midpoint of the shortest segment and the distance between rays.
    """
    C0 = np.asarray(C0, dtype=np.float64).reshape(3)
    d0 = np.asarray(d0, dtype=np.float64).reshape(3)
    C1 = np.asarray(C1, dtype=np.float64).reshape(3)
    d1 = np.asarray(d1, dtype=np.float64).reshape(3)
    d0 = d0 / np.linalg.norm(d0)
    d1 = d1 / np.linalg.norm(d1)
    w0 = C0 - C1
    a = np.dot(d0, d0)
    b = np.dot(d0, d1)
    c = np.dot(d1, d1)
    d = np.dot(d0, w0)
    e = np.dot(d1, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        mid = 0.5 * (C0 + C1)
        return mid, float(np.linalg.norm(C0 - C1))
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    p0 = C0 + s * d0
    p1 = C1 + t * d1
    mid = 0.5 * (p0 + p1)
    dist = float(np.linalg.norm(p0 - p1))
    return mid, dist
