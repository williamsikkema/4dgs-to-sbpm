"""Pixel ↔ ray in world frame."""

from __future__ import annotations

import cv2
import numpy as np

from face_mc_gs.calibration.load import Camera


def undistort_pixels(uv: np.ndarray, cam: Camera) -> np.ndarray:
    """Undistort 2D pixels (N,2) using camera intrinsics + distortion."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    K = cam.K.astype(np.float64)
    dist = cam.dist.astype(np.float64).reshape(-1)
    if dist.size < 4:
        dist = np.pad(dist, (0, 5 - len(dist)))
    dist = dist[:5].reshape(1, 5)
    und = cv2.undistortPoints(uv, K, dist, P=K)
    return und.reshape(-1, 2)


def pixel_to_ray_world(uv: np.ndarray, cam: Camera, undistort: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Return camera center C_w (3,) and unit direction d_w (3,) for each pixel row.
    Ray: X(s) = C_w + s * d_w, s >= 0.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if undistort:
        uv_u = undistort_pixels(uv, cam)
    else:
        uv_u = uv
    Kinv = np.linalg.inv(cam.K)
    ones = np.ones((len(uv_u), 1), dtype=np.float64)
    xy1 = np.hstack([uv_u, ones])
    d_c = (Kinv @ xy1.T).T
    d_c = d_c / np.linalg.norm(d_c, axis=1, keepdims=True)
    R_cw = cam.R_cw
    t_cw = cam.t_cw.reshape(3)
    R_wc = R_cw.T
    d_w = (R_wc @ d_c.T).T
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    # Camera center in world: X_c = 0 -> X_w = R^T (-t)
    C_w = (-R_wc @ t_cw.reshape(3, 1)).reshape(3)
    centers = np.tile(C_w, (len(uv), 1))
    return centers, d_w
