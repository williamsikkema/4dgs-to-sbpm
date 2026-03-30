"""World 3D points → image pixels."""

from __future__ import annotations

import cv2
import numpy as np

from face_mc_gs.calibration.load import Camera


def project_points_world(X_w: np.ndarray, cam: Camera) -> np.ndarray:
    """
    Project Nx3 world points to distorted pixels (Nx2).
    """
    X = np.asarray(X_w, dtype=np.float64).reshape(-1, 3)
    R = cam.R_cw
    t = cam.t_cw.reshape(3, 1)
    Xc = (R @ X.T + t).T
    z = Xc[:, 2:3]
    z[z < 1e-9] = 1e-9
    x = Xc[:, 0:1] / z
    y = Xc[:, 1:2] / z
    xy = np.hstack([x, y]).reshape(-1, 1, 2)
    K = cam.K.astype(np.float64)
    dist = cam.dist.astype(np.float64).reshape(-1)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    img, _ = cv2.projectPoints(
        Xc.astype(np.float32),
        rvec,
        tvec,
        K,
        dist,
    )
    return img.reshape(-1, 2)
