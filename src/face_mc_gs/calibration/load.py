"""Load camera intrinsics / extrinsics from JSON or YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class Camera:
    """Single pinhole camera: world ↔ camera using OpenCV conventions."""

    camera_id: str
    K: np.ndarray  # 3x3
    dist: np.ndarray  # 1x5 or (5,) etc.
    T_cw: np.ndarray  # 4x4 homogeneous: X_c = T_cw @ X_w (column vectors)

    @property
    def R_cw(self) -> np.ndarray:
        return self.T_cw[:3, :3].copy()

    @property
    def t_cw(self) -> np.ndarray:
        return self.T_cw[:3, 3].copy()

    def P(self) -> np.ndarray:
        """3x4 projection matrix: x ~ K [R|t] X_w."""
        R = self.R_cw
        t = self.t_cw.reshape(3, 1)
        Rt = np.hstack([R, t])
        return self.K @ Rt


def _parse_camera(cid: str, raw: dict[str, Any]) -> Camera:
    K = np.asarray(raw["K"], dtype=np.float64)
    dist = np.asarray(raw.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
    if "T_cw" in raw:
        T = np.asarray(raw["T_cw"], dtype=np.float64)
    elif "R_cw" in raw and "t_cw" in raw:
        R = np.asarray(raw["R_cw"], dtype=np.float64)
        t = np.asarray(raw["t_cw"], dtype=np.float64).reshape(3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
    else:
        raise KeyError("Camera needs T_cw or (R_cw, t_cw)")
    if T.shape != (4, 4):
        raise ValueError(f"camera {cid}: T_cw must be 4x4")
    if K.shape != (3, 3):
        raise ValueError(f"camera {cid}: K must be 3x3")
    return Camera(camera_id=str(cid), K=K, dist=dist, T_cw=T)


def load_cameras_json(path: str | Path) -> dict[str, Camera]:
    import json

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    cams = data.get("cameras", data)
    out: dict[str, Camera] = {}
    for cid, raw in cams.items():
        out[str(cid)] = _parse_camera(str(cid), raw)
    return out


def load_cameras_yaml(path: str | Path) -> dict[str, Camera]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    cams = data.get("cameras", data)
    out: dict[str, Camera] = {}
    for cid, raw in cams.items():
        out[str(cid)] = _parse_camera(str(cid), raw)
    return out


def load_cameras(path: str | Path) -> dict[str, Camera]:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        return load_cameras_yaml(p)
    return load_cameras_json(p)
