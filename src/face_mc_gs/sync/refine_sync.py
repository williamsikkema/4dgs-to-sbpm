"""Refine per-camera time offsets b_k with multi-view reprojection."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from face_mc_gs.calibration.load import Camera
from face_mc_gs.geometry.reprojection import project_points_world
from face_mc_gs.geometry.triangulation import triangulate_two_view


def refine_offsets(
    tracks: dict[str, dict[str, Any]],
    cameras: dict[str, Camera],
    reference_id: str,
    fps: float,
    *,
    use_affine_time: bool = False,
    huber_delta_px: float = 2.0,
    grid_step_frames: float = 2.0,
    max_nfev: int = 200,
    initial_offsets_sec: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Optimize b_k for k != ref with t_global = t_k + b_k (seconds).

    Uses two-view triangulation between ref and each other cam on a grid of ref frames,
    then minimizes sum of squared (Huber) reprojection errors.
    """
    ref = tracks[reference_id]
    f_ref = np.asarray(ref["frame_idx"], dtype=np.float64)
    u_ref = np.asarray(ref["u"], dtype=np.float64)
    v_ref = np.asarray(ref["v"], dtype=np.float64)

    other_ids = sorted([k for k in tracks.keys() if k != reference_id])
    if not other_ids:
        return {
            "reference": reference_id,
            "offsets_sec": {reference_id: 0.0},
            "affine": {},
            "success": True,
            "cost": 0.0,
            "message": "only one camera",
        }

    cam_ref = cameras[reference_id]

    def residuals(x: NDArray[np.floating]) -> NDArray[np.floating]:
        offsets: dict[str, float] = {reference_id: 0.0}
        aff: dict[str, tuple[float, float]] = {}
        idx = 0
        for oid in other_ids:
            if use_affine_time:
                aff[oid] = (float(x[idx]), float(x[idx + 1]))
                idx += 2
            else:
                offsets[oid] = float(x[idx])
                idx += 1

        res_list: list[float] = []
        f_min = float(np.min(f_ref))
        f_max = float(np.max(f_ref))
        grid = np.arange(f_min, f_max + 1e-6, grid_step_frames)

        for f0 in grid:
            u0 = float(np.interp(f0, f_ref, u_ref, left=np.nan, right=np.nan))
            v0 = float(np.interp(f0, f_ref, v_ref, left=np.nan, right=np.nan))
            if not (np.isfinite(u0) and np.isfinite(v0)):
                continue
            t_g = f0 / fps

            for oid in other_ids:
                cam_o = cameras[oid]
                otr = tracks[oid]
                fo = np.asarray(otr["frame_idx"], dtype=np.float64)
                u_o = np.asarray(otr["u"], dtype=np.float64)
                v_o = np.asarray(otr["v"], dtype=np.float64)

                if use_affine_time:
                    a, b = aff[oid]
                    t_local = (t_g - b) / (a + 1e-12)
                else:
                    t_local = t_g - offsets[oid]
                f_o = t_local * fps
                u1 = float(np.interp(f_o, fo, u_o, left=np.nan, right=np.nan))
                v1 = float(np.interp(f_o, fo, v_o, left=np.nan, right=np.nan))
                if not (np.isfinite(u1) and np.isfinite(v1)):
                    continue

                X = triangulate_two_view(
                    np.array([[u0, v0]]),
                    np.array([[u1, v1]]),
                    cam_ref,
                    cam_o,
                )[0]
                uv_r = project_points_world(X.reshape(1, 3), cam_ref)[0]
                uv_o = project_points_world(X.reshape(1, 3), cam_o)[0]
                res_list.extend(
                    [
                        uv_r[0] - u0,
                        uv_r[1] - v0,
                        uv_o[0] - u1,
                        uv_o[1] - v1,
                    ]
                )

        if not res_list:
            return np.array([0.0])
        return np.asarray(res_list, dtype=np.float64)

    n_other = len(other_ids)
    if use_affine_time:
        x0 = np.zeros(n_other * 2)
        for i in range(n_other):
            x0[2 * i] = 1.0
            oid = other_ids[i]
            if initial_offsets_sec and oid in initial_offsets_sec:
                x0[2 * i + 1] = initial_offsets_sec[oid]
    else:
        x0 = np.zeros(n_other)
        for i, oid in enumerate(other_ids):
            if initial_offsets_sec and oid in initial_offsets_sec:
                x0[i] = initial_offsets_sec[oid]

    ls = least_squares(residuals, x0, max_nfev=max_nfev, verbose=0)
    out_offsets: dict[str, float] = {reference_id: 0.0}
    out_affine: dict[str, tuple[float, float]] = {}
    idx = 0
    for oid in other_ids:
        if use_affine_time:
            out_affine[oid] = (float(ls.x[idx]), float(ls.x[idx + 1]))
            idx += 2
        else:
            out_offsets[oid] = float(ls.x[idx])
            idx += 1

    return {
        "reference": reference_id,
        "offsets_sec": out_offsets,
        "affine": out_affine,
        "success": bool(ls.success),
        "cost": float(ls.cost),
        "message": str(ls.message),
    }
