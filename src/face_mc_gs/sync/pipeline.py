"""Full LED track extraction and sync estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_mc_gs.calibration.load import Camera, load_cameras
from face_mc_gs.sync.coarse_sync import combine_lags, cross_correlation_lag
from face_mc_gs.sync.led_detector import detect_led_frame
from face_mc_gs.sync.refine_sync import refine_offsets


def run_led_tracks(
    videos: dict[str, str],
    cfg: dict[str, Any],
    *,
    max_frames: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Process each video; return per-camera track dicts."""
    led_cfg = cfg.get("led_detection", {})
    hsv = led_cfg.get("color_mask_hsv")
    hsv_l = hsv_u = None
    if hsv:
        hsv_l = np.array(hsv["lower"], dtype=np.uint8)
        hsv_u = np.array(hsv["upper"], dtype=np.uint8)

    tracks: dict[str, dict[str, Any]] = {}
    fps = float(cfg.get("fps_assumed", 30.0))

    for cid, path in videos.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)
        vfps = cap.get(cv2.CAP_PROP_FPS)
        if vfps and vfps > 1e-3:
            fps = float(vfps)

        frames: list[int] = []
        us: list[float] = []
        vs: list[float] = []
        bright: list[float] = []
        conf: list[float] = []
        fidx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and fidx >= max_frames:
                break
            det = detect_led_frame(
                frame,
                grayscale=led_cfg.get("grayscale", True),
                channel=led_cfg.get("channel", "g"),
                threshold=int(led_cfg.get("threshold", 240)),
                min_area_px=float(led_cfg.get("min_area_px", 3)),
                max_area_px=float(led_cfg.get("max_area_px", 50000)),
                morphology_open=int(led_cfg.get("morphology_open", 0)),
                morphology_close=int(led_cfg.get("morphology_close", 0)),
                subpixel_gaussian_fit=bool(led_cfg.get("subpixel_gaussian_fit", False)),
                hsv_lower=hsv_l,
                hsv_upper=hsv_u,
            )
            frames.append(fidx)
            us.append(det.u if det.valid else np.nan)
            vs.append(det.v if det.valid else np.nan)
            bright.append(det.brightness)
            conf.append(det.confidence)
            fidx += 1
        cap.release()

        t_local = np.asarray(frames, dtype=np.float64) / fps
        tracks[cid] = {
            "frame_idx": np.asarray(frames, dtype=np.float64),
            "t_local": t_local,
            "u": np.asarray(us, dtype=np.float64),
            "v": np.asarray(vs, dtype=np.float64),
            "brightness": np.asarray(bright, dtype=np.float64),
            "confidence": np.asarray(conf, dtype=np.float64),
            "fps": fps,
        }
    return tracks


def coarse_offsets(
    tracks: dict[str, dict[str, Any]],
    reference_id: str,
    cfg: dict[str, Any],
) -> dict[str, int]:
    """Integer frame lags (positive = other is delayed vs ref)."""
    cs = cfg.get("coarse_sync", {})
    max_lag = int(cs.get("max_lag_frames", 120))
    signals = cs.get("signals", ["x", "y", "speed", "brightness"])
    ref = tracks[reference_id]
    fu = ref["u"]
    fv = ref["v"]
    sp = np.sqrt(np.gradient(fu) ** 2 + np.gradient(fv) ** 2)
    br = ref["brightness"]

    out: dict[str, int] = {reference_id: 0}
    for cid, tr in tracks.items():
        if cid == reference_id:
            continue
        lags = []
        weights = []
        gu = tr["u"]
        gv = tr["v"]
        gsp = np.sqrt(np.gradient(gu) ** 2 + np.gradient(gv) ** 2)
        gbr = tr["brightness"]
        for sig in signals:
            if sig == "x":
                a, w = cross_correlation_lag(fu, gu, max_lag)
            elif sig == "y":
                a, w = cross_correlation_lag(fv, gv, max_lag)
            elif sig == "speed":
                a, w = cross_correlation_lag(sp, gsp, max_lag)
            elif sig == "brightness":
                a, w = cross_correlation_lag(br, gbr, max_lag)
            else:
                continue
            lags.append(a)
            weights.append(max(w, 0.01))
        method = cs.get("combine", "median")
        out[cid] = combine_lags(lags, weights, method=method)  # type: ignore[arg-type]
    return out


def run_sync_full(
    cfg: dict[str, Any],
    cameras: dict[str, Camera] | None = None,
) -> dict[str, Any]:
    """Run detection, coarse CC, optional refinement; save outputs."""
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cpath = cfg["calibration_path"]
    if cameras is None:
        cameras = load_cameras(cpath)

    videos = cfg["videos"]
    tracks = run_led_tracks(videos, cfg)
    _serial = {}
    for k, v in tracks.items():
        _serial[k] = {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()}
    (out_dir / "tracks.json").write_text(json.dumps(_serial, indent=2), encoding="utf-8")

    ref = str(cfg["reference_camera_id"])
    lags = coarse_offsets(tracks, ref, cfg)
    with open(out_dir / "coarse_lags.json", "w", encoding="utf-8") as f:
        json.dump(lags, f, indent=2)

    fps_ref = float(tracks[ref]["fps"])
    initial_sec: dict[str, float] = {ref: 0.0}
    for cid, lag in lags.items():
        if cid != ref:
            initial_sec[cid] = float(lag) / fps_ref

    ref_cfg = cfg.get("refinement", {})
    refined = refine_offsets(
        tracks,
        cameras,
        ref,
        fps_ref,
        use_affine_time=bool(ref_cfg.get("use_affine_time", False)),
        huber_delta_px=float(ref_cfg.get("huber_delta_px", 2.0)),
        grid_step_frames=float(ref_cfg.get("grid_step_frames", 2.0)),
        max_nfev=int(ref_cfg.get("max_nfev", 200)),
        initial_offsets_sec=initial_sec,
    )

    result = {
        **refined,
        "coarse_lags_frames": lags,
        "fps_reference": fps_ref,
    }
    with open(out_dir / "offsets.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
