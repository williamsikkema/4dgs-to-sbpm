"""Build synchronized multi-camera image dataset + JSONL manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_mc_gs.calibration.load import Camera, load_cameras


def _frame_at_time(cap: cv2.VideoCapture, t_sec: float, fps: float) -> tuple[np.ndarray | None, int]:
    """Seek to frame index for local camera time t_sec."""
    idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None, idx
    return frame, idx


def build_synchronized_dataset(cfg: dict[str, Any]) -> Path:
    cal = load_cameras(cfg["calibration_path"])
    with open(cfg["offsets_path"], encoding="utf-8") as f:
        sync = json.load(f)
    offsets = {str(k): float(v) for k, v in sync["offsets_sec"].items()}

    videos = cfg["videos"]
    out_root = Path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    img_dir = out_root / "images"
    img_dir.mkdir(exist_ok=True)

    target_fps = float(cfg.get("target_fps", 30.0))
    dt = 1.0 / target_fps
    t_start = float(cfg.get("time_start_global", 0.0))
    t_end = cfg.get("time_end_global")

    caps: dict[str, cv2.VideoCapture] = {}
    fps_map: dict[str, float] = {}
    for cid, path in videos.items():
        caps[cid] = cv2.VideoCapture(path)
        f = caps[cid].get(cv2.CAP_PROP_FPS)
        fps_map[cid] = float(f) if f and f > 1e-3 else 30.0

    ref_id = str(cfg.get("reference_camera_id", sorted(videos.keys(), key=str)[0]))
    cap0 = caps[ref_id]
    nframes = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    fps0 = fps_map[ref_id]
    if t_end is None:
        t_end = nframes / fps0
    times = np.arange(t_start, t_end, dt)

    manifest_path = out_root / cfg.get("manifest_name", "manifest.jsonl")
    undistort = bool(cfg.get("undistort_images", True))
    ext = cfg.get("image_format", "png")

    rows: list[dict[str, Any]] = []
    for ti, t_g in enumerate(times):
        for cid in sorted(videos.keys(), key=lambda x: int(x) if x.isdigit() else x):
            cam: Camera = cal[cid]
            t_local = t_g - offsets[str(cid)]
            if t_local < 0:
                continue
            cap = caps[cid]
            fps = fps_map[cid]
            frame, fidx = _frame_at_time(cap, t_local, fps)
            if frame is None:
                continue
            if undistort:
                h, w = frame.shape[:2]
                new_K, _ = cv2.getOptimalNewCameraMatrix(cam.K, cam.dist, (w, h), 1, (w, h))
                frame = cv2.undistort(frame, cam.K, cam.dist, None, new_K)
                K_use = new_K
                dist_use = np.zeros(5, dtype=np.float64)
            else:
                K_use = cam.K
                dist_use = cam.dist

            fname = f"t{ti:06d}_cam{cid}.{ext}"
            fpath = img_dir / fname
            if ext.lower() == "png":
                cv2.imwrite(str(fpath), frame)
            else:
                cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            row = {
                "image_path": str(fpath.relative_to(out_root)),
                "t_global": float(t_g),
                "frame_index": int(fidx),
                "camera_id": str(cid),
                "K": K_use.tolist(),
                "dist": dist_use.reshape(-1).tolist(),
                "T_cw": cam.T_cw.tolist(),
                "mask_path": None,
            }
            rows.append(row)

    for c in caps.values():
        c.release()

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    meta = {
        "num_rows": len(rows),
        "t_start": t_start,
        "t_end": float(t_end),
        "target_fps": target_fps,
        "offsets_sec": offsets,
    }
    (out_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return manifest_path
