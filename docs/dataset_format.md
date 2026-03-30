# Synchronized dataset manifest (JSONL)

Each line is one JSON object (one training sample):

| Field | Type | Description |
|-------|------|-------------|
| `image_path` | string | Relative to dataset root |
| `t_global` | float | Global timeline (seconds), aligned across cameras |
| `frame_index` | int | Source video frame index used |
| `camera_id` | string | Camera identifier |
| `K` | 3×3 | Intrinsics (possibly undistorted-optimal `newK`) |
| `dist` | array | Distortion coeffs (often zeros if image undistorted) |
| `T_cw` | 4×4 | World to camera homogeneous transform |
| `mask_path` | string or null | Optional segmentation mask |

Example line:

```json
{"image_path": "images/t000000_cam0.png", "t_global": 0.0, "frame_index": 0, "camera_id": "0", "K": [[800,0,320],[0,800,240],[0,0,1]], "dist": [0,0,0,0,0], "T_cw": [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]], "mask_path": null}
```

`dataset_meta.json` in the same folder summarizes time range and offsets.
