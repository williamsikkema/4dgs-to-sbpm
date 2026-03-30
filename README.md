# 4DGS to SBPM

Minimal static page: webcam feed with **MediaPipe Face Mesh** overlays (dense mesh + eyes + face oval). Use the **Face markers** checkbox to turn face detection and drawing on or off (when off, inference is skipped and the overlay is cleared).

## Run locally

Browsers require a **secure context** for the camera. Opening the file as `file://` often blocks `getUserMedia`. Serve the folder over HTTP:

```bash
cd 4dgs-to-sbpm
python3 -m http.server 8080
```

Then open [http://localhost:8080](http://localhost:8080) and allow camera access.

If you are not on `localhost`, you may need HTTPS for camera permissions.

## Stack

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html) via jsDelivr CDN (`face_mesh`, `camera_utils`, `drawing_utils`)
