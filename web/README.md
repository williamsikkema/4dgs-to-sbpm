# Web demo — MediaPipe face mesh

Minimal static page: webcam with **MediaPipe Face Mesh** overlays. Use **Face markers** to toggle detection on or off.

## Run locally

```bash
cd web
python3 -m http.server 8080
```

Open [http://localhost:8080](http://localhost:8080) and allow camera access.

## Stack

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html) via jsDelivr CDN
