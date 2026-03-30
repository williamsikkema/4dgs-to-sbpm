"""Run LED detection on all videos; save tracks.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from face_mc_gs.config_util import load_config
from face_mc_gs.sync.pipeline import run_led_tracks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tracks = run_led_tracks(cfg["videos"], cfg)
    serial = {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()} for k, v in tracks.items()}
    (out / "tracks.json").write_text(json.dumps(serial, indent=2), encoding="utf-8")
    print("Wrote", out / "tracks.json")


if __name__ == "__main__":
    main()
