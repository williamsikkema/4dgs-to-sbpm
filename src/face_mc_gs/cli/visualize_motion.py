"""Plot tracks, trajectories, optional NPZ from export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from face_mc_gs.analysis.plotting import plot_led_tracks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    from face_mc_gs.config_util import load_config

    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    (out / "plots").mkdir(parents=True, exist_ok=True)
    sync_dir = Path(cfg.get("sync_dir", "data/sync_out"))
    if (sync_dir / "tracks.json").exists():
        plot_led_tracks(sync_dir / "tracks.json", out / "plots")
    if (out / "positions.npy").exists():
        pos = np.load(out / "positions.npy")
        # mean trajectory magnitude over Gaussians
        m = np.linalg.norm(pos - pos.mean(axis=1, keepdims=True), axis=-1).mean(axis=1)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(m)
        ax.set_title("mean ||x - mean|| over Gaussians vs time index")
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "plots" / "motion_curve.png", dpi=120)
        plt.close(fig)
    print("Wrote plots under", out / "plots")


if __name__ == "__main__":
    main()
