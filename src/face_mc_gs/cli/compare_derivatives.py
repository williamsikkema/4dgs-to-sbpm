"""Compare finite-difference vs autograd velocity norms."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from face_mc_gs.config_util import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    v_fd = np.load(out / "velocity_fd.npy")
    p_ag = out / "velocity_autograd.npy"
    if not p_ag.exists():
        print("velocity_autograd.npy missing; run export with derivative_method: both")
        return
    v_ag = np.load(p_ag)
    err = np.linalg.norm(v_fd - v_ag, axis=-1).mean()
    print(f"mean ||v_fd - v_ag||: {err:.6f}")
    print(f"mean ||v_fd||: {np.linalg.norm(v_fd, axis=-1).mean():.6f}")
    print(f"mean ||v_ag||: {np.linalg.norm(v_ag, axis=-1).mean():.6f}")


if __name__ == "__main__":
    main()
