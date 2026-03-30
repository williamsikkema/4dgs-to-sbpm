"""Export trajectories and derivatives."""

from __future__ import annotations

import argparse

from face_mc_gs.config_util import load_config
from face_mc_gs.export.trajectory import export_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    export_all(cfg)
    print("Export done ->", cfg["output_dir"])


if __name__ == "__main__":
    main()
