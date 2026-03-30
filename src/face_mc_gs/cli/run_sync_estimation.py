"""Full coarse + refined sync."""

from __future__ import annotations

import argparse

from face_mc_gs.config_util import load_config
from face_mc_gs.sync.pipeline import run_sync_full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = run_sync_full(cfg)
    print("offsets:", r.get("offsets_sec"))


if __name__ == "__main__":
    main()
