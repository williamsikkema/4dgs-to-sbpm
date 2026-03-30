"""Build synchronized dataset manifest."""

from __future__ import annotations

import argparse

from face_mc_gs.config_util import load_config
from face_mc_gs.data.builder import build_synchronized_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = build_synchronized_dataset(cfg)
    print("Wrote manifest", p)


if __name__ == "__main__":
    main()
