"""Train dynamic Gaussian model."""

from __future__ import annotations

import argparse

from face_mc_gs.config_util import load_config
from face_mc_gs.dynamic_gs.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
