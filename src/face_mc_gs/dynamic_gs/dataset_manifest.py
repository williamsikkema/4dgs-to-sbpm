"""Load JSONL manifest rows for training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """Each line: image_path, t_global, camera_id, K, dist, T_cw."""

    def __init__(
        self,
        manifest_path: str | Path,
        root: Path | None = None,
        image_size: tuple[int, int] | None = None,
        t_normalize: bool = True,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        p = Path(manifest_path)
        self.root = root if root is not None else p.parent
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        ts = [r["t_global"] for r in self.rows]
        self.t_min = min(ts)
        self.t_max = max(ts)
        self.t_normalize = t_normalize
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def _norm_t(self, t: float) -> float:
        if not self.t_normalize or self.t_max <= self.t_min:
            return float(t)
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min + 1e-9) - 1.0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        img_path = self.root / r["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.image_size:
            img = img.resize(self.image_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        K = np.array(r["K"], dtype=np.float32)
        T = np.array(r["T_cw"], dtype=np.float32)
        R = T[:3, :3]
        t = T[:3, 3]
        return {
            "image": torch.from_numpy(chw),
            "t": torch.tensor(self._norm_t(float(r["t_global"])), dtype=torch.float32),
            "K": torch.from_numpy(K),
            "R_cw": torch.from_numpy(R),
            "t_cw": torch.from_numpy(t),
        }
