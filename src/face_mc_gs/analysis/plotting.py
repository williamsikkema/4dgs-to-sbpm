"""Diagnostic plots for sync and motion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_led_tracks(tracks_path: str | Path, out_dir: str | Path) -> None:
    """tracks.json from sync pipeline."""
    p = Path(tracks_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid, tr in data.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        f = np.asarray(tr["frame_idx"])
        ax.plot(f, tr["u"], label="u")
        ax.plot(f, tr["v"], label="v")
        ax.set_title(f"LED track cam {cid}")
        ax.legend()
        fig.savefig(out_dir / f"track_uv_{cid}.png", dpi=120)
        plt.close(fig)


def plot_sync_correlation(
    ref_signal: np.ndarray,
    oth_signal: np.ndarray,
    out_path: str | Path,
    title: str = "NCC alignment",
) -> None:
    ref_signal = np.asarray(ref_signal, dtype=np.float64)
    oth_signal = np.asarray(oth_signal, dtype=np.float64)
    n = min(len(ref_signal), len(oth_signal))
    ref_signal = ref_signal[:n] - np.nanmean(ref_signal[:n])
    oth_signal = oth_signal[:n] - np.nanmean(oth_signal[:n])
    lags = range(-min(60, n // 4), min(60, n // 4) + 1)
    corrs = []
    for lag in lags:
        if lag > 0:
            a, b = ref_signal[lag:], oth_signal[:-lag]
        elif lag < 0:
            a, b = ref_signal[:lag], oth_signal[-lag:]
        else:
            a, b = ref_signal, oth_signal
        if len(a) < 4:
            corrs.append(0.0)
            continue
        corrs.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(list(lags), corrs)
    ax.set_xlabel("lag (frames)")
    ax.set_ylabel("normalized correlation")
    ax.set_title(title)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
