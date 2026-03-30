"""
Optional gsplat-backed renderer. Install: pip install '.[gsplat]'

TODO: wire gsplat.rasterization to canonical+deformed Gaussians and camera tuple.
This module is a placeholder so training can swap renderers without changing sync/export.
"""

from __future__ import annotations

try:
    import gsplat  # noqa: F401
except ImportError:
    gsplat = None


def is_available() -> bool:
    return gsplat is not None
