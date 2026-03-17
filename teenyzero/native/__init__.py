from __future__ import annotations

try:
    from . import _speedups as speedups
except Exception:
    speedups = None
