from __future__ import annotations

import os


_NATIVE_SPEEDUPS = None
_NATIVE_IMPORT_ERROR = None


def requested_board_backend():
    return os.environ.get("TEENYZERO_BOARD_BACKEND", "auto").strip().lower() or "auto"


def native_speedups_module():
    global _NATIVE_SPEEDUPS, _NATIVE_IMPORT_ERROR
    if _NATIVE_SPEEDUPS is not None:
        return _NATIVE_SPEEDUPS
    if _NATIVE_IMPORT_ERROR is not None:
        return None

    try:
        from teenyzero.native import speedups
    except Exception as exc:
        _NATIVE_IMPORT_ERROR = exc
        return None

    _NATIVE_SPEEDUPS = speedups
    return _NATIVE_SPEEDUPS


def native_speedups_available():
    return native_speedups_module() is not None


def resolve_board_backend_name():
    requested = requested_board_backend()
    if requested == "python":
        return "python"
    if requested == "native" and native_speedups_available():
        return "native"
    if requested == "native":
        return "python"
    return "native" if native_speedups_available() else "python"


def board_backend_payload():
    return {
        "requested_board_backend": requested_board_backend(),
        "resolved_board_backend": resolve_board_backend_name(),
        "native_speedups_available": native_speedups_available(),
    }
