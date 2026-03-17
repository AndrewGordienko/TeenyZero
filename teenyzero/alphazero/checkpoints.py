from __future__ import annotations

import os
from pathlib import Path

import torch

from teenyzero.alphazero.config import (
    INPUT_PLANES,
    MODEL_CHANNELS,
    MODEL_RES_BLOCKS,
    MODEL_VERSION,
    REPLAY_ENCODER_VERSION,
)
from teenyzero.alphazero.model import AlphaNet
from teenyzero.alphazero.runtime import get_runtime_profile


PROFILE = get_runtime_profile()


def model_metadata():
    return {
        "model_version": MODEL_VERSION,
        "input_planes": INPUT_PLANES,
        "num_res_blocks": MODEL_RES_BLOCKS,
        "channels": MODEL_CHANNELS,
        "encoder_version": REPLAY_ENCODER_VERSION,
        "runtime_profile": PROFILE.name,
    }


def build_model():
    return AlphaNet(
        input_planes=INPUT_PLANES,
        num_res_blocks=MODEL_RES_BLOCKS,
        channels=MODEL_CHANNELS,
    )


def _extract_state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("meta", {})
    return payload, {}


def load_checkpoint(model, path, map_location="cpu", allow_partial=True):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {
            "loaded": False,
            "partial": False,
            "reason": "missing",
            "meta": {},
        }

    try:
        payload = torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        return {
            "loaded": False,
            "partial": False,
            "reason": f"read_failed: {exc}",
            "meta": {},
        }

    state_dict, meta = _extract_state_dict(payload)

    try:
        model.load_state_dict(state_dict)
        return {
            "loaded": True,
            "partial": False,
            "reason": "ok",
            "meta": meta,
        }
    except RuntimeError as exc:
        if not allow_partial:
            return {
                "loaded": False,
                "partial": False,
                "reason": str(exc),
                "meta": meta,
            }

    current_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    if not compatible:
        return {
            "loaded": False,
            "partial": False,
            "reason": "no_compatible_parameters",
            "meta": meta,
        }

    current_state.update(compatible)
    model.load_state_dict(current_state)
    return {
        "loaded": True,
        "partial": True,
        "reason": "partial",
        "meta": meta,
    }


def save_checkpoint(model, path, extra_meta=None):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    meta = model_metadata()
    if extra_meta:
        meta.update(extra_meta)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": meta,
        },
        tmp_path,
    )
    os.replace(tmp_path, checkpoint_path)
