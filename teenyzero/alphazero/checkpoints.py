from __future__ import annotations

import os
from pathlib import Path

import torch

from teenyzero.alphazero.config import (
    INPUT_PLANES,
    MODEL_CHANNELS,
    MODEL_RES_BLOCKS,
    MODEL_VERSION,
    POLICY_HEAD_CHANNELS,
    REPLAY_ENCODER_VERSION,
    VALUE_HEAD_HIDDEN,
)
from teenyzero.alphazero.model import AlphaNet
from teenyzero.alphafold.model import AlphaFoldBoardModel
from teenyzero.alphazero.runtime import PROFILES, get_runtime_profile


PROFILE = get_runtime_profile()


def model_architecture_name(model_version=None):
    version = MODEL_VERSION if model_version is None else int(model_version)
    return "alphafold_board" if version >= 6 else "alphazero_resnet"


def model_metadata():
    return {
        "model_version": MODEL_VERSION,
        "architecture": model_architecture_name(),
        "input_planes": INPUT_PLANES,
        "num_res_blocks": MODEL_RES_BLOCKS,
        "channels": MODEL_CHANNELS,
        "policy_head_channels": POLICY_HEAD_CHANNELS,
        "value_head_hidden": VALUE_HEAD_HIDDEN,
        "encoder_version": REPLAY_ENCODER_VERSION,
        "runtime_profile": PROFILE.name,
    }


def build_model():
    return build_model_from_metadata(model_metadata())


def build_model_from_metadata(meta=None):
    meta = meta or {}
    runtime_profile_name = str(meta.get("runtime_profile", "")).strip().lower()
    profile = PROFILES.get(runtime_profile_name, PROFILE)
    model_version = int(meta.get("model_version", profile.model_version))
    architecture = str(meta.get("architecture") or model_architecture_name(model_version)).strip().lower()
    input_planes = int(meta.get("input_planes", profile.input_planes))
    num_res_blocks = int(meta.get("num_res_blocks", profile.model_res_blocks))
    channels = int(meta.get("channels", profile.model_channels))
    policy_head_channels = int(meta.get("policy_head_channels", profile.policy_head_channels))
    value_hidden = int(meta.get("value_head_hidden", profile.value_head_hidden))

    if architecture == "alphafold_board":
        return AlphaFoldBoardModel(
            input_planes=input_planes,
            num_relation_blocks=num_res_blocks,
            channels=channels,
            policy_head_channels=policy_head_channels,
            value_hidden=value_hidden,
        )
    return AlphaNet(
        input_planes=input_planes,
        num_res_blocks=num_res_blocks,
        channels=channels,
        policy_head_channels=policy_head_channels,
        value_hidden=value_hidden,
    )


def read_checkpoint_meta(path, map_location="cpu"):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {}
    try:
        payload = torch.load(checkpoint_path, map_location=map_location)
    except Exception:
        return {}
    _, meta = _extract_state_dict(payload)
    return meta


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
