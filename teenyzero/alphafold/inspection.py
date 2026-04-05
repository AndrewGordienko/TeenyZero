from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import chess
import numpy as np
import torch

from teenyzero.alphafold.features import (
    GEOMETRY_TARGET_NAMES,
    board_from_encoded_state,
    build_square_target_tensor_from_state,
)


@dataclass(frozen=True)
class ReplayShard:
    shard_format: str
    path: str
    sample_count: int
    mtime: float
    states_path: str | None = None


def sample_replay_state(data_dir: str, rng_seed: int | None = None, recent_file_count: int = 64):
    rng = np.random.default_rng(rng_seed)
    shards = _replay_shards(data_dir)
    if not shards:
        return None
    candidates = shards[-max(1, int(recent_file_count)) :]
    weights = np.asarray([max(1, shard.sample_count) for shard in candidates], dtype=np.float64)
    weights = weights / weights.sum()
    shard = candidates[int(rng.choice(len(candidates), p=weights))]
    sample_idx = int(rng.integers(0, max(1, shard.sample_count)))
    state = _load_state_sample(shard, sample_idx)
    if state is None:
        return None
    return {
        "source": "replay",
        "state": state,
    }


def sample_random_state(rng_seed: int | None = None, min_plies: int = 6, max_plies: int = 40):
    rng = np.random.default_rng(rng_seed)
    board = _sample_board(rng, min_plies=min_plies, max_plies=max_plies)
    if board is None:
        return None
    from teenyzero.alphafold.features import encode_board_state

    return {
        "source": "random",
        "state": encode_board_state(board),
    }


def sample_geometry_payload(state, source: str, model=None, device: str = "cpu", top_k_connections: int = 8):
    state_array = np.asarray(state, dtype=np.float32)
    board = board_from_encoded_state(state_array)
    targets = build_square_target_tensor_from_state(state_array)
    payload = {
        "source": source,
        "board_fen": board.board_fen(),
        "targets": {
            name: targets[index].tolist()
            for index, name in enumerate(GEOMETRY_TARGET_NAMES)
        },
    }

    if model is None:
        return payload

    tensor = torch.from_numpy(state_array).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        _, _, aux_outputs, attention = model(tensor, return_aux=True, return_attention=True)

    predicted_maps = {
        name: torch.sigmoid(aux_outputs[name])[0].detach().cpu().numpy().astype(np.float32).tolist()
        for name in GEOMETRY_TARGET_NAMES
    }
    payload["predictions"] = predicted_maps
    if attention is not None:
        attention_matrix = attention[0].detach().cpu().numpy().astype(np.float32)
        payload["top_connections"] = top_attention_connections(attention_matrix, limit=top_k_connections)
    else:
        payload["top_connections"] = []
    return payload


def top_attention_connections(attention_matrix: np.ndarray, limit: int = 8):
    matrix = np.asarray(attention_matrix, dtype=np.float32)
    if matrix.shape != (64, 64):
        return []

    ranked = []
    for source in range(64):
        for target in range(64):
            if source == target:
                continue
            ranked.append((float(matrix[source, target]), source, target))
    ranked.sort(reverse=True)

    connections = []
    for weight, source, target in ranked[: max(1, int(limit))]:
        connections.append(
            {
                "weight": weight,
                "from": chess.square_name(source),
                "to": chess.square_name(target),
            }
        )
    return connections


def _replay_shards(data_dir: str):
    shards = []
    for path in glob.glob(os.path.join(data_dir, "*.npz")):
        try:
            with np.load(path) as data:
                sample_count = int(data["states"].shape[0])
        except Exception:
            continue
        shards.append(
            ReplayShard(
                shard_format="npz",
                path=path,
                sample_count=sample_count,
                mtime=os.path.getmtime(path),
            )
        )

    for meta_path in glob.glob(os.path.join(data_dir, "*.meta.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            stem = meta_path[: -len(".meta.json")]
            states_path = stem + ".states.npy"
            if not os.path.exists(states_path):
                continue
            sample_count = int(meta["sample_count"])
        except Exception:
            continue
        shards.append(
            ReplayShard(
                shard_format="raw",
                path=stem,
                sample_count=sample_count,
                mtime=os.path.getmtime(meta_path),
                states_path=states_path,
            )
        )

    shards.sort(key=lambda item: item.mtime)
    return shards


def _load_state_sample(shard: ReplayShard, sample_idx: int):
    try:
        if shard.shard_format == "raw":
            if not shard.states_path:
                return None
            states = np.load(shard.states_path, mmap_mode="r")
            return np.array(states[sample_idx], dtype=np.float32, copy=True)
        with np.load(shard.path) as data:
            return np.asarray(data["states"][sample_idx], dtype=np.float32)
    except Exception:
        return None


def _sample_board(rng: np.random.Generator, min_plies: int, max_plies: int):
    for _ in range(12):
        board = chess.Board()
        target_plies = int(rng.integers(max(0, min_plies), max(min_plies, max_plies) + 1))
        for _ in range(target_plies):
            if board.is_game_over(claim_draw=True):
                break
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = legal_moves[int(rng.integers(0, len(legal_moves)))]
            board.push(move)
        if not board.is_game_over(claim_draw=True):
            return board
    return chess.Board()
