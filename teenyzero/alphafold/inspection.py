from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import chess
import numpy as np
import torch

from teenyzero.alphafold.features import (
    board_from_encoded_state,
    build_square_target_tensor_from_state,
    build_square_targets_absolute,
    oriented_map_to_absolute,
    piece_grid_from_board,
    state_to_piece_grid,
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
        "board": board,
    }


def sample_geometry_payload(
    state,
    source: str,
    model=None,
    device: str = "cpu",
    top_k_connections: int = 8,
    reference_board: chess.Board | None = None,
):
    state_array = np.asarray(state, dtype=np.float32)
    if reference_board is not None:
        board = reference_board.copy(stack=False)
        payload = {
            "source": source,
            "board_fen": board.board_fen(),
            "piece_grid": piece_grid_from_board(board),
            "board_view": "white_bottom_absolute",
            "board_view_label": "White at bottom",
            "board_view_note": "True board colors from the random pretraining sample.",
            "targets": _absolute_display_targets(board),
            "map_info": _absolute_map_info(),
        }
    else:
        board = board_from_encoded_state(state_array)
        targets = build_square_target_tensor_from_state(state_array)
        payload = {
            "source": source,
            "board_fen": board.board_fen(),
            "piece_grid": state_to_piece_grid(state_array),
            "board_view": "side_to_move_bottom",
            "board_view_label": "Side to move at bottom",
            "board_view_note": "Replay states are stored in normalized perspective, so colors indicate bottom side versus top side here.",
            "targets": {
                "bottom_attack": targets[0].tolist(),
                "top_attack": targets[1].tolist(),
                "bottom_king_pressure": targets[2].tolist(),
                "top_king_pressure": targets[3].tolist(),
            },
            "map_info": _relative_map_info(),
        }

    if model is None:
        return payload

    tensor = torch.from_numpy(state_array).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        _, _, aux_outputs, attention = model(tensor, return_aux=True, return_attention=True)

    if reference_board is not None:
        predicted_maps = _absolute_display_predictions(aux_outputs, reference_board.turn)
    else:
        predicted_maps = {
            "bottom_attack": torch.sigmoid(aux_outputs["friendly_attack"])[0].detach().cpu().numpy().astype(np.float32).tolist(),
            "top_attack": torch.sigmoid(aux_outputs["enemy_attack"])[0].detach().cpu().numpy().astype(np.float32).tolist(),
            "bottom_king_pressure": torch.sigmoid(aux_outputs["friendly_king_pressure"])[0].detach().cpu().numpy().astype(np.float32).tolist(),
            "top_king_pressure": torch.sigmoid(aux_outputs["enemy_king_pressure"])[0].detach().cpu().numpy().astype(np.float32).tolist(),
        }
    payload["predictions"] = predicted_maps
    if attention is not None:
        attention_matrix = attention[0].detach().cpu().numpy().astype(np.float32)
        payload["attention_matrix"] = _display_attention_matrix(
            attention_matrix,
            perspective=board.turn if reference_board is not None else None,
        ).tolist()
        payload["top_connections"] = top_attention_connections(
            attention_matrix,
            limit=top_k_connections,
            perspective=board.turn if reference_board is not None else None,
        )
    else:
        payload["attention_matrix"] = None
        payload["top_connections"] = []
    return payload


def top_attention_connections(attention_matrix: np.ndarray, limit: int = 8, perspective: bool | None = None):
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
        if perspective == chess.BLACK:
            source = chess.square_mirror(source)
            target = chess.square_mirror(target)
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


def _absolute_display_targets(board: chess.Board) -> dict[str, list[list[float]]]:
    targets = build_square_targets_absolute(board)
    return {key: value.tolist() for key, value in targets.items()}


def _absolute_display_predictions(aux_outputs, perspective: bool) -> dict[str, list[list[float]]]:
    friendly_attack = torch.sigmoid(aux_outputs["friendly_attack"])[0].detach().cpu().numpy().astype(np.float32)
    enemy_attack = torch.sigmoid(aux_outputs["enemy_attack"])[0].detach().cpu().numpy().astype(np.float32)
    friendly_pressure = torch.sigmoid(aux_outputs["friendly_king_pressure"])[0].detach().cpu().numpy().astype(np.float32)
    enemy_pressure = torch.sigmoid(aux_outputs["enemy_king_pressure"])[0].detach().cpu().numpy().astype(np.float32)

    if perspective == chess.WHITE:
        white_attack, black_attack = friendly_attack, enemy_attack
        white_pressure, black_pressure = friendly_pressure, enemy_pressure
    else:
        white_attack, black_attack = enemy_attack, friendly_attack
        white_pressure, black_pressure = enemy_pressure, friendly_pressure

    return {
        "white_attack": oriented_map_to_absolute(white_attack, perspective).tolist(),
        "black_attack": oriented_map_to_absolute(black_attack, perspective).tolist(),
        "white_king_pressure": oriented_map_to_absolute(white_pressure, perspective).tolist(),
        "black_king_pressure": oriented_map_to_absolute(black_pressure, perspective).tolist(),
    }


def _absolute_map_info():
    return {
        "white_attack": {
            "title": "White Control / Attack",
            "note": "Squares white pieces currently attack or defend. Higher values mean more or stronger white attackers on that square.",
        },
        "black_attack": {
            "title": "Black Control / Attack",
            "note": "Squares black pieces currently attack or defend. Friendly occupied squares can still light up because they are defended.",
        },
        "white_king_pressure": {
            "title": "White King Pressure",
            "note": "White pressure on the black king zone, normalized to keep this map smooth instead of binary.",
        },
        "black_king_pressure": {
            "title": "Black King Pressure",
            "note": "Black pressure on the white king zone, normalized to keep this map smooth instead of binary.",
        },
    }


def _relative_map_info():
    return {
        "bottom_attack": {
            "title": "Bottom Side Control / Attack",
            "note": "Squares the bottom-side pieces currently attack or defend. This is board control, not move preference.",
        },
        "top_attack": {
            "title": "Top Side Control / Attack",
            "note": "Squares the top-side pieces currently attack or defend. Occupied friendly squares can still light up because they are defended.",
        },
        "bottom_king_pressure": {
            "title": "Bottom Side King Pressure",
            "note": "Pressure created by the bottom side on the top-side king zone.",
        },
        "top_king_pressure": {
            "title": "Top Side King Pressure",
            "note": "Pressure created by the top side on the bottom-side king zone.",
        },
    }


def _display_attention_matrix(attention_matrix: np.ndarray, perspective: bool | None = None) -> np.ndarray:
    matrix = np.asarray(attention_matrix, dtype=np.float32)
    if matrix.shape != (64, 64) or perspective is None or perspective == chess.WHITE:
        return np.array(matrix, dtype=np.float32, copy=True)

    remapped = np.zeros_like(matrix)
    for src in range(64):
        abs_src = chess.square_mirror(src)
        for dst in range(64):
            abs_dst = chess.square_mirror(dst)
            remapped[abs_src, abs_dst] = matrix[src, dst]
    return remapped
