from __future__ import annotations

import chess
import numpy as np
import torch

from teenyzero.alphazero.config import (
    AUX_PLANES,
    INPUT_HISTORY_LENGTH,
    INPUT_PLANES,
    PIECE_PLANES_PER_POSITION,
)


PAIR_FEATURE_NAMES = (
    "same_rank",
    "same_file",
    "same_diagonal",
    "same_ray",
    "knight_move",
    "king_move",
    "manhattan_distance",
    "chebyshev_distance",
    "delta_rank",
    "delta_file",
)
GEOMETRY_TARGET_NAMES = (
    "friendly_attack",
    "enemy_attack",
    "friendly_king_pressure",
    "enemy_king_pressure",
)
PIECE_SYMBOLS = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")


def orient_square(square: int, perspective: bool) -> int:
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def _oriented_row_col(square: int, perspective: bool) -> tuple[int, int]:
    oriented = orient_square(square, perspective)
    return chess.square_rank(oriented), chess.square_file(oriented)


def build_square_pair_feature_tensor(dtype=torch.float32) -> torch.Tensor:
    feature_count = len(PAIR_FEATURE_NAMES)
    features = np.zeros((64, 64, feature_count), dtype=np.float32)

    for source in chess.SQUARES:
        src_rank = chess.square_rank(source)
        src_file = chess.square_file(source)
        for target in chess.SQUARES:
            dst_rank = chess.square_rank(target)
            dst_file = chess.square_file(target)
            delta_rank = dst_rank - src_rank
            delta_file = dst_file - src_file
            abs_rank = abs(delta_rank)
            abs_file = abs(delta_file)
            manhattan = abs_rank + abs_file
            chebyshev = max(abs_rank, abs_file)

            features[source, target, 0] = 1.0 if src_rank == dst_rank else 0.0
            features[source, target, 1] = 1.0 if src_file == dst_file else 0.0
            features[source, target, 2] = 1.0 if abs_rank == abs_file else 0.0
            features[source, target, 3] = 1.0 if (
                source != target and (src_rank == dst_rank or src_file == dst_file or abs_rank == abs_file)
            ) else 0.0
            features[source, target, 4] = 1.0 if sorted((abs_rank, abs_file)) == [1, 2] else 0.0
            features[source, target, 5] = 1.0 if 0 < chebyshev <= 1 else 0.0
            features[source, target, 6] = manhattan / 14.0
            features[source, target, 7] = chebyshev / 7.0
            features[source, target, 8] = delta_rank / 7.0
            features[source, target, 9] = delta_file / 7.0

    return torch.tensor(features, dtype=dtype)


def encode_board_state(board: chess.Board) -> np.ndarray:
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    perspective = board.turn
    scratch = board.copy(stack=max(0, INPUT_HISTORY_LENGTH - 1))
    for history_idx in range(INPUT_HISTORY_LENGTH):
        offset = history_idx * PIECE_PLANES_PER_POSITION
        _fill_piece_planes(planes, offset, scratch, perspective)
        if not scratch.move_stack:
            break
        scratch.pop()

    aux_offset = INPUT_HISTORY_LENGTH * PIECE_PLANES_PER_POSITION
    _fill_aux_planes(planes, aux_offset, board, perspective)
    return planes


def _empty_map() -> np.ndarray:
    return np.zeros((8, 8), dtype=np.float32)


def _fill_attack_map(target: np.ndarray, board: chess.Board, color: bool, perspective: bool):
    for square in chess.SQUARES:
        if board.is_attacked_by(color, square):
            row, col = _oriented_row_col(square, perspective)
            target[row, col] = 1.0


def _fill_king_zone_pressure(target: np.ndarray, board: chess.Board, attacker_color: bool, perspective: bool):
    defender_king = board.king(not attacker_color)
    if defender_king is None:
        return

    zone_squares = {defender_king}
    king_rank = chess.square_rank(defender_king)
    king_file = chess.square_file(defender_king)
    for delta_rank in (-1, 0, 1):
        for delta_file in (-1, 0, 1):
            rank = king_rank + delta_rank
            file = king_file + delta_file
            if 0 <= rank < 8 and 0 <= file < 8:
                zone_squares.add(chess.square(file, rank))

    for square in zone_squares:
        attackers = len(board.attackers(attacker_color, square))
        if attackers <= 0:
            continue
        row, col = _oriented_row_col(square, perspective)
        target[row, col] = min(attackers, 4) / 4.0


def build_square_targets(board: chess.Board) -> dict[str, np.ndarray]:
    perspective = board.turn
    friendly_attack = _empty_map()
    enemy_attack = _empty_map()
    friendly_king_pressure = _empty_map()
    enemy_king_pressure = _empty_map()

    _fill_attack_map(friendly_attack, board, perspective, perspective)
    _fill_attack_map(enemy_attack, board, not perspective, perspective)
    _fill_king_zone_pressure(friendly_king_pressure, board, perspective, perspective)
    _fill_king_zone_pressure(enemy_king_pressure, board, not perspective, perspective)

    return {
        "friendly_attack": friendly_attack,
        "enemy_attack": enemy_attack,
        "friendly_king_pressure": friendly_king_pressure,
        "enemy_king_pressure": enemy_king_pressure,
    }


def build_square_target_tensor(board: chess.Board) -> np.ndarray:
    targets = build_square_targets(board)
    return np.stack(
        [
            targets["friendly_attack"],
            targets["enemy_attack"],
            targets["friendly_king_pressure"],
            targets["enemy_king_pressure"],
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def board_from_encoded_state(state) -> chess.Board:
    array = _state_array(state)
    board = chess.Board(None)
    board.clear_stack()
    board.turn = chess.WHITE

    active_planes = min(12, int(array.shape[0]))
    for plane_idx in range(active_planes):
        piece_type = _piece_type_for_plane(plane_idx)
        color = chess.WHITE if plane_idx < 6 else chess.BLACK
        positions = np.argwhere(array[plane_idx] > 0.5)
        for row, col in positions:
            square = chess.square(int(col), int(row))
            board.set_piece_at(square, chess.Piece(piece_type, color))

    return board


def build_square_target_tensor_from_state(state) -> np.ndarray:
    return build_square_target_tensor(board_from_encoded_state(state))


def state_to_piece_grid(state) -> list[list[str]]:
    board = board_from_encoded_state(state)
    grid = [["" for _ in range(8)] for _ in range(8)]
    for rank in range(8):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece is None:
                continue
            grid[rank][file] = piece.symbol()
    return grid


def _fill_piece_planes(planes: np.ndarray, offset: int, board: chess.Board, perspective: bool):
    friendly = board.occupied_co[perspective]
    piece_bitboards = (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
    )

    for piece_offset, piece_bits in enumerate(piece_bitboards):
        _write_piece_bitboard(
            planes,
            offset + piece_offset,
            piece_bits & friendly,
            perspective,
        )
        _write_piece_bitboard(
            planes,
            offset + piece_offset + 6,
            piece_bits & board.occupied_co[not perspective],
            perspective,
        )


def _fill_aux_planes(planes: np.ndarray, offset: int, board: chess.Board, perspective: bool):
    if AUX_PLANES < 8:
        raise ValueError(f"Expected at least 8 auxiliary planes, got {AUX_PLANES}")
    planes[offset, :, :] = 1.0
    planes[offset + 1, :, :] = 1.0 if board.has_kingside_castling_rights(perspective) else 0.0
    planes[offset + 2, :, :] = 1.0 if board.has_queenside_castling_rights(perspective) else 0.0
    planes[offset + 3, :, :] = 1.0 if board.has_kingside_castling_rights(not perspective) else 0.0
    planes[offset + 4, :, :] = 1.0 if board.has_queenside_castling_rights(not perspective) else 0.0
    planes[offset + 5, :, :] = min(board.halfmove_clock, 100) / 100.0
    planes[offset + 6, :, :] = min(board.fullmove_number, 200) / 200.0
    planes[offset + 7, :, :] = 1.0 if board.is_repetition(2) else 0.0


def _write_piece_bitboard(planes: np.ndarray, plane_idx: int, bitboard: int, perspective: bool):
    while bitboard:
        lsb = bitboard & -bitboard
        square = lsb.bit_length() - 1
        row, col = _oriented_row_col(square, perspective)
        planes[plane_idx, row, col] = 1.0
        bitboard ^= lsb


def _state_array(state) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    array = np.asarray(state, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    return array


def _piece_type_for_plane(plane_idx: int) -> int:
    piece_offset = int(plane_idx % 6)
    return (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    )[piece_offset]
