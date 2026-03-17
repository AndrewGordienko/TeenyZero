import chess
import time
import torch
import numpy as np

from teenyzero.alphazero.config import (
    AUX_PLANES,
    INPUT_HISTORY_LENGTH,
    INPUT_PLANES,
    PIECE_PLANES_PER_POSITION,
)
from teenyzero.alphazero.runtime import get_runtime_profile


PROFILE = get_runtime_profile()


class AlphaZeroEvaluator:
    def __init__(
        self,
        model=None,
        device="cpu",
        task_queue=None,
        response_queue=None,
        worker_id=None,
        use_cache=True,
    ):
        """
        If task_queue/response_queue are provided, evaluator runs in batch mode.
        Otherwise, it runs local inference using the given model.
        """
        self.model = model
        self.device = device
        self.task_queue = task_queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        self.batch_mode = task_queue is not None and response_queue is not None
        self.use_cache = use_cache
        self.history_length = INPUT_HISTORY_LENGTH
        self.requires_history = self.history_length > 1
        self.cache = {}
        self.encoded_cache = {}
        self.legal_move_cache = {}
        self.pending_results = {}
        self.request_counter = 0
        if self.device == "cuda" and PROFILE.inference_precision == "bf16":
            self.inference_dtype = torch.bfloat16
        else:
            self.inference_dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
        self.use_channels_last = device in {"mps", "cuda"}
        if self.device == "cuda":
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        if self.model is not None:
            self.model.to(device=self.device, dtype=self.inference_dtype)
            if self.use_channels_last:
                self.model.to(memory_format=torch.channels_last)
            self.model.eval()

        self._dir_map = self._build_direction_map()
        self.profile = self._empty_profile()

    def clear_cache(self):
        self.cache.clear()
        self.encoded_cache.clear()
        self.legal_move_cache.clear()
        self.pending_results.clear()

    def reset_profile(self):
        self.profile = self._empty_profile()

    def snapshot_profile(self):
        return {
            "encode_ms": float(self.profile["encode_ms"]),
            "mask_ms": float(self.profile["mask_ms"]),
            "inference_wait_ms": float(self.profile["inference_wait_ms"]),
            "inference_forward_ms": float(self.profile["inference_forward_ms"]),
            "cache_hits": int(self.profile["cache_hits"]),
            "cache_misses": int(self.profile["cache_misses"]),
            "positions_evaluated": int(self.profile["positions_evaluated"]),
            "single_requests": int(self.profile["single_requests"]),
            "batch_requests": int(self.profile["batch_requests"]),
        }

    def _next_task_id(self):
        task_id = self.request_counter
        self.request_counter += 1
        return task_id

    def evaluate(self, board: chess.Board):
        """
        Returns:
            move_priors: dict[chess.Move, float]
            value: float in [-1, 1]
        """
        key = None
        if self.use_cache:
            key = self._cache_key(board)
            cached = self.cache.get(key)
            if cached is not None:
                self.profile["cache_hits"] += 1
                return cached
            self.profile["cache_misses"] += 1

        encode_start = time.perf_counter()
        encoded = self._encode_cached(board)
        self.profile["encode_ms"] += (time.perf_counter() - encode_start) * 1000.0
        self.profile["positions_evaluated"] += 1

        if self.batch_mode:
            result = self._evaluate_batched(encoded, board)
        else:
            if self.model is None:
                raise ValueError("Evaluator in local mode requires a model.")
            result = self._evaluate_local(encoded, board)

        if self.use_cache and key is not None:
            self.cache[key] = result

        return result

    def evaluate_many(self, boards):
        """
        Batched evaluation for a list of boards.
        Returns:
            list of (move_priors, value)
        """
        if not boards:
            return []

        results = [None] * len(boards)
        uncached_indices = []
        uncached_boards = []
        uncached_encoded = []

        for i, board in enumerate(boards):
            if self.use_cache:
                key = self._cache_key(board)
                cached = self.cache.get(key)
                if cached is not None:
                    self.profile["cache_hits"] += 1
                    results[i] = cached
                    continue
                self.profile["cache_misses"] += 1

            uncached_indices.append(i)
            uncached_boards.append(board)
            encode_start = time.perf_counter()
            uncached_encoded.append(self._encode_cached(board))
            self.profile["encode_ms"] += (time.perf_counter() - encode_start) * 1000.0

        if uncached_boards:
            self.profile["positions_evaluated"] += len(uncached_boards)
            if self.batch_mode:
                batch_results = self._evaluate_many_batched(uncached_encoded, uncached_boards)
            else:
                if self.model is None:
                    raise ValueError("Evaluator in local mode requires a model.")
                batch_results = self._evaluate_many_local(uncached_encoded, uncached_boards)

            for idx, board, result in zip(uncached_indices, uncached_boards, batch_results):
                results[idx] = result
                if self.use_cache:
                    self.cache[self._cache_key(board)] = result

        return results

    def _evaluate_local(self, encoded_state, board: chess.Board):
        tensor = torch.from_numpy(encoded_state).unsqueeze(0).to(
            device=self.device,
            dtype=self.inference_dtype,
        )
        if self.use_channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)

        forward_start = time.perf_counter()
        with torch.inference_mode():
            policy_logits, value = self.model(tensor)
        self.profile["inference_forward_ms"] += (time.perf_counter() - forward_start) * 1000.0
        self.profile["single_requests"] += 1

        logits = policy_logits.detach().cpu().numpy()[0]
        v = float(value.item())
        return self._mask_and_normalize_logits(logits, board, v)

    def _evaluate_many_local(self, encoded_states, boards):
        tensor = torch.from_numpy(np.asarray(encoded_states, dtype=np.float32)).to(
            device=self.device,
            dtype=self.inference_dtype,
        )
        if self.use_channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)

        forward_start = time.perf_counter()
        with torch.inference_mode():
            policy_logits, values = self.model(tensor)
        self.profile["inference_forward_ms"] += (time.perf_counter() - forward_start) * 1000.0
        self.profile["batch_requests"] += 1

        logits_batch = policy_logits.detach().cpu().numpy()
        values_batch = values.detach().cpu().numpy().reshape(-1)

        results = []
        for logits, value, board in zip(logits_batch, values_batch, boards):
            results.append(self._mask_and_normalize_logits(logits, board, float(value)))
        return results

    def _evaluate_batched(self, encoded_state, board: chess.Board):
        task_id = self._next_task_id()
        self.task_queue.put((task_id, encoded_state, self.worker_id, False))

        wait_start = time.perf_counter()
        while True:
            response = self.response_queue.get()
            if len(response) == 5:
                returned_task_id, logits, v, is_batch, meta = response
            else:
                returned_task_id, logits, v, is_batch = response
                meta = {}
            if is_batch:
                self.pending_results[returned_task_id] = (logits, v, is_batch, meta)
                continue

            if returned_task_id == task_id:
                self.profile["inference_wait_ms"] += (time.perf_counter() - wait_start) * 1000.0
                self.profile["inference_forward_ms"] += float(meta.get("forward_ms", 0.0))
                self.profile["single_requests"] += 1
                return self._mask_and_normalize_logits(logits, board, float(v))

            self.pending_results[returned_task_id] = (logits, v, is_batch, meta)

    def _evaluate_many_batched(self, encoded_states, boards):
        task_id = self._next_task_id()
        encoded_batch = np.asarray(encoded_states, dtype=np.float32)
        self.task_queue.put((task_id, encoded_batch, self.worker_id, True))

        wait_start = time.perf_counter()
        if task_id in self.pending_results:
            logits_batch, values_batch, is_batch, meta = self.pending_results.pop(task_id)
            if not is_batch:
                raise ValueError("Expected batched inference response but got single response.")
        else:
            while True:
                response = self.response_queue.get()
                if len(response) == 5:
                    returned_task_id, logits_batch, values_batch, is_batch, meta = response
                else:
                    returned_task_id, logits_batch, values_batch, is_batch = response
                    meta = {}

                if returned_task_id == task_id:
                    if not is_batch:
                        raise ValueError("Expected batched inference response but got single response.")
                    break

                self.pending_results[returned_task_id] = (logits_batch, values_batch, is_batch, meta)

        self.profile["inference_wait_ms"] += (time.perf_counter() - wait_start) * 1000.0
        self.profile["inference_forward_ms"] += float(meta.get("forward_ms", 0.0))
        self.profile["batch_requests"] += 1

        results = []
        for logits, value, board in zip(logits_batch, values_batch, boards):
            results.append(self._mask_and_normalize_logits(logits, board, float(value)))
        return results

    def _mask_and_normalize_logits(self, logits, board: chess.Board, v: float):
        """
        Normalize only over legal moves, using logits directly.
        """
        mask_start = time.perf_counter()
        legal_moves, legal_indices = self._get_legal_moves_and_indices(board)
        if not legal_moves:
            self.profile["mask_ms"] += (time.perf_counter() - mask_start) * 1000.0
            return {}, float(np.clip(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0))

        legal_logits = np.asarray(logits[legal_indices], dtype=np.float32)
        safe_value = float(np.clip(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0))
        if not np.isfinite(legal_logits).all():
            uniform = 1.0 / len(legal_moves)
            self.profile["mask_ms"] += (time.perf_counter() - mask_start) * 1000.0
            return {move: uniform for move in legal_moves}, safe_value

        max_logit = float(np.max(legal_logits))
        exp_vals = np.exp(legal_logits - max_logit).astype(np.float32, copy=False)
        total = float(np.sum(exp_vals))

        if not np.isfinite(total) or total <= 0.0:
            uniform = 1.0 / len(legal_moves)
            self.profile["mask_ms"] += (time.perf_counter() - mask_start) * 1000.0
            return {move: uniform for move in legal_moves}, safe_value

        inv_total = 1.0 / total
        self.profile["mask_ms"] += (time.perf_counter() - mask_start) * 1000.0
        return {
            move: float(prob)
            for move, prob in zip(legal_moves, exp_vals * inv_total)
        }, safe_value

    def encode_board(self, board: chess.Board):
        planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
        perspective = board.turn

        history_boards = []
        scratch = board.copy(stack=True)
        history_boards.append(scratch.copy(stack=False))
        for _ in range(1, self.history_length):
            if not scratch.move_stack:
                break
            scratch.pop()
            history_boards.append(scratch.copy(stack=False))

        for history_idx, history_board in enumerate(history_boards):
            offset = history_idx * PIECE_PLANES_PER_POSITION
            self._fill_piece_planes(planes, offset, history_board, perspective)

        aux_offset = self.history_length * PIECE_PLANES_PER_POSITION
        self._fill_aux_planes(planes, aux_offset, board, perspective)
        return planes

    def _fill_piece_planes(self, planes, offset, board: chess.Board, perspective: bool):
        for square, piece in board.piece_map().items():
            oriented_square = self._orient_square(square, perspective)
            rank = chess.square_rank(oriented_square)
            file = chess.square_file(oriented_square)
            plane = piece.piece_type - 1
            if piece.color != perspective:
                plane += 6
            planes[offset + plane, rank, file] = 1.0

    def _fill_aux_planes(self, planes, offset, board: chess.Board, perspective: bool):
        planes[offset, :, :] = 1.0
        planes[offset + 1, :, :] = 1.0 if board.has_kingside_castling_rights(perspective) else 0.0
        planes[offset + 2, :, :] = 1.0 if board.has_queenside_castling_rights(perspective) else 0.0
        planes[offset + 3, :, :] = 1.0 if board.has_kingside_castling_rights(not perspective) else 0.0
        planes[offset + 4, :, :] = 1.0 if board.has_queenside_castling_rights(not perspective) else 0.0
        planes[offset + 5, :, :] = min(board.halfmove_clock, 100) / 100.0
        planes[offset + 6, :, :] = min(board.fullmove_number, 200) / 200.0
        repetition = 1.0 if board.is_repetition(2) else 0.0
        planes[offset + 7, :, :] = repetition

    def _orient_square(self, square, perspective: bool):
        return square if perspective == chess.WHITE else chess.square_mirror(square)

    def move_to_idx(self, move: chess.Move, turn: bool):
        """
        Maps a move into AlphaZero-style 8x8x73 policy indexing.
        """
        from_sq = move.from_square
        to_sq = move.to_square

        if turn == chess.BLACK:
            from_sq = chess.square_mirror(from_sq)
            to_sq = chess.square_mirror(to_sq)

        f_rank, f_file = divmod(from_sq, 8)
        t_rank, t_file = divmod(to_sq, 8)
        dr = t_rank - f_rank
        df = t_file - f_file

        if move.promotion and move.promotion != chess.QUEEN:
            piece_offset = {
                chess.KNIGHT: 0,
                chess.BISHOP: 1,
                chess.ROOK: 2,
            }
            direction = df + 1
            plane_idx = 64 + piece_offset[move.promotion] * 3 + direction
            return from_sq * 73 + plane_idx

        knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1),
        ]
        if (dr, df) in knight_moves:
            plane_idx = 56 + knight_moves.index((dr, df))
            return from_sq * 73 + plane_idx

        direction_key = (int(np.sign(dr)), int(np.sign(df)))
        dist = max(abs(dr), abs(df))
        dir_idx = self._dir_map[direction_key]
        plane_idx = dir_idx * 7 + (dist - 1)
        return from_sq * 73 + plane_idx

    def _build_direction_map(self):
        return {
            (1, 0): 0,
            (-1, 0): 1,
            (0, 1): 2,
            (0, -1): 3,
            (1, 1): 4,
            (1, -1): 5,
            (-1, 1): 6,
            (-1, -1): 7,
        }

    def _get_legal_moves_and_indices(self, board: chess.Board):
        key = self._position_key(board)
        cached = self.legal_move_cache.get(key)
        if cached is not None:
            return cached

        legal_moves = tuple(board.legal_moves)
        legal_indices = np.fromiter(
            (self.move_to_idx(move, board.turn) for move in legal_moves),
            dtype=np.int32,
            count=len(legal_moves),
        )
        cached = (legal_moves, legal_indices)
        self.legal_move_cache[key] = cached
        return cached

    def _encode_cached(self, board: chess.Board):
        key = self._position_key(board)
        cached = self.encoded_cache.get(key)
        if cached is not None:
            return cached

        encoded = self.encode_board(board)
        self.encoded_cache[key] = encoded
        return encoded

    def _position_key(self, board: chess.Board):
        ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
        history = tuple(move.uci() for move in board.move_stack[-(self.history_length - 1):])
        return (
            board.board_fen(),
            board.turn,
            board.castling_rights,
            ep,
            board.halfmove_clock,
            history,
        )

    def _cache_key(self, board: chess.Board):
        return self._position_key(board) + (board.fullmove_number,)

    def _empty_profile(self):
        return {
            "encode_ms": 0.0,
            "mask_ms": 0.0,
            "inference_wait_ms": 0.0,
            "inference_forward_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "positions_evaluated": 0,
            "single_requests": 0,
            "batch_requests": 0,
        }
