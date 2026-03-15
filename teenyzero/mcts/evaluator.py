import torch
import numpy as np
import chess
import uuid

class AlphaZeroEvaluator:
    def __init__(self, model=None, device="cpu", task_queue=None, result_dict=None):
        """
        Modular Evaluator for teenyzero.
        If task_queue/result_dict are provided, it operates in 'Batch Mode' for training/play.
        Otherwise, it runs local inference using the provided model.
        """
        self.model = model
        self.device = device
        self.task_queue = task_queue
        self.result_dict = result_dict
        self.batch_mode = task_queue is not None
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()

        # Build the direction lookup table for the 73 planes once at init
        self._dir_map = self._build_direction_map()

    def evaluate(self, board: chess.Board):
        """
        Entry point for MCTS. Returns (priors_dict, value_float).
        """
        # 1. Encode board state to (13, 8, 8)
        encoded_state = self.encode_board(board)

        if self.batch_mode:
            return self._evaluate_batched(encoded_state, board)
        else:
            if self.model is None:
                raise ValueError("Evaluator in local mode requires a model.")
            return self._evaluate_local(encoded_state, board)

    def _evaluate_local(self, encoded_state, board):
        """Standard PyTorch inference."""
        tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(tensor)
            
        probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        v = value.cpu().item()
        return self._mask_and_normalize(probs, board, v)

    def _evaluate_batched(self, encoded_state, board):
        """Dynamic batching via the inference_server."""
        task_id = str(uuid.uuid4())
        self.task_queue.put((task_id, encoded_state))
        
        # Block until the inference worker puts our result in the shared dict
        while task_id not in self.result_dict:
            continue
            
        probs, v = self.result_dict.pop(task_id)
        return self._mask_and_normalize(probs, board, v)

    def _mask_and_normalize(self, probs, board, v):
        """
        Applies legal move masking and re-normalizes the distribution.
        This is where 'Abductive Pruning' hooks will eventually live.
        """
        move_priors = {}
        total_prob = 0.0
        
        for move in board.legal_moves:
            idx = self.move_to_idx(move, board.turn)
            p = probs[idx]
            move_priors[move] = p
            total_prob += p

        if total_prob > 0:
            for move in move_priors:
                move_priors[move] /= total_prob
        else:
            # Fallback to uniform if the network is zeroed out
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                move_priors[move] = 1.0 / len(legal_moves)

        return move_priors, v

    # --- STATE ENCODING (13, 8, 8) ---

    def encode_board(self, board: chess.Board):
        """
        Encodes the board into 13 planes.
        0-5: White P, N, B, R, Q, K
        6-11: Black p, n, b, r, q, k
        12: To Move (All 1s for White, 0s for Black)
        """
        ds = np.zeros((13, 8, 8), dtype=np.float32)
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            plane = piece.piece_type - 1
            if piece.color == chess.BLACK:
                plane += 6
            ds[plane, rank, file] = 1.0
            
        if board.turn == chess.WHITE:
            ds[12, :, :] = 1.0
        return ds

    # --- ACTION ENCODING (4672 planes) ---

    def move_to_idx(self, move: chess.Move, turn: bool):
        """
        Maps a chess.Move to a unique index in the policy vector [0...4671].
        """
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Mirror for Black so the network only learns 'White' perspective
        if turn == chess.BLACK:
            from_sq = chess.square_mirror(from_sq)
            to_sq = chess.square_mirror(to_sq)

        f_rank, f_file = divmod(from_sq, 8)
        t_rank, t_file = divmod(to_sq, 8)
        dr, df = t_rank - f_rank, t_file - f_file

        plane_idx = 0
        
        # 1. Handle Under-promotions (9 planes)
        if move.promotion and move.promotion != chess.QUEEN:
            # Under-promotions are stored in planes 64-72
            # 3 directions (left-diag, straight, right-diag) * 3 pieces (N, B, R)
            piece_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
            direction = df + 1 # 0, 1, 2
            plane_idx = 64 + (piece_offset[move.promotion] * 3) + direction
        
        # 2. Handle Knight Moves (8 planes)
        elif (abs(dr), abs(df)) in [(1, 2), (2, 1)]:
            knight_moves = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]
            plane_idx = 56 + knight_moves.index((dr, df))
        
        # 3. Handle Queen-style moves (56 planes)
        else:
            # 8 directions * 7 squares max distance
            direction_key = (np.sign(dr), np.sign(df))
            dist = max(abs(dr), abs(df))
            dir_idx = self._dir_map[direction_key]
            plane_idx = (dir_idx * 7) + (dist - 1)

        return from_sq * 73 + plane_idx

    def _build_direction_map(self):
        """Maps unit vectors to 0-7 for the 8 compass directions."""
        return {
            (1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3,
            (1, 1): 4, (1, -1): 5, (-1, 1): 6, (-1, -1): 7
        }