from __future__ import annotations


class SearchSession:
    def __init__(self, engine):
        self.engine = engine
        self.root = None
        self._seen_moves = ()

    def reset(self):
        self.root = None
        self._seen_moves = ()

    def sync_to_board(self, board):
        move_stack = tuple(board.move_stack)
        prefix_len = min(len(move_stack), len(self._seen_moves))

        if len(move_stack) < len(self._seen_moves) or self._seen_moves[:prefix_len] != move_stack[:prefix_len]:
            self.reset()
            prefix_len = 0

        for move in move_stack[prefix_len:]:
            self.root = self.engine.advance_root(self.root, move)

        self._seen_moves = move_stack
        return self.root

    def search(self, board, **kwargs):
        self.sync_to_board(board)
        best_move, pi_dist, root = self.engine.search(board, root=self.root, **kwargs)
        self.root = root
        return best_move, pi_dist, root
