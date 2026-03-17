import numpy as np


class MCTSNode:
    __slots__ = (
        "moves",
        "move_to_index",
        "priors",
        "visits",
        "value_sums",
        "children",
        "total_n",
        "total_w",
        "dirichlet_applied",
    )

    def __init__(self, priors):
        if hasattr(priors, "moves") and hasattr(priors, "probs"):
            moves = tuple(priors.moves)
            probs = np.asarray(priors.probs, dtype=np.float32)
        elif isinstance(priors, dict):
            moves = tuple(priors.keys())
            probs = np.fromiter((float(prob) for prob in priors.values()), dtype=np.float32, count=len(moves))
        else:
            moves, raw_probs = priors
            moves = tuple(moves)
            probs = np.asarray(raw_probs, dtype=np.float32)

        self.moves = moves
        self.move_to_index = {move: idx for idx, move in enumerate(self.moves)}
        self.priors = np.array(probs, dtype=np.float32, copy=True, order="C")
        self.visits = np.zeros(len(self.moves), dtype=np.float32)
        self.value_sums = np.zeros(len(self.moves), dtype=np.float32)
        self.children = [None] * len(self.moves)
        self.total_n = 0.0
        self.total_w = 0.0
        self.dirichlet_applied = False

    def get_child(self, move):
        idx = self.move_to_index.get(move)
        if idx is None:
            return None
        return self.children[idx]

    def add_child(self, move, priors):
        idx = self.move_to_index.get(move)
        if idx is None:
            return None
        child = MCTSNode(priors)
        self.children[idx] = child
        return child
