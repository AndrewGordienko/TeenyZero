class MCTSNode:
    __slots__ = ("children", "P", "N", "W", "total_n", "total_w", "dirichlet_applied")

    def __init__(self, priors: dict):
        self.children = {}
        self.P = {move: float(prob) for move, prob in priors.items()}
        self.N = {move: 0 for move in priors}
        self.W = {move: 0.0 for move in priors}
        self.total_n = 0.0
        self.total_w = 0.0
        self.dirichlet_applied = False

    def get_child(self, move):
        return self.children.get(move)

    def add_child(self, move, priors):
        child = MCTSNode(priors)
        self.children[move] = child
        return child
