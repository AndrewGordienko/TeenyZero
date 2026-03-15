import numpy as np

class MCTSNode:
    def __init__(self, priors: dict):
        """
        priors: dict of {action: probability}
        """
        self.children = {} # action -> MCTSNode
        
        # We use numpy arrays for these to make bulk math faster later
        self.actions = list(priors.keys())
        self.P = np.array([priors[a] for a in self.actions])
        self.N = np.zeros(len(self.actions))
        self.W = np.zeros(len(self.actions))
        self.Q = np.zeros(len(self.actions))

    def get_child(self, action):
        return self.children.get(action)

    def add_child(self, action, priors):
        self.children[action] = MCTSNode(priors)
        return self.children[action]