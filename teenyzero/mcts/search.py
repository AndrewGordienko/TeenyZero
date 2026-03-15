import math
import numpy as np
import threading
import concurrent.futures
from .node import MCTSNode

class MCTS:
    def __init__(self, evaluator, params=None):
        self.evaluator = evaluator
        self.params = params or {
            'SIMULATIONS': 400,
            'C_PUCT': 1.1,
            'VIRTUAL_LOSS': 1.0,
            'PARALLEL_THREADS': 4,
            'FPU_REDUCTION': 0.5
        }
        self.tree_lock = threading.Lock()

    def search(self, state, is_training=False):
        """
        state: A chess.Board instance. 
        MCTS performs search and returns the best move and root node.
        """
        # 1. Create root node with network priors
        priors, _ = self.evaluator.evaluate(state)
        root = MCTSNode(priors)

        # 2. Run Simulations in Parallel
        # Threads use copies of the board to remain thread-safe
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.params['PARALLEL_THREADS']) as ex:
            futures = [ex.submit(self._run_simulation, state, root) 
                       for _ in range(self.params['SIMULATIONS'])]
            concurrent.futures.wait(futures)

        # 3. Choose the move with the highest visit count (N)
        best_idx = np.argmax(root.N)
        best_move = root.actions[best_idx]
        
        return best_move, root

    def _run_simulation(self, root_state, root_node):
        """
        A single simulation: Selection, Expansion, Evaluation, and Backprop.
        """
        node = root_node
        path = []
        
        # --- FIX: STATE CLONING ---
        # We work on a copy so we don't mutate the board used by the UI or other threads.
        state = root_state.copy() 

        # --- SELECTION ---
        with self.tree_lock:
            while True:
                action_idx = self._select_child_idx(node)
                action = node.actions[action_idx]
                path.append((node, action_idx))
                
                # Apply Virtual Loss to discourage other threads from 
                # following this exact path simultaneously
                node.N[action_idx] += self.params['VIRTUAL_LOSS']
                node.W[action_idx] -= self.params['VIRTUAL_LOSS']
                
                state.push(action)
                
                if action not in node.children:
                    break
                node = node.children[action]

        # --- EXPANSION & EVALUATION ---
        if not state.is_game_over():
            priors, value = self.evaluator.evaluate(state)
            with self.tree_lock:
                if action not in node.children:
                    node.add_child(action, priors)
        else:
            value = self._get_terminal_value(state)

        # --- BACKPROPAGATION ---
        with self.tree_lock:
            for n, idx in reversed(path):
                # Remove virtual loss and add the real evaluation result
                n.N[idx] = n.N[idx] - self.params['VIRTUAL_LOSS'] + 1
                n.W[idx] = n.W[idx] + self.params['VIRTUAL_LOSS'] + value
                n.Q[idx] = n.W[idx] / n.N[idx]
                value = -value # Zero-sum perspective flip

    def _select_child_idx(self, node):
        """
        PUCT Selection logic.
        """
        total_n = np.sum(node.N)
        total_n_sqrt = math.sqrt(total_n + 1)
        
        # Vectorized UCB calculation for speed
        u_scores = self.params['C_PUCT'] * node.P * total_n_sqrt / (1 + node.N)
        full_scores = node.Q + u_scores
        
        return np.argmax(full_scores)

    def _get_terminal_value(self, state):
        """
        Converts chess game outcome to a scalar [-1, 0, 1].
        """
        res = state.result()
        if res == "1-0": return 1.0
        if res == "0-1": return -1.0
        return 0.0