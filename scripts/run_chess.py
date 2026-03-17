import argparse
import chess
import torch
import numpy as np
import sys
import os

# Package imports
from teenyzero.alphazero.checkpoints import build_model, load_checkpoint
from teenyzero.alphazero.runtime import get_runtime_profile
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.mcts.search import MCTS


PROFILE = get_runtime_profile()

def maybe_reload_model(model, evaluator, model_path, last_mtime):
    if not os.path.exists(model_path):
        return last_mtime

    current_mtime = os.path.getmtime(model_path)
    if last_mtime is not None and current_mtime <= last_mtime:
        return last_mtime

    load_result = load_checkpoint(model, model_path, map_location=evaluator.device, allow_partial=True)
    if load_result["loaded"]:
        model.eval()
        evaluator.clear_cache()
    return current_mtime


def run_self_play(engine, model, evaluator, model_path):
    """The terminal-based self-play loop."""
    board = chess.Board()
    last_mtime = None
    print("\n" + "="*30)
    print("[*] Starting Self-Play Mode")
    print("="*30 + "\n")
    print(board)

    while not board.is_game_over():
        last_mtime = maybe_reload_model(model, evaluator, model_path, last_mtime)
        current_turn = "White" if board.turn == chess.WHITE else "Black"
        best_move, _, root_node = engine.search(board)
        
        # Calculate expected value from root
        expected_value = np.mean(root_node.W) if np.sum(root_node.N) > 0 else 0.0
        
        print(f"\n[{current_turn}] Move: {best_move} | Value: {expected_value:.3f}")
        board.push(best_move)
        print(board)
        print("-" * 20)

    print(f"\n[*] Game Over. Result: {board.result()}")

def main():
    parser = argparse.ArgumentParser(description="teenyzero: AlphaZero Research Project")
    parser.add_argument("--mode", choices=["play", "visualize"], default="visualize", 
                        help="Choose between terminal self-play or the web dashboard.")
    args = parser.parse_args()

    # 1. Hardware Setup
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Initializing teenyzero on: {device}")
    print(f"[*] Active runtime profile: {PROFILE.name}")
    model_path = "models/latest_model.pth"

    # 2. Brain Setup (Tuned for internship research)
    model = build_model()
    if os.path.exists(model_path):
        load_checkpoint(model, model_path, map_location=device, allow_partial=True)
    
    evaluator = AlphaZeroEvaluator(model=model, device=device)
    
    # Focused parameters to reduce shuffling
    mcts_params = {
        'SIMULATIONS': max(160, PROFILE.arena_simulations),
        'C_PUCT': 1.1,         # Lower exploration to focus on 'better' moves
        'VIRTUAL_LOSS': 1.0,
        'PARALLEL_THREADS': 4,
        'FPU_REDUCTION': 0.5,  # Higher penalty for unexplored paths
        'LEAF_BATCH_SIZE': max(8, PROFILE.selfplay_leaf_batch_size // 2),
    }
    engine = MCTS(evaluator=evaluator, params=mcts_params)

    # 3. Mode Execution
    if args.mode == "play":
        run_self_play(engine, model, evaluator, model_path)
    
    elif args.mode == "visualize":
        print("[*] Injecting engine into Visualizer...")
        import teenyzero.visualizers.app as vis_app
        vis_app.engine = engine 
        
        print("[*] Launching Lab UI on http://localhost:5001")
        vis_app.app.run(debug=False, port=5001, host='0.0.0.0')

if __name__ == "__main__":
    main()
