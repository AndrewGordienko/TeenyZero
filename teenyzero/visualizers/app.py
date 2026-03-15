import os
import numpy as np
import chess
from flask import Flask, request, jsonify, render_template
from teenyzero.mcts.search import MCTS
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.alphazero.model import AlphaNet

app = Flask(__name__)

# Fallback engine (will be overwritten if launched via run.py)
model = AlphaNet() 
evaluator = AlphaZeroEvaluator(model=model)
engine = MCTS(evaluator=evaluator, params={'SIMULATIONS': 400, 'PARALLEL_THREADS': 4})

board = chess.Board()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/move", methods=["POST"])
def move():
    global board
    data = request.json
    uci = data.get("uci")
    
    try:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            # 1. Human Move
            board.push(move)
            
            # 2. Engine Response
            if not board.is_game_over():
                engine_move, root = engine.search(board)
                # Check legality before pushing to avoid AssertionError
                if engine_move in board.legal_moves:
                    board.push(engine_move)
                
                # Calculate win prob: Map Q [-1, 1] to [0, 100]
                win_prob = (np.mean(root.W) + 1) / 2 if np.sum(root.N) > 0 else 0.5
                
                return jsonify({
                    "ok": True,
                    "fen": board.fen(),
                    "engine_move": engine_move.uci(),
                    "win_prob": round(float(win_prob) * 100, 1)
                })
            
            return jsonify({"ok": True, "fen": board.fen()})
        return jsonify({"ok": False, "error": "Illegal move"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/reset", methods=["POST"])
def reset():
    global board
    data = request.json or {}
    side = data.get("side", "w")
    
    # Strictly reset to a fresh board instance
    board = chess.Board()
    
    # If playing as Black, Engine makes the first move for White
    if side == "b":
        engine_move, root = engine.search(board)
        if engine_move in board.legal_moves:
            board.push(engine_move)
            win_prob = (np.mean(root.W) + 1) / 2 if np.sum(root.N) > 0 else 0.5
            return jsonify({
                "ok": True, 
                "fen": board.fen(), 
                "engine_move": engine_move.uci(),
                "win_prob": round(float(win_prob) * 100, 1)
            })
        
    return jsonify({"ok": True, "fen": board.fen(), "win_prob": 50.0})

if __name__ == "__main__":
    app.run(debug=True, port=5001)