# Scripts

The runnable surface is intentionally flat now. Use the top-level scripts only.

- `scripts/benchmark_self_play.py`
  Benchmarks move latency and positions/sec for the current runtime/device/profile.
- `scripts/run_actors.py`
  Starts self-play workers plus the inference server.
- `scripts/run_arena.py`
  Runs checkpoint promotion, arena Elo tracking, and optional Stockfish anchors.
- `scripts/run_chess.py`
  Starts terminal self-play or the gameplay UI.
- `scripts/run_visualizers.py`
  Starts the full dashboard stack.
- `scripts/train.py`
  Runs the continuous trainer loop.

Common flags:

- `--device mps`
- `--device h200 --profile h200`
- `--profile mps`
- `--board-backend native`
- `--no-dashboard` on `run_actors.py`

Runtime state lives under `var/`:

- `var/models/` for live and archived checkpoints
- `var/data/` for replay shards and trainer/arena state
