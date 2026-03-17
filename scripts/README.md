# Scripts

Top-level scripts remain as compatibility entrypoints. The grouped entrypoints below are the preferred surface now.

## UI

- `scripts/ui/run_dashboard.py`
  Starts the full local dashboard stack: self-play actors, trainer, arena, and Flask UI.
- `scripts/ui/run_chess.py`
  Starts the gameplay UI / terminal play entrypoint against the current network.

## Training

- `scripts/training/run_loop.py`
  Runs the continuous trainer loop directly.
- `scripts/training/run_trainer.py`
  Thin compatibility wrapper for the trainer loop.

## Self-Play

- `scripts/selfplay/run_factory.py`
  Starts the actor factory plus inference server used for self-play data generation.

## Evaluation

- `scripts/eval/run_arena.py`
  Runs the checkpoint promotion gate, arena Elo loop, and optional Stockfish anchors.
- `scripts/eval/benchmark_self_play.py`
  Benchmarks move latency and positions/sec so profile changes can be measured before and after tuning.

## Profiles

- `TEENYZERO_PROFILE=local`
  Mac-friendly defaults with moderate model/search settings.
- `TEENYZERO_PROFILE=h100`
  Hopper-oriented settings: larger network, BF16, compile, larger replay window, deeper search.
- `TEENYZERO_PROFILE=h200`
  More aggressive Hopper profile with larger replay and evaluation budgets.

The runtime profile is also auto-detected on CUDA if the GPU name contains `H100` or `H200`.

## External Benchmarks

- `TEENYZERO_STOCKFISH_PATH=/abs/path/to/stockfish`
  Enables anchored arena matches against calibrated Stockfish strength levels for human-readable progress tracking.
