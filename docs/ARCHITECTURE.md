# Architecture

## Core

- `teenyzero/mcts/search.py`
  Array-backed MCTS edge stats and batched leaf evaluation.
- `teenyzero/mcts/evaluator.py`
  Board encoding, legal move indexing, policy masking, and inference dispatch.
- `teenyzero/alphazero/servers/inference.py`
  Dynamic batching server for self-play workers.
- `teenyzero/alphazero/search_session.py`
  Keeps and advances the root across played moves for UI and arena flows.

## Runtime

- `teenyzero/alphazero/runtime.py`
  Runtime profiles and device/profile selection.
- `teenyzero/runtime_bootstrap.py`
  CLI flag bootstrap that sets runtime env before config import.
- `teenyzero/paths.py`
  Shared runtime paths for live checkpoints, replay, state, locks, and archive storage.

## Native Speedups

- `native/speedups.cpp`
  Optional C++ extension for move signatures and policy-index mapping.
- `teenyzero/alphazero/backend.py`
  Chooses between Python and native speedup helpers.

This is not yet a full native chess engine. Board state and legal move generation still come from `python-chess`.

## Visualizers

The visualizer tree is flat now under `teenyzero/visualizers/`:

- `hub/`
- `gameplay/`
- `training_status/`
- `arena_status/`
- `cluster_monitor/`
- `assets/`
- `common/`
