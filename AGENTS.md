# AGENTS

## Repo Shape

- Keep the runnable command surface flat under `scripts/`.
- Keep runtime-generated state out of the source package. Use `teenyzero/paths.py`.
- Keep visualizer pages directly under `teenyzero/visualizers/`, not nested under extra wrapper folders.

## Performance Priorities

- Favor batched GPU inference and lower Python overhead in the MCTS path.
- Avoid reintroducing dict-heavy edge stats in the search core.
- Reuse roots across played moves whenever the line has not diverged.
- Treat `python-chess` as the current fallback, not the long-term fast path.

## Storage

- Do not commit generated checkpoints or replay shards.
- Prefer pruning or moving archives to `var/` over keeping binary history in the repo.
- Keep docs updated when commands or runtime paths change.
