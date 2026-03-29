# Autotune Results

This file tracks promoted phase 1 runtime recommendations that can be shared in the repo.
Each entry comes from an autotune run and captures a recommended hardware/runtime setup for a workload.

The source of truth is the checked-in registry at `teenyzero/autotune/recommendations.json`.

| ID | Title | Workload | Seed | Best Trial | Self-Play Pos/Sec | Train Samples/Sec |
| --- | --- | --- | --- | --- | ---: | ---: |
| apple_silicon_mps_balanced | Apple Silicon MPS 24GB Balanced | balanced | mps/mps/native | T03 | 900.4 | 295.3 |

## Apple Silicon MPS 24GB Balanced

- `id`: `apple_silicon_mps_balanced`
- `workload`: `balanced`
- `device family`: `apple_silicon_mps`
- `phase 1 score`: `2.146`
- `self-play positions/sec`: `900.4`
- `train samples/sec`: `295.3`
- `best config`: `mode=inprocess`, `workers=6`, `leaf=48`, `train_batch=144`, `train_workers=4`, `precision=fp16`, `compile=False`
- `summary`: Promoted from phase 1 autotune run 20260329_133749.

```bash
python3 scripts/run_visualizers.py --device mps --profile mps --board-backend native --actor-mode inprocess --actor-workers 6 --selfplay-leaf-batch-size 48 --train-batch-size 144 --train-num-workers 4 --train-precision fp16 --no-train-pin-memory --no-train-compile
```
