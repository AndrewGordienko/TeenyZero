from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

from teenyzero.alphazero.checkpoints import build_model
from teenyzero.alphazero.config import INPUT_SHAPE
from teenyzero.alphazero.logic.trainer import AlphaTrainer
from teenyzero.alphazero.runtime import get_runtime_selection


RUNTIME = get_runtime_selection()
PROFILE = RUNTIME.profile


class SyntheticTrainDataset(Dataset):
    def __init__(self, sample_count: int, policy_size: int, seed: int = 0):
        self.sample_count = max(1, int(sample_count))
        rng = np.random.default_rng(seed)
        self.states = rng.standard_normal((self.sample_count, *INPUT_SHAPE)).astype(np.float32)
        self.pis = rng.random((self.sample_count, policy_size)).astype(np.float32)
        self.pis /= np.maximum(self.pis.sum(axis=1, keepdims=True), 1e-9)
        self.zs = rng.uniform(-1.0, 1.0, size=(self.sample_count, 1)).astype(np.float32)

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]).float(),
            torch.from_numpy(self.pis[idx]).float(),
            torch.from_numpy(self.zs[idx]).float(),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TeenyZero trainer throughput with synthetic batches.")
    parser.add_argument("--batch-size", type=int, default=PROFILE.train_batch_size)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=PROFILE.train_num_workers)
    parser.add_argument("--precision", default=PROFILE.train_precision, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile if supported.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", default=None, help="Optional path to write JSON benchmark results.")
    return parser.parse_args()


def build_result_payload(args, metrics, elapsed_s, policy_size):
    return {
        "benchmark": "train",
        "device": RUNTIME.device,
        "profile": PROFILE.name,
        "batch_size": int(args.batch_size),
        "batches": int(args.batches),
        "num_workers": int(args.num_workers),
        "precision": str(args.precision),
        "compile": bool(args.compile),
        "pin_memory": bool(args.pin_memory),
        "policy_size": int(policy_size),
        "sample_count": int(args.batch_size) * int(args.batches),
        "duration_s": float(elapsed_s),
        "metrics": metrics,
    }


def emit_result(payload, json_output: str | None):
    print("\nTeenyZero Train Benchmark")
    print(f"Device: {payload['device']}")
    print(f"Profile: {payload['profile']}")
    print(f"Batch Size: {payload['batch_size']}")
    print(f"Batches: {payload['batches']}")
    print(f"Num Workers: {payload['num_workers']}")
    print(f"Precision: {payload['precision']}")
    print(f"Compile: {payload['compile']}")
    print(f"Pin Memory: {payload['pin_memory']}")
    print(f"Duration: {payload['duration_s']:.2f}s")
    print(f"Samples/sec: {payload['metrics']['samples_per_s']:.2f}")
    print(f"Batches/sec: {payload['metrics']['batches_per_s']:.2f}")
    print(f"Avg Batch Time: {payload['metrics']['avg_batch_time_ms']:.1f} ms")
    if json_output:
        os.makedirs(os.path.dirname(json_output), exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)


def main():
    args = parse_args()
    device = RUNTIME.device
    model = build_model()
    policy_size = int(model.pol_fc.out_features)
    trainer = AlphaTrainer(
        model,
        device=device,
        lr=PROFILE.train_lr,
        optimizer_name=PROFILE.train_optimizer,
        weight_decay=PROFILE.train_weight_decay,
        momentum=PROFILE.train_momentum,
        grad_accum_steps=PROFILE.train_grad_accum_steps,
        precision=args.precision,
        use_compile=args.compile,
        max_grad_norm=PROFILE.max_grad_norm,
    )

    sample_count = max(1, int(args.batch_size) * int(args.batches))
    dataset = SyntheticTrainDataset(sample_count=sample_count, policy_size=policy_size, seed=args.seed)
    loader_kwargs = {
        "batch_size": max(1, int(args.batch_size)),
        "shuffle": False,
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": max(0, int(args.num_workers)) > 0,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = PROFILE.train_prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    started = time.perf_counter()
    metrics = trainer.train_epoch(loader)
    elapsed_s = time.perf_counter() - started
    payload = build_result_payload(args, metrics, elapsed_s, policy_size)
    emit_result(payload, args.json_output)


if __name__ == "__main__":
    main()
