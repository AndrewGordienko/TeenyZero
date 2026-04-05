import json
import os
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

from teenyzero.alphafold.model import AlphaFoldBoardModel
from teenyzero.alphafold.pretrain import AlphaFoldPretrainer, RandomBoardPositionDataset
from teenyzero.alphazero.config import INPUT_PLANES, MODEL_CHANNELS, MODEL_RES_BLOCKS
from teenyzero.alphazero.runtime import get_runtime_selection
from teenyzero.paths import (
    ALPHAFOLD_MODEL_PATH,
    ALPHAFOLD_TRAINING_HISTORY_PATH,
    ALPHAFOLD_TRAINING_STATE_PATH,
    ensure_runtime_dirs,
)


RUNTIME = get_runtime_selection()
PROFILE = RUNTIME.profile


def _write_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _append_history(path: Path, entry: dict):
    history = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as handle:
                history = json.load(handle)
        except Exception:
            history = []
    history.append(entry)
    history = history[-200:]
    _write_state(path, history)


def _build_model():
    return AlphaFoldBoardModel(
        input_planes=INPUT_PLANES,
        channels=MODEL_CHANNELS,
        num_relation_blocks=MODEL_RES_BLOCKS,
    )


def main():
    parser = argparse.ArgumentParser(description="Pretrain the AlphaFold-style board geometry model.")
    parser.add_argument("--positions", type=int, default=16384, help="Random board positions to generate per epoch.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--min-plies", type=int, default=6, help="Minimum random plies before sampling a board.")
    parser.add_argument("--max-plies", type=int, default=80, help="Maximum random plies before sampling a board.")
    parser.add_argument("--seed", type=int, default=1337, help="Base RNG seed for board generation.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=PROFILE.train_precision)
    parser.add_argument("--compile", action="store_true", help="Compile the model during training when supported.")
    parser.add_argument(
        "--checkpoint-path",
        default=str(ALPHAFOLD_MODEL_PATH),
        help="Output checkpoint path for the AlphaFold-style pretrained model.",
    )
    args = parser.parse_args()

    ensure_runtime_dirs()
    device = RUNTIME.device
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    state_path = Path(ALPHAFOLD_TRAINING_STATE_PATH)
    history_path = Path(ALPHAFOLD_TRAINING_HISTORY_PATH)

    model = _build_model()
    trainer = AlphaFoldPretrainer(
        model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        precision=args.precision,
        use_compile=args.compile,
        grad_accum_steps=PROFILE.train_grad_accum_steps,
        max_grad_norm=PROFILE.max_grad_norm,
    )

    load_result = trainer.load_checkpoint(checkpoint_path)
    if load_result["loaded"]:
        flavor = "partial" if load_result["partial"] else "full"
        print(f"[AlphaFold] Loaded {flavor} checkpoint from {checkpoint_path} ({load_result['reason']}).")
    else:
        print(f"[AlphaFold] Starting from fresh weights ({load_result['reason']}).")

    state = {
        "status": "starting",
        "device": device,
        "runtime_profile": PROFILE.name,
        "checkpoint_path": checkpoint_path,
        "positions_per_epoch": int(args.positions),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "min_plies": int(args.min_plies),
        "max_plies": int(args.max_plies),
        "precision": str(args.precision),
        "seed": int(args.seed),
        "last_epoch": 0,
        "last_loss": 0.0,
        "last_attack_loss": 0.0,
        "last_pressure_loss": 0.0,
        "last_samples_per_s": 0.0,
        "last_batches_per_s": 0.0,
        "started_at": time.time(),
    }
    _write_state(state_path, state)

    print(
        "[AlphaFold] Training geometry pretrain with "
        f"{args.positions} positions/epoch for {args.epochs} epoch(s) on {device}."
    )

    for epoch_idx in range(1, int(args.epochs) + 1):
        dataset = RandomBoardPositionDataset(
            sample_count=args.positions,
            min_plies=args.min_plies,
            max_plies=args.max_plies,
            rng_seed=args.seed + epoch_idx * 1_000_003,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=device == "cuda",
        )

        def on_progress(progress):
            state.update(
                {
                    "status": progress.get("stage", "alphafold_training_batches"),
                    "last_epoch": epoch_idx,
                    "completed_batches": int(progress.get("completed_batches", 0)),
                    "total_batches": int(progress.get("total_batches", 0)),
                    "running_loss": float(progress.get("running_loss", 0.0)),
                    "running_attack_loss": float(progress.get("running_attack_loss", 0.0)),
                    "running_pressure_loss": float(progress.get("running_pressure_loss", 0.0)),
                    "last_samples_per_s": float(progress.get("samples_per_s", 0.0)),
                    "last_batches_per_s": float(progress.get("batches_per_s", 0.0)),
                    "heartbeat_at": time.time(),
                }
            )
            _write_state(state_path, state)

        metrics = trainer.train_epoch(dataloader, progress_callback=on_progress)
        trainer.save_checkpoint(
            checkpoint_path,
            extra_meta={
                "epoch": epoch_idx,
                "positions_per_epoch": int(args.positions),
                "min_plies": int(args.min_plies),
                "max_plies": int(args.max_plies),
            },
        )

        state.update(
            {
                "status": "saved_checkpoint",
                "last_epoch": epoch_idx,
                "last_loss": float(metrics["loss"]),
                "last_attack_loss": float(metrics["attack_loss"]),
                "last_pressure_loss": float(metrics["pressure_loss"]),
                "last_samples_per_s": float(metrics.get("samples_per_s", 0.0)),
                "last_batches_per_s": float(metrics.get("batches_per_s", 0.0)),
                "heartbeat_at": time.time(),
            }
        )
        _write_state(state_path, state)
        _append_history(
            history_path,
            {
                "epoch": epoch_idx,
                "finished_at": time.time(),
                "loss": float(metrics["loss"]),
                "attack_loss": float(metrics["attack_loss"]),
                "pressure_loss": float(metrics["pressure_loss"]),
                "samples_seen": int(metrics.get("samples_seen", 0)),
                "batches": int(metrics.get("batches", 0)),
                "duration_s": float(metrics.get("duration_s", 0.0)),
                "samples_per_s": float(metrics.get("samples_per_s", 0.0)),
                "batches_per_s": float(metrics.get("batches_per_s", 0.0)),
                "avg_batch_time_ms": float(metrics.get("avg_batch_time_ms", 0.0)),
            },
        )

        print(
            f"[AlphaFold] Epoch {epoch_idx}/{args.epochs}: "
            f"loss={metrics['loss']:.4f} "
            f"attack={metrics['attack_loss']:.4f} "
            f"pressure={metrics['pressure_loss']:.4f}"
        )

    state["status"] = "done"
    state["finished_at"] = time.time()
    _write_state(state_path, state)
    print(f"[AlphaFold] Saved pretrained checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
