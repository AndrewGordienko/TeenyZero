import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

from teenyzero.alphazero.runtime import get_runtime_selection
from teenyzero.autotune.phase1 import (
    build_phase1_candidates,
    hardware_fingerprint,
    save_phase1_run,
)
from teenyzero.paths import AUTOTUNE_DIR, ensure_runtime_dirs, runtime_paths_payload


RUNTIME = get_runtime_selection()
PROFILE = RUNTIME.profile


def _project_root() -> Path:
    return PROJECT_ROOT


def _board_backend_name() -> str:
    return (os.environ.get("TEENYZERO_BOARD_BACKEND", "") or "auto").strip().lower() or "auto"


def _subprocess_env() -> dict:
    env = dict(os.environ)
    project_root = str(_project_root())
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = project_root if not existing else f"{project_root}{os.pathsep}{existing}"
    return env


def _temp_json_path(prefix: str) -> str:
    ensure_runtime_dirs()
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".json", dir=str(AUTOTUNE_DIR))
    os.close(fd)
    return path


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _cleanup_temp(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _run_command(command: list[str], timeout_s: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(_project_root()),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _run_selfplay_trial(trial_config: dict, args) -> dict:
    output_path = _temp_json_path("selfplay_")
    command = [
        sys.executable,
        str(_project_root() / "scripts" / "benchmark_self_play.py"),
        "--workers",
        str(trial_config["selfplay_workers"]),
        "--searches-per-worker",
        str(args.searches_per_worker),
        "--simulations",
        str(args.selfplay_simulations),
        "--leaf-batch-size",
        str(trial_config["selfplay_leaf_batch_size"]),
        "--actor-mode",
        str(trial_config["actor_mode"]),
        "--json-output",
        output_path,
    ]
    try:
        completed = _run_command(command, timeout_s=args.trial_timeout_s)
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": completed.stderr.strip() or completed.stdout.strip() or "self-play benchmark failed",
                "command": command,
            }
        payload = _load_json(output_path)
        summary = payload.get("summary", {})
        return {
            "ok": True,
            "command": command,
            "raw": payload,
            "searches_per_s": float(payload.get("searches_per_s", 0.0)),
            "simulations_per_s": float(payload.get("simulations_per_s", 0.0)),
            "positions_per_s": float(payload.get("positions_per_s", 0.0)),
            "move_total_mean_ms": float(summary.get("move_total_ms", {}).get("mean", 0.0)),
            "selection_mean_ms": float(summary.get("selection_ms", {}).get("mean", 0.0)),
            "leaf_eval_mean_ms": float(summary.get("leaf_eval_ms", {}).get("mean", 0.0)),
            "duration_s": float(payload.get("duration_s", 0.0)),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"self-play benchmark timed out after {args.trial_timeout_s:.0f}s", "command": command}
    finally:
        _cleanup_temp(output_path)


def _run_train_trial(trial_config: dict, args) -> dict:
    output_path = _temp_json_path("train_")
    command = [
        sys.executable,
        str(_project_root() / "scripts" / "benchmark_train.py"),
        "--batch-size",
        str(trial_config["train_batch_size"]),
        "--batches",
        str(args.train_batches),
        "--num-workers",
        str(trial_config["train_num_workers"]),
        "--precision",
        str(trial_config["train_precision"]),
        "--json-output",
        output_path,
    ]
    if trial_config["train_compile"]:
        command.append("--compile")
    if trial_config["train_pin_memory"]:
        command.append("--pin-memory")
    try:
        completed = _run_command(command, timeout_s=args.trial_timeout_s)
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": completed.stderr.strip() or completed.stdout.strip() or "train benchmark failed",
                "command": command,
            }
        payload = _load_json(output_path)
        metrics = payload.get("metrics", {})
        return {
            "ok": True,
            "command": command,
            "raw": payload,
            "samples_per_s": float(metrics.get("samples_per_s", 0.0)),
            "batches_per_s": float(metrics.get("batches_per_s", 0.0)),
            "avg_batch_time_ms": float(metrics.get("avg_batch_time_ms", 0.0)),
            "loss": float(metrics.get("loss", 0.0)),
            "duration_s": float(payload.get("duration_s", 0.0)),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"train benchmark timed out after {args.trial_timeout_s:.0f}s", "command": command}
    finally:
        _cleanup_temp(output_path)


def _build_trial_record(candidate: dict, args) -> dict:
    started = time.time()
    config = dict(candidate["config"])
    selfplay_result = _run_selfplay_trial(config, args)
    train_result = _run_train_trial(config, args)
    errors = []
    if not selfplay_result.get("ok"):
        errors.append(selfplay_result.get("error", "self-play benchmark failed"))
    if not train_result.get("ok"):
        errors.append(train_result.get("error", "train benchmark failed"))

    return {
        "index": int(candidate["index"]),
        "label": candidate["label"],
        "is_baseline": bool(candidate.get("is_baseline")),
        "config": config,
        "started_at": started,
        "finished_at": time.time(),
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "selfplay": selfplay_result if selfplay_result.get("ok") else {},
        "train": train_result if train_result.get("ok") else {},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run phase 1 TeenyZero hardware/runtime autotuning.")
    parser.add_argument("--trials", type=int, default=10, help="Maximum number of phase 1 configurations to try.")
    parser.add_argument(
        "--objective",
        choices=["balanced", "selfplay", "train"],
        default="balanced",
        help="Scoring objective for ranking phase 1 results.",
    )
    parser.add_argument("--searches-per-worker", type=int, default=8, help="Searches per worker for the self-play benchmark.")
    parser.add_argument("--selfplay-simulations", type=int, default=PROFILE.selfplay_simulations)
    parser.add_argument("--train-batches", type=int, default=8, help="Measured batches for the synthetic trainer benchmark.")
    parser.add_argument("--time-budget-minutes", type=float, default=30.0, help="Stop early if the wall-clock budget is exceeded.")
    parser.add_argument("--trial-timeout-s", type=float, default=600.0, help="Timeout per benchmark subprocess.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for candidate ordering.")
    return parser.parse_args()


def _print_trial_summary(trial: dict) -> None:
    config = trial["config"]
    print(f"\n[{trial['label']}] {trial['status'].upper()}")
    print(
        "  config: "
        f"mode={config['actor_mode']} "
        f"workers={config['selfplay_workers']} "
        f"leaf={config['selfplay_leaf_batch_size']} "
        f"train_batch={config['train_batch_size']} "
        f"train_workers={config['train_num_workers']} "
        f"precision={config['train_precision']} "
        f"compile={config['train_compile']}"
    )
    if trial["status"] != "ok":
        for error in trial.get("errors", []):
            print(f"  error: {error}")
        return
    print(
        "  self-play: "
        f"{trial['selfplay']['positions_per_s']:.1f} pos/s, "
        f"{trial['selfplay']['searches_per_s']:.1f} searches/s, "
        f"{trial['selfplay']['move_total_mean_ms']:.1f} ms/move"
    )
    print(
        "  train: "
        f"{trial['train']['samples_per_s']:.1f} samples/s, "
        f"{trial['train']['batches_per_s']:.2f} batches/s, "
        f"{trial['train']['avg_batch_time_ms']:.1f} ms/batch"
    )


def main():
    args = parse_args()
    ensure_runtime_dirs()
    board_backend = _board_backend_name()
    started_at = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))

    run_payload = {
        "phase": "phase1",
        "status": "running",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "objective": args.objective,
        "runtime_args": {
            "device": RUNTIME.device,
            "profile": PROFILE.name,
            "board_backend": board_backend,
        },
        "hardware": hardware_fingerprint(RUNTIME),
        "runtime_paths": runtime_paths_payload(),
        "search_settings": {
            "trials": int(args.trials),
            "searches_per_worker": int(args.searches_per_worker),
            "selfplay_simulations": int(args.selfplay_simulations),
            "train_batches": int(args.train_batches),
            "time_budget_minutes": float(args.time_budget_minutes),
            "trial_timeout_s": float(args.trial_timeout_s),
            "seed": int(args.seed),
        },
        "current_trial": None,
        "trials": [],
    }
    save_phase1_run(run_payload)

    candidates = build_phase1_candidates(PROFILE, RUNTIME.device, trial_count=args.trials, seed=args.seed)
    deadline = started_at + max(0.0, float(args.time_budget_minutes)) * 60.0

    print("\nTeenyZero Autotune Phase 1")
    print(f"Device: {RUNTIME.device}")
    print(f"Profile: {PROFILE.name}")
    print(f"Board backend: {board_backend}")
    print(f"Trials requested: {len(candidates)}")
    print(f"Objective: {args.objective}")

    try:
        for candidate in candidates:
            if time.time() >= deadline:
                print("\n[*] Time budget reached. Stopping phase 1 early.")
                break

            run_payload["current_trial"] = {
                "index": candidate["index"],
                "label": candidate["label"],
                "config": candidate["config"],
            }
            save_phase1_run(run_payload)
            trial = _build_trial_record(candidate, args)
            run_payload["trials"].append(trial)
            _print_trial_summary(trial)
            save_phase1_run(run_payload)
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        run_payload["finished_at"] = time.time()
        save_phase1_run(run_payload, archive=True)
        print("\n[!] Autotune interrupted. Partial results were saved.")
        return

    run_payload["status"] = "completed"
    run_payload["finished_at"] = time.time()
    archive_path = save_phase1_run(run_payload, archive=True)
    latest = save_phase1_run(run_payload)
    print(f"\n[*] Phase 1 results saved to {latest}")
    print(f"[*] Archived run saved to {archive_path}")
    print("[*] Open http://localhost:5001/autotune to inspect the dashboard.")
    print("[*] Promote the best run into the shared catalog with: python3 scripts/promote_autotune.py")


if __name__ == "__main__":
    main()
