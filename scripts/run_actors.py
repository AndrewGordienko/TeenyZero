import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from contextlib import nullcontext
import time
import torch
import argparse
import multiprocessing as mp

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

from teenyzero.alphazero.checkpoints import build_model, load_checkpoint, save_checkpoint
from teenyzero.alphazero.runtime import get_runtime_selection
from teenyzero.alphazero.servers.inference import inference_worker
from teenyzero.alphazero.logic.collector import DataCollector
from teenyzero.mcts.evaluator import AlphaZeroEvaluator
from teenyzero.mcts.search import MCTS
from teenyzero.paths import BEST_MODEL_PATH, REPLAY_BUFFER_PATH, ensure_runtime_dirs
from teenyzero.visualizers.cluster_monitor.dashboard import run_dashboard


RUNTIME = get_runtime_selection()
PROFILE = RUNTIME.profile


def _available_cpu_count():
    count_fn = getattr(os, "process_cpu_count", os.cpu_count)
    count = count_fn()
    return int(count or 1)


def _default_worker_count(device: str) -> int:
    if device not in {"cuda", "mps"}:
        return 4
    cpu_budget = max(2, _available_cpu_count())
    reserved = 4 if device == "cuda" else 2
    usable = max(2, cpu_budget - reserved)
    target = usable // 2 if device == "mps" else usable // 3
    return max(2, min(PROFILE.selfplay_workers, target or 1))


def bootstrap_model(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = build_model()
    if os.path.exists(path):
        load_result = load_checkpoint(model, path, map_location="cpu", allow_partial=True)
        if load_result["loaded"]:
            return
        print(f"[*] Replacing unusable checkpoint at {path} ({load_result['reason']})...")
    else:
        print(f"[*] Initializing fresh AlphaNet at {path}...")
    save_checkpoint(model, path)


def worker_task(worker_id, task_queue, response_queue, shared_stats, leaf_batch_size):
    """
    Self-play worker loop.
    """
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    evaluator = AlphaZeroEvaluator(
        task_queue=task_queue,
        response_queue=response_queue,
        worker_id=worker_id,
        use_cache=True,
    )

    mcts_params = {
        "SIMULATIONS": PROFILE.selfplay_simulations,
        "C_PUCT": 1.5,
        "ALPHA": 0.3,
        "EPS": 0.30,
        "VIRTUAL_LOSS": 0.0,
        "PARALLEL_THREADS": 1,
        "FPU_REDUCTION": 0.4,
        "LEAF_BATCH_SIZE": leaf_batch_size,
    }
    engine = MCTS(evaluator=evaluator, params=mcts_params)

    collector = DataCollector(
        evaluator,
        engine,
        buffer_path=str(REPLAY_BUFFER_PATH),
    )

    print(f"[Worker {worker_id}] Online and searching...")

    while True:
        game_data = collector.collect_game(worker_id=worker_id, stats_dict=shared_stats)

        timestamp = int(time.time() * 1000)
        filename = f"game_{worker_id}_{timestamp}.npz"
        collector.save_batch(game_data, filename)

        print(f"[Worker {worker_id}] Saved {len(game_data)} positions to {filename}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = str(BEST_MODEL_PATH)
    DEVICE = RUNTIME.device

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of self-play processes (defaults come from the active runtime profile)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the cluster dashboard and shared live stats for max throughput.",
    )
    args = parser.parse_args()
    env_workers = os.environ.get("TEENYZERO_SELFPLAY_WORKERS", "").strip()
    env_workers = int(env_workers) if env_workers.isdigit() else None
    default_workers = _default_worker_count(DEVICE)
    leaf_batch_size = PROFILE.selfplay_leaf_batch_size if DEVICE in {"cuda", "mps"} else 8
    worker_count = args.workers if args.workers is not None else (env_workers if env_workers is not None else default_workers)
    dashboard_enabled = not args.no_dashboard

    ensure_runtime_dirs()
    bootstrap_model(MODEL_PATH)

    manager_context = mp.Manager() if dashboard_enabled else nullcontext()
    with manager_context as manager:
        shared_stats = manager.dict() if dashboard_enabled else None
        if shared_stats is not None:
            shared_stats["__cluster__"] = {
                "config": {
                    "device": DEVICE,
                    "profile": PROFILE.name,
                    "available_cpus": _available_cpu_count(),
                    "profile_worker_budget": PROFILE.selfplay_workers,
                    "workers": worker_count,
                    "model_path": MODEL_PATH,
                    "simulations": PROFILE.selfplay_simulations,
                    "leaf_batch_size": leaf_batch_size,
                }
            }

        task_queue = mp.Queue(maxsize=4096)
        response_queues = [mp.Queue(maxsize=1024) for _ in range(worker_count)]
        processes = []

        inf_p = mp.Process(
            target=inference_worker,
            args=(MODEL_PATH, DEVICE, task_queue, response_queues, shared_stats),
        )
        inf_p.daemon = True
        inf_p.start()
        processes.append(inf_p)

        if dashboard_enabled and shared_stats is not None:
            dash_p = mp.Process(target=run_dashboard, args=(shared_stats,))
            dash_p.daemon = True
            dash_p.start()
            processes.append(dash_p)

        for i in range(worker_count):
            p = mp.Process(
                target=worker_task,
                args=(i, task_queue, response_queues[i], shared_stats, leaf_batch_size),
            )
            p.daemon = True
            p.start()
            processes.append(p)

        print("\n" + "=" * 50)
        print("  TEENYZERO FACTORY ONLINE")
        print("=" * 50)
        print(f"[*] Workers:   {worker_count}")
        if worker_count != PROFILE.selfplay_workers and DEVICE in {'cuda', 'mps'}:
            print(f"[*] Capped from profile default {PROFILE.selfplay_workers} based on available CPU budget")
        print(f"[*] Device:    {DEVICE}")
        print(f"[*] Profile:   {PROFILE.name}")
        if dashboard_enabled:
            print(f"[*] Dashboard: http://localhost:5002")
        else:
            print("[*] Dashboard: disabled")
        print(f"[*] Data:      {REPLAY_BUFFER_PATH}")
        print("Press Ctrl+C to stop cluster.")
        print("=" * 50 + "\n")

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\n[!] Shutting down cluster...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join(timeout=1.0)
