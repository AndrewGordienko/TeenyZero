from __future__ import annotations

import itertools
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from teenyzero.alphazero.runtime import RuntimeProfile, RuntimeSelection
from teenyzero.paths import AUTOTUNE_LATEST_PATH, AUTOTUNE_RUNS_DIR, ensure_runtime_dirs


@dataclass(frozen=True)
class Phase1Config:
    actor_mode: str
    selfplay_workers: int
    selfplay_leaf_batch_size: int
    train_batch_size: int
    train_num_workers: int
    train_pin_memory: bool
    train_precision: str
    train_compile: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _physical_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    except (AttributeError, OSError, ValueError):
        return None


def _cuda_device_payload() -> dict | None:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(getattr(props, "multi_processor_count", 0) or 0),
            "major": int(getattr(props, "major", 0) or 0),
            "minor": int(getattr(props, "minor", 0) or 0),
        }
    except Exception:
        return {"name": "cuda", "total_memory_bytes": None}


def hardware_fingerprint(selection: RuntimeSelection) -> dict:
    return {
        "device": selection.device,
        "runtime_profile": selection.profile.name,
        "requested_device": selection.requested_device,
        "requested_profile": selection.requested_profile,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu_count": int(os.cpu_count() or 1),
        "physical_memory_bytes": _physical_memory_bytes(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cuda_device": _cuda_device_payload(),
    }


def _unique_int_candidates(base_value: int, minimum: int, maximum: int) -> list[int]:
    points = {int(base_value), minimum, maximum}
    for ratio in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        points.add(int(round(base_value * ratio)))
    values = sorted(max(minimum, min(maximum, point)) for point in points)
    return [value for value in values if value >= minimum]


def _baseline_actor_mode(device: str) -> str:
    return "inprocess" if device in {"mps", "cuda"} else "mp"


def _precision_options(profile: RuntimeProfile, device: str) -> list[str]:
    if device != "cuda":
        return [profile.train_precision]
    values = [profile.train_precision, "fp32", "fp16", "bf16"]
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _compile_options(profile: RuntimeProfile, device: str) -> list[bool]:
    if device != "cuda" or not hasattr(torch, "compile"):
        return [False]
    values = [bool(profile.train_compile), False, True]
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _pin_memory_options(profile: RuntimeProfile, device: str) -> list[bool]:
    if device != "cuda":
        return [False]
    values = [bool(profile.train_pin_memory), False, True]
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def build_phase1_candidates(
    profile: RuntimeProfile,
    device: str,
    trial_count: int,
    seed: int = 0,
) -> list[dict]:
    cpu_count = max(1, int(os.cpu_count() or 1))
    worker_cap = max(2, min(cpu_count, max(profile.selfplay_workers * 2, 4)))
    train_worker_cap = max(0, min(cpu_count, max(profile.train_num_workers * 2 + 2, 4)))
    leaf_cap = max(8, profile.selfplay_leaf_batch_size * 2)
    batch_cap = max(16, min(512, profile.train_batch_size * 2))

    baseline = Phase1Config(
        actor_mode=_baseline_actor_mode(device),
        selfplay_workers=max(1, int(profile.selfplay_workers)),
        selfplay_leaf_batch_size=max(1, int(profile.selfplay_leaf_batch_size)),
        train_batch_size=max(1, int(profile.train_batch_size)),
        train_num_workers=max(0, int(profile.train_num_workers)),
        train_pin_memory=bool(profile.train_pin_memory if device == "cuda" else False),
        train_precision=str(profile.train_precision),
        train_compile=bool(profile.train_compile if device == "cuda" and hasattr(torch, "compile") else False),
    )

    actor_modes = ["inprocess", "mp"] if device in {"mps", "cuda"} else ["mp", "inprocess"]
    combos = itertools.product(
        actor_modes,
        _unique_int_candidates(baseline.selfplay_workers, minimum=1, maximum=worker_cap),
        _unique_int_candidates(baseline.selfplay_leaf_batch_size, minimum=4, maximum=leaf_cap),
        _unique_int_candidates(baseline.train_batch_size, minimum=8, maximum=batch_cap),
        _unique_int_candidates(baseline.train_num_workers, minimum=0, maximum=train_worker_cap),
        _pin_memory_options(profile, device),
        _precision_options(profile, device),
        _compile_options(profile, device),
    )

    candidates = []
    seen = set()
    for item in combos:
        config = Phase1Config(
            actor_mode=item[0],
            selfplay_workers=item[1],
            selfplay_leaf_batch_size=item[2],
            train_batch_size=item[3],
            train_num_workers=item[4],
            train_pin_memory=item[5],
            train_precision=item[6],
            train_compile=item[7],
        )
        key = tuple(config.to_dict().items())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(config)

    baseline_key = tuple(baseline.to_dict().items())
    sampled = [baseline]
    remainder = [candidate for candidate in candidates if tuple(candidate.to_dict().items()) != baseline_key]
    random.Random(seed).shuffle(remainder)
    sampled.extend(remainder[: max(0, int(trial_count) - 1)])

    total = len(sampled)
    return [
        {
            "index": idx,
            "label": f"T{idx + 1:02d}",
            "config": config.to_dict(),
            "is_baseline": idx == 0,
            "trial_count": total,
        }
        for idx, config in enumerate(sampled)
    ]


def _ratio(value: float | None, baseline: float | None) -> float:
    if value is None or baseline is None or baseline <= 0:
        return 0.0
    return float(value) / float(baseline)


def phase1_trial_score(trial: dict, baseline: dict | None, objective: str) -> float:
    if trial.get("status") != "ok":
        return -1.0

    selfplay = trial.get("selfplay", {})
    train = trial.get("train", {})
    baseline_selfplay = (baseline or {}).get("selfplay", {})
    baseline_train = (baseline or {}).get("train", {})

    positions_ratio = _ratio(selfplay.get("positions_per_s"), baseline_selfplay.get("positions_per_s"))
    searches_ratio = _ratio(selfplay.get("searches_per_s"), baseline_selfplay.get("searches_per_s"))
    latency_ratio = _ratio(
        baseline_selfplay.get("move_total_mean_ms"),
        selfplay.get("move_total_mean_ms"),
    )
    train_ratio = _ratio(train.get("samples_per_s"), baseline_train.get("samples_per_s"))

    if objective == "selfplay":
        return (0.65 * positions_ratio) + (0.20 * searches_ratio) + (0.15 * latency_ratio)
    if objective == "train":
        return train_ratio
    return (0.40 * positions_ratio) + (0.15 * searches_ratio) + (0.20 * latency_ratio) + (0.25 * train_ratio)


def build_apply_command(runtime_args: dict, config: dict) -> str:
    args = [
        "python3",
        "scripts/run_visualizers.py",
        f"--device {runtime_args['device']}",
        f"--profile {runtime_args['profile']}",
        f"--board-backend {runtime_args['board_backend']}",
        f"--actor-mode {config['actor_mode']}",
        f"--actor-workers {config['selfplay_workers']}",
        f"--selfplay-leaf-batch-size {config['selfplay_leaf_batch_size']}",
        f"--train-batch-size {config['train_batch_size']}",
        f"--train-num-workers {config['train_num_workers']}",
        f"--train-precision {config['train_precision']}",
    ]
    args.append("--train-pin-memory" if config["train_pin_memory"] else "--no-train-pin-memory")
    args.append("--train-compile" if config["train_compile"] else "--no-train-compile")
    return " ".join(args)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _normalized_phase1_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    runtime_args = normalized.get("runtime_args") or {}
    best = normalized.get("best_trial") or {}
    config = best.get("config") or {}
    if runtime_args and config:
        try:
            normalized["apply_command"] = build_apply_command(runtime_args, config)
        except KeyError:
            pass
    return normalized


def finalize_phase1_run(run_payload: dict) -> dict:
    trials = list(run_payload.get("trials", []))
    baseline = next((item for item in trials if item.get("is_baseline")), None)
    objective = str(run_payload.get("objective", "balanced"))
    for trial in trials:
        trial["score"] = phase1_trial_score(trial, baseline, objective)
    ranked = sorted(trials, key=lambda item: float(item.get("score", -1.0)), reverse=True)
    best = ranked[0] if ranked else None
    run_payload["trials"] = ranked
    run_payload["best_trial"] = best
    if best is not None:
        run_payload["apply_command"] = build_apply_command(run_payload["runtime_args"], best["config"])
    return run_payload


def save_phase1_run(run_payload: dict, archive: bool = False) -> Path:
    ensure_runtime_dirs()
    finalized = finalize_phase1_run(dict(run_payload))
    latest_path = AUTOTUNE_LATEST_PATH
    _write_json(latest_path, finalized)
    if not archive:
        return latest_path
    run_id = finalized.get("run_id") or time.strftime("%Y%m%d_%H%M%S")
    archive_path = AUTOTUNE_RUNS_DIR / f"phase1_{run_id}.json"
    _write_json(archive_path, finalized)
    return archive_path


def latest_phase1_run() -> dict | None:
    if not AUTOTUNE_LATEST_PATH.exists():
        return None
    try:
        with open(AUTOTUNE_LATEST_PATH, "r", encoding="utf-8") as handle:
            return _normalized_phase1_payload(json.load(handle))
    except Exception:
        return None


def list_phase1_runs(limit: int = 8) -> list[dict]:
    ensure_runtime_dirs()
    payloads = []
    for path in sorted(AUTOTUNE_RUNS_DIR.glob("phase1_*.json"), reverse=True)[: max(1, int(limit))]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = _normalized_phase1_payload(json.load(handle))
        except Exception:
            continue
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads
