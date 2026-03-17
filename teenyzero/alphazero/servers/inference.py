import time
import queue
import os
from collections import deque
import torch
import numpy as np

from teenyzero.alphazero.checkpoints import build_model, load_checkpoint
from teenyzero.alphazero.runtime import get_runtime_profile


PROFILE = get_runtime_profile()


def inference_worker(model_path, device, task_queue, response_queues, shared_stats=None):
    """
    Dynamic batched inference worker.

    Request format from workers:
        (task_id, encoded_state_or_batch, worker_id, is_batch)

    Single response format:
        (task_id, logits, value, False)

    Batched response format:
        (task_id, logits_batch, values_batch, True)
    """
    print(f"[Inference] Initializing AlphaNet on {device}...")
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    model = build_model()
    if device == "cuda" and PROFILE.inference_precision == "bf16":
        inference_dtype = torch.bfloat16
    else:
        inference_dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    use_channels_last = device in {"mps", "cuda"}

    if device == "cuda":
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    try:
        load_result = load_checkpoint(model, model_path, map_location="cpu", allow_partial=True)
        if load_result["loaded"]:
            print(f"[Inference] Weights loaded successfully from {model_path}")
        else:
            print(f"[Inference] Warning: Starting with fresh weights ({load_result['reason']})")
    except Exception as e:
        print(f"[Inference] Warning: Starting with fresh weights ({e})")

    try:
        model_mtime = os.path.getmtime(model_path)
    except OSError:
        model_mtime = None

    model = model.to(device=device, dtype=inference_dtype)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if device == "cuda" and PROFILE.inference_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model.eval()

    MAX_SINGLE_BATCH = PROFILE.inference_single_batch if device in {"cuda", "mps"} else 64
    MAX_MERGED_POSITIONS = PROFILE.inference_merged_batch if device in {"cuda", "mps"} else 64
    WAIT_TIMEOUT = PROFILE.inference_wait_timeout
    IDLE_GET_TIMEOUT = 1.0
    last_stats_push = 0.0
    pending_tasks = deque()
    cluster_stats = {
        "device": device,
        "total_requests": 0,
        "total_positions": 0,
        "single_requests": 0,
        "explicit_batch_requests": 0,
        "dynamic_batches": 0,
        "merged_explicit_batches": 0,
        "server_forwards": 0,
        "avg_dynamic_batch_size": 0.0,
        "avg_explicit_batch_size": 0.0,
        "avg_merged_batch_size": 0.0,
        "avg_gather_wait_ms": 0.0,
        "avg_forward_ms": 0.0,
        "queue_depth": 0,
    }

    def maybe_reload_model():
        nonlocal model_mtime
        try:
            current_mtime = os.path.getmtime(model_path)
        except OSError:
            return

        if model_mtime is not None and current_mtime <= model_mtime:
            return

        try:
            load_checkpoint(model, model_path, map_location="cpu", allow_partial=True)
            model.to(device=device, dtype=inference_dtype)
            if use_channels_last:
                model.to(memory_format=torch.channels_last)
            model.eval()
            model_mtime = current_mtime
            print(f"[Inference] Reloaded weights from {model_path}")
        except Exception as exc:
            print(f"[Inference] Failed to reload weights: {exc}")

    def next_task():
        if pending_tasks:
            return pending_tasks.popleft()
        return task_queue.get(timeout=IDLE_GET_TIMEOUT)

    while True:
        maybe_reload_model()
        singles = []
        singles_meta = []
        batch_requests = []
        merged_positions = 0

        try:
            task = next_task()
        except queue.Empty:
            continue
        except Exception:
            continue

        if len(task) == 4:
            task_id, payload, worker_id, is_batch = task
        else:
            task_id, payload, worker_id = task
            is_batch = False

        if is_batch:
            batch_requests.append((task_id, payload, worker_id))
            merged_positions += int(len(payload))
        else:
            singles.append(payload)
            singles_meta.append((task_id, worker_id))
            merged_positions += 1

        gather_start = time.perf_counter()
        if not is_batch:
            while len(singles) < MAX_SINGLE_BATCH and merged_positions < MAX_MERGED_POSITIONS:
                try:
                    task = pending_tasks.popleft() if pending_tasks else task_queue.get_nowait()
                except queue.Empty:
                    if (time.perf_counter() - gather_start) >= WAIT_TIMEOUT:
                        break
                    continue
                except Exception:
                    break

                if len(task) == 4:
                    task_id, payload, worker_id, is_batch = task
                else:
                    task_id, payload, worker_id = task
                    is_batch = False

                if is_batch:
                    payload_len = int(len(payload))
                    if merged_positions + payload_len > MAX_MERGED_POSITIONS and merged_positions > 0:
                        pending_tasks.appendleft((task_id, payload, worker_id, True))
                        break
                    batch_requests.append((task_id, payload, worker_id))
                    merged_positions += payload_len
                else:
                    singles.append(payload)
                    singles_meta.append((task_id, worker_id))
                    merged_positions += 1
        else:
            while merged_positions < MAX_MERGED_POSITIONS:
                try:
                    task = pending_tasks.popleft() if pending_tasks else task_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break

                if len(task) == 4:
                    task_id, payload, worker_id, nested_is_batch = task
                else:
                    task_id, payload, worker_id = task
                    nested_is_batch = False

                if nested_is_batch:
                    payload_len = int(len(payload))
                    if merged_positions + payload_len > MAX_MERGED_POSITIONS and merged_positions > 0:
                        pending_tasks.appendleft((task_id, payload, worker_id, True))
                        break
                    batch_requests.append((task_id, payload, worker_id))
                    merged_positions += payload_len
                elif len(singles) < MAX_SINGLE_BATCH and merged_positions < MAX_MERGED_POSITIONS:
                    singles.append(payload)
                    singles_meta.append((task_id, worker_id))
                    merged_positions += 1
                else:
                    pending_tasks.appendleft((task_id, payload, worker_id, False))
                    break

        gather_wait_ms = (time.perf_counter() - gather_start) * 1000.0

        single_batch = np.asarray(singles, dtype=np.float32) if singles else None
        explicit_arrays = [np.asarray(payload, dtype=np.float32) for _, payload, _ in batch_requests]
        forward_inputs = []

        if single_batch is not None and len(single_batch) > 0:
            forward_inputs.append(single_batch)
        if explicit_arrays:
            forward_inputs.extend(explicit_arrays)

        if forward_inputs:
            merged_batch = np.concatenate(forward_inputs, axis=0)
            tensor = torch.from_numpy(merged_batch).to(device=device, dtype=inference_dtype, non_blocking=True)
            if use_channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)

            forward_start = time.perf_counter()
            with torch.inference_mode():
                logits, values = model(tensor)
                logits = logits.detach()
                values = values.detach()
                if logits.dtype == torch.bfloat16:
                    logits = logits.to(dtype=torch.float32)
                if values.dtype == torch.bfloat16:
                    values = values.to(dtype=torch.float32)
                logits = logits.cpu().numpy().astype(np.float16, copy=False)
                vals = values.cpu().numpy().reshape(-1)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            cluster_stats["server_forwards"] += 1
            forward_events = cluster_stats["server_forwards"]
            cluster_stats["avg_forward_ms"] += (
                (forward_ms - cluster_stats["avg_forward_ms"]) / forward_events
            )

            offset = 0
            if single_batch is not None and len(single_batch) > 0:
                meta = {
                    "forward_ms": float(forward_ms),
                    "gather_wait_ms": float(gather_wait_ms),
                    "batch_size": int(len(single_batch)),
                    "merged_batch_size": int(len(merged_batch)),
                }
                for i, (task_id, worker_id) in enumerate(singles_meta):
                    response_queues[worker_id].put(
                        (task_id, logits[offset + i], float(vals[offset + i]), False, meta)
                    )

                offset += len(single_batch)
                cluster_stats["dynamic_batches"] += 1
                cluster_stats["single_requests"] += len(singles_meta)
                cluster_stats["total_requests"] += len(singles_meta)
                cluster_stats["total_positions"] += len(single_batch)
                dynamic_batches = cluster_stats["dynamic_batches"]
                cluster_stats["avg_dynamic_batch_size"] += (
                    (len(single_batch) - cluster_stats["avg_dynamic_batch_size"]) / dynamic_batches
                )
                cluster_stats["avg_gather_wait_ms"] += (
                    (gather_wait_ms - cluster_stats["avg_gather_wait_ms"]) / dynamic_batches
                )

            if explicit_arrays:
                meta = {
                    "forward_ms": float(forward_ms),
                    "gather_wait_ms": float(gather_wait_ms),
                    "merged_batch_size": int(len(merged_batch)),
                    "request_count": int(len(batch_requests)),
                }
                for (task_id, payload, worker_id), np_batch in zip(batch_requests, explicit_arrays):
                    batch_len = len(np_batch)
                    response_queues[worker_id].put(
                        (task_id, logits[offset:offset + batch_len], vals[offset:offset + batch_len], True, meta)
                    )
                    offset += batch_len

                cluster_stats["explicit_batch_requests"] += len(batch_requests)
                cluster_stats["merged_explicit_batches"] += 1
                cluster_stats["total_requests"] += len(batch_requests)
                cluster_stats["total_positions"] += sum(len(batch) for batch in explicit_arrays)
                explicit_requests = cluster_stats["explicit_batch_requests"]
                merged_explicit_batches = cluster_stats["merged_explicit_batches"]
                cluster_stats["avg_explicit_batch_size"] += (
                    ((sum(len(batch) for batch in explicit_arrays) / max(1, len(batch_requests))) - cluster_stats["avg_explicit_batch_size"]) / explicit_requests
                )
                cluster_stats["avg_merged_batch_size"] += (
                    (sum(len(batch) for batch in explicit_arrays) - cluster_stats["avg_merged_batch_size"]) / merged_explicit_batches
                )

        if shared_stats is not None and (time.perf_counter() - last_stats_push) >= 0.5:
            try:
                queue_depth = task_queue.qsize()
            except (AttributeError, NotImplementedError):
                queue_depth = -1

            cluster_stats["queue_depth"] = queue_depth + len(pending_tasks) if queue_depth >= 0 else len(pending_tasks)
            current_cluster = dict(shared_stats.get("__cluster__", {}))
            current_cluster["inference"] = {
                "device": cluster_stats["device"],
                "total_requests": int(cluster_stats["total_requests"]),
                "total_positions": int(cluster_stats["total_positions"]),
                "single_requests": int(cluster_stats["single_requests"]),
                "explicit_batch_requests": int(cluster_stats["explicit_batch_requests"]),
                "dynamic_batches": int(cluster_stats["dynamic_batches"]),
                "merged_explicit_batches": int(cluster_stats["merged_explicit_batches"]),
                "avg_dynamic_batch_size": float(cluster_stats["avg_dynamic_batch_size"]),
                "avg_explicit_batch_size": float(cluster_stats["avg_explicit_batch_size"]),
                "avg_merged_batch_size": float(cluster_stats["avg_merged_batch_size"]),
                "avg_gather_wait_ms": float(cluster_stats["avg_gather_wait_ms"]),
                "avg_forward_ms": float(cluster_stats["avg_forward_ms"]),
                "queue_depth": int(cluster_stats["queue_depth"]),
                "max_batch_size": int(MAX_MERGED_POSITIONS),
            }
            shared_stats["__cluster__"] = current_cluster
            last_stats_push = time.perf_counter()
