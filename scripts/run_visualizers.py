import socket
import subprocess
import sys
from pathlib import Path

from teenyzero.runtime_bootstrap import bootstrap_runtime_cli


bootstrap_runtime_cli()

from teenyzero.alphazero.runtime import get_runtime_selection
from teenyzero.visualizers.app import app


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    runtime = get_runtime_selection()
    profile = runtime.profile
    actor_process = None
    trainer_process = None
    arena_process = None

    print(f"[*] Launching TeenyZero dashboard stack with runtime profile: {profile.name} on {runtime.device}")

    if not _port_in_use(5002):
        run_actors = _project_root() / "scripts" / "run_actors.py"
        actor_process = subprocess.Popen(
            [sys.executable, str(run_actors)],
            cwd=str(_project_root()),
        )

    run_trainer = _project_root() / "scripts" / "train.py"
    trainer_process = subprocess.Popen(
        [sys.executable, str(run_trainer)],
        cwd=str(_project_root()),
    )

    run_arena = _project_root() / "scripts" / "run_arena.py"
    arena_process = subprocess.Popen(
        [sys.executable, str(run_arena)],
        cwd=str(_project_root()),
    )

    try:
        app.run(debug=False, port=5001, host="0.0.0.0")
    finally:
        if actor_process is not None and actor_process.poll() is None:
            actor_process.terminate()
        if trainer_process is not None and trainer_process.poll() is None:
            trainer_process.terminate()
        if arena_process is not None and arena_process.poll() is None:
            arena_process.terminate()
