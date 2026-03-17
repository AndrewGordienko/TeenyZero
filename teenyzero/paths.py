from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "var"
RUNTIME_ROOT = Path(os.environ.get("TEENYZERO_RUNTIME_ROOT", DEFAULT_RUNTIME_ROOT)).expanduser()
DATA_DIR = RUNTIME_ROOT / "data"
MODELS_DIR = RUNTIME_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
LATEST_MODEL_PATH = MODELS_DIR / "latest_model.pth"
REPLAY_BUFFER_PATH = DATA_DIR / "replay_buffer"
TRAINING_STATE_PATH = DATA_DIR / "training_state.json"
TRAINING_HISTORY_PATH = DATA_DIR / "training_history.json"
TRAINER_LOCK_PATH = DATA_DIR / "trainer.lock"
ARENA_STATE_PATH = DATA_DIR / "arena_state.json"
ARENA_HISTORY_PATH = DATA_DIR / "arena_history.json"
ARENA_MATCHES_PATH = DATA_DIR / "arena_matches.json"
ARENA_LOCK_PATH = DATA_DIR / "arena.lock"
MODEL_ARCHIVE_PATH = RUNTIME_ROOT / "models" / "archive"
LEGACY_BEST_MODEL_PATH = LEGACY_MODELS_DIR / "best_model.pth"
LEGACY_LATEST_MODEL_PATH = LEGACY_MODELS_DIR / "latest_model.pth"


def _migrate_legacy_checkpoint(legacy_path: Path, target_path: Path):
    if target_path.exists() or not legacy_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.replace(target_path)


def ensure_runtime_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_BUFFER_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_ARCHIVE_PATH.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_checkpoint(LEGACY_BEST_MODEL_PATH, BEST_MODEL_PATH)
    _migrate_legacy_checkpoint(LEGACY_LATEST_MODEL_PATH, LATEST_MODEL_PATH)


def runtime_paths_payload():
    return {
        "project_root": str(PROJECT_ROOT),
        "runtime_root": str(RUNTIME_ROOT),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "replay_buffer_path": str(REPLAY_BUFFER_PATH),
        "training_state_path": str(TRAINING_STATE_PATH),
        "training_history_path": str(TRAINING_HISTORY_PATH),
        "arena_state_path": str(ARENA_STATE_PATH),
        "arena_history_path": str(ARENA_HISTORY_PATH),
        "arena_matches_path": str(ARENA_MATCHES_PATH),
        "model_archive_path": str(MODEL_ARCHIVE_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "latest_model_path": str(LATEST_MODEL_PATH),
    }
