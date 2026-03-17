import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    target = Path(__file__).resolve().parents[1] / "run_arena.py"
    runpy.run_path(str(target), run_name="__main__")
