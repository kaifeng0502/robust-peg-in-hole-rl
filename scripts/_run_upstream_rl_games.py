"""Run an Isaac Lab RL-Games script after registering this external task package."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def run(script_name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    isaaclab_root = Path(os.environ.get("ISAACLAB_PATH", "/workspace/IsaacLab"))
    upstream = isaaclab_root / "scripts" / "reinforcement_learning" / "rl_games" / script_name
    if not upstream.is_file():
        raise FileNotFoundError(f"Isaac Lab RL-Games script not found: {upstream}")

    source = upstream.read_text()
    marker = "import isaaclab_tasks  # noqa: F401"
    replacement = marker + "\nimport local_insertion  # noqa: F401"
    if source.count(marker) != 1:
        raise RuntimeError(f"Expected one registration marker in {upstream}")
    source = source.replace(marker, replacement)

    sys.argv[0] = str(upstream)
    namespace = {"__name__": "__main__", "__file__": str(upstream)}
    exec(compile(source, str(upstream), "exec"), namespace)
