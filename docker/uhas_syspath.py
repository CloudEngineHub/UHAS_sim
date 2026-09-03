"""Put the editable UHAS package ahead of cwd on sys.path.

The repo root contains a folder named ``sphere_ctrl_isaaclab/`` (scripts, source,
logs). If that root is cwd, Python treats it as a namespace package and hides
the real module under ``source/sphere_ctrl_isaaclab/sphere_ctrl_isaaclab/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def prefer_editable_package() -> None:
    root = Path(os.environ.get("UHAS_PATH", "/workspace/UHAS_sim"))
    real = root / "sphere_ctrl_isaaclab" / "source" / "sphere_ctrl_isaaclab"
    if not real.is_dir():
        return
    path = str(real)
    try:
        sys.path.remove(path)
    except ValueError:
        pass
    sys.path.insert(0, path)


prefer_editable_package()
