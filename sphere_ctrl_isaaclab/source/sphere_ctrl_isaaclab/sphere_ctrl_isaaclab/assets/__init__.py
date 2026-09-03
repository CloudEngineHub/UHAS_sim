"""UHAS gripper / sphere assets.

``grippers/`` is a symlink to the repo-root ``grippers/`` tree (USD, URDF, sphere_cik.json).
"""

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent
IRVL_ASSET_PATH = str(ASSET_DIR)
