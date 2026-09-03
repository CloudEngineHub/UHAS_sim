#!/usr/bin/env bash
# Bind-mount of this repo is the live source tree. Re-install the extension so
# Isaac Sim's Python can import sphere_ctrl_isaaclab (and pick up setup.py changes).
set -euo pipefail

PYTHON="${ISAACSIM_ROOT_PATH:-/isaac-sim}/python.sh"
UHAS="${UHAS_PATH:-/workspace/UHAS_sim}"
EXT="${UHAS}/sphere_ctrl_isaaclab/source/sphere_ctrl_isaaclab"

if [[ -f "${EXT}/setup.py" ]]; then
    echo "[INFO] Installing sphere_ctrl_isaaclab (editable) from ${EXT}"
    "${PYTHON}" -m pip install -q -e "${EXT}"
else
    echo "[WARN] ${EXT} not found; is UHAS_sim bind-mounted at ${UHAS}?"
fi

if [[ -d "${UHAS}" ]]; then
    cd "${UHAS}"
fi

if [[ "$#" -eq 0 ]]; then
    exec bash
fi
exec "$@"
