#!/usr/bin/env bash
# Install Isaac Lab 2.2.1 against the Isaac Sim 4.5 Python, without replacing
# the simulator's PyTorch. isaaclab.sh --install always forces torch==2.7.0+cu128,
# which needs driver 570+; this workstation is on 535.
set -euo pipefail

PYTHON="${ISAACSIM_ROOT_PATH}/python.sh"
ISAACLAB="${ISAACLAB_PATH}"

echo "[INFO] Installing Isaac Lab into ${ISAACLAB} with ${PYTHON}"

# Keep Isaac Sim's pip (24.0+nv1). Upgrading pip also upgrades packaging and
# breaks other Omniverse pins.
"${PYTHON}" -m pip install wheel
# setuptools 82+ dropped pkg_resources; flatdict==4.0.1 still imports it in setup.py.
"${PYTHON}" -m pip install "setuptools>=70,<81"
"${PYTHON}" -m pip install toml

# Pin torch/torchvision to whatever Isaac Sim 4.5 shipped.
"${PYTHON}" - <<'PY' > /tmp/isaac-torch-constraint.txt
import sys
import torch
import torchvision

print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
print(
    f"[INFO] Isaac Sim torch={torch.__version__} torchvision={torchvision.__version__}",
    file=sys.stderr,
    flush=True,
)
PY
echo "[INFO] pip constraint:"
cat /tmp/isaac-torch-constraint.txt

# isaaclab 2.2.1 declares torch>=2.7; install the packages without pulling a new torch.
for ext in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    echo "[INFO] editable install (no-deps): ${ext}"
    "${PYTHON}" -m pip install --no-deps --editable "${ISAACLAB}/source/${ext}"
done

# Non-torch deps from Isaac Lab 2.2.1 setup.py files. Constraint blocks any
# transitive torch/torchvision upgrade.
"${PYTHON}" -m pip install --constraint /tmp/isaac-torch-constraint.txt \
    "numpy<2" \
    "onnx>=1.18.0" \
    prettytable==3.3.0 \
    toml \
    gymnasium==1.2.0 \
    trimesh \
    "pyglet<2" \
    transformers \
    einops \
    warp-lang \
    pillow==11.2.1 \
    starlette==0.45.3 \
    pytest \
    pytest-mock \
    junitparser \
    flaky \
    "protobuf>=4.25.8,!=5.26.0" \
    tensorboard \
    scikit-learn \
    numba \
    hydra-core \
    h5py \
    moviepy \
    tomli \
    ipywidgets==8.1.5 \
    psutil \
    urdfdom-py \
    pyvista

# Isolated builds pull latest setuptools, which no longer has pkg_resources.
"${PYTHON}" -m pip install --no-build-isolation --constraint /tmp/isaac-torch-constraint.txt \
    flatdict==4.0.1

# rsl-rl is what UHAS train.py / play.py use. If the package requires torch>=2.7,
# fall back to --no-deps so we keep Isaac Sim's torch.
if ! "${PYTHON}" -m pip install --constraint /tmp/isaac-torch-constraint.txt "rsl-rl-lib==2.3.3"; then
    echo "[WARN] rsl-rl-lib could not be installed under the torch pin; installing --no-deps"
    "${PYTHON}" -m pip install --no-deps "rsl-rl-lib==2.3.3"
fi

# Optional Isaac Lab extras (gamepad HID, Pink IK, dex retargeting). Not used by
# UHAS in-hand training; ignore failures so the image still builds.
if ! "${PYTHON}" -m pip install --constraint /tmp/isaac-torch-constraint.txt \
        hidapi==0.14.0.post2 pin-pink==3.1.0 dex-retargeting==0.4.6; then
    echo "[WARN] Optional hidapi / pin-pink / dex-retargeting failed; skipping"
fi

# Official Isaac Lab docker does this; the solver is unused by UHAS.
"${PYTHON}" -m pip uninstall -y quadprog || true

echo "[INFO] Isaac Lab install finished"
# Do not `import isaaclab` here: it loads omni.physics, which only exists after Kit starts.
"${PYTHON}" -m pip show isaaclab isaaclab_rl isaaclab_tasks isaaclab_mimic
"${PYTHON}" -c "import torch; print('torch', torch.__version__)"
