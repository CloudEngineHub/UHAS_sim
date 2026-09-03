#!/usr/bin/env bash
# Finish Docker Engine + NVIDIA Container Toolkit install on Ubuntu 20.04.
# The official get.docker.com script often dies mid-way on this machine because
# `apt-get update` fails on an unrelated repo (ROS / Slack / TeamViewer) and
# the script uses `set -e`.
#
# Usage (from a real terminal, it needs your sudo password):
#   sudo bash docker/install-host.sh
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Re-run as root: sudo bash $0" >&2
    exit 1
fi

TARGET_USER="${SUDO_USER:-felipe}"
export DEBIAN_FRONTEND=noninteractive

apt_update_one() {
    local list_file="$1"
    apt-get update \
        -o Dir::Etc::sourcelist="${list_file}" \
        -o Dir::Etc::sourceparts=- \
        -o APT::Get::List-Cleanup=0
}

echo "[1/5] Installing Docker Engine from the repo already in docker.list"
if [[ ! -f /etc/apt/sources.list.d/docker.list ]]; then
    echo "Missing /etc/apt/sources.list.d/docker.list" >&2
    echo "First run: curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && sudo sh /tmp/get-docker.sh --setup-repo" >&2
    exit 1
fi

apt_update_one /etc/apt/sources.list.d/docker.list
apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

echo "[2/5] Enabling docker.service and adding ${TARGET_USER} to the docker group"
systemctl enable --now docker
groupadd -f docker
usermod -aG docker "${TARGET_USER}"

echo "[3/5] Installing NVIDIA Container Toolkit"
install -d -m 0755 /usr/share/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt_update_one /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "[4/5] Smoke tests"
docker run --rm hello-world
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi

echo "[5/5] Done"
docker --version
docker compose version
echo
echo "Log out and back in (or run: newgrp docker) so group membership applies."
echo "Then: docker login nvcr.io   (username \$oauthtoken, password = NGC API key)"
echo "Then: docker compose -f docker/docker-compose.yml up -d --build"
