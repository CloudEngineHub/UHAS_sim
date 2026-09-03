# UHAS_sim image: Isaac Sim 4.5.0 + Isaac Lab 2.2.1.
# Pulling nvcr.io/nvidia/isaac-sim requires an NGC login:
#   docker login nvcr.io
#   username: $oauthtoken
#   password: <NGC API key>
# Running the image accepts the NVIDIA Omniverse EULA (ACCEPT_EULA=Y).

ARG ISAACSIM_VERSION=4.5.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

ARG ISAACSIM_VERSION=4.5.0
ARG ISAACLAB_VERSION=v2.2.1
ARG ISAACSIM_ROOT_PATH=/isaac-sim
ARG ISAACLAB_PATH=/workspace/isaaclab
ARG DOCKER_USER_HOME=/root

ENV ISAACSIM_VERSION=${ISAACSIM_VERSION} \
    ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH} \
    ISAACLAB_PATH=${ISAACLAB_PATH} \
    DOCKER_USER_HOME=${DOCKER_USER_HOME} \
    UHAS_PATH=/workspace/UHAS_sim \
    OMNI_KIT_ALLOW_ROOT=1 \
    ACCEPT_EULA=Y \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]
USER root

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libglib2.0-0 \
        libhidapi-hidraw0 \
        libhidapi-libusb0 \
        libusb-1.0-0 \
        ncurses-term \
        wget && \
    apt-get -y autoremove && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${ISAACLAB_VERSION}" \
        https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_PATH}" && \
    chmod +x "${ISAACLAB_PATH}/isaaclab.sh" && \
    ln -sf "${ISAACSIM_ROOT_PATH}" "${ISAACLAB_PATH}/_isaac_sim"

# Cache/log dirs that compose bind-mounts as named volumes.
RUN mkdir -p \
        "${ISAACSIM_ROOT_PATH}/kit/cache" \
        "${DOCKER_USER_HOME}/.cache/ov" \
        "${DOCKER_USER_HOME}/.cache/pip" \
        "${DOCKER_USER_HOME}/.cache/nvidia/GLCache" \
        "${DOCKER_USER_HOME}/.nv/ComputeCache" \
        "${DOCKER_USER_HOME}/.nvidia-omniverse/logs" \
        "${DOCKER_USER_HOME}/.local/share/ov/data" \
        "${DOCKER_USER_HOME}/Documents" \
        "${UHAS_PATH}"

COPY install_isaaclab.sh /tmp/install_isaaclab.sh
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    chmod +x /tmp/install_isaaclab.sh && \
    /tmp/install_isaaclab.sh && \
    rm /tmp/install_isaaclab.sh

# Apt deps declared in Isaac Lab extension.toml files (none on a stock 2.2.1 tree,
# but keep the same hook as NVIDIA's Dockerfile.base).
RUN --mount=type=cache,target=/var/cache/apt \
    "${ISAACLAB_PATH}/isaaclab.sh" -p "${ISAACLAB_PATH}/tools/install_deps.py" apt "${ISAACLAB_PATH}/source" && \
    apt-get -y autoremove && apt-get clean && \
    rm -rf /var/lib/apt/lists/* || true

# python.sh must be exec'd by path (a symlink from /usr/local/bin breaks its
# dirname logic). Wrappers make non-interactive `python train.py` work.
RUN printf '%s\n' '#!/bin/bash' "exec ${ISAACSIM_ROOT_PATH}/python.sh \"\$@\"" \
        > /usr/local/bin/python && chmod +x /usr/local/bin/python && \
    cp /usr/local/bin/python /usr/local/bin/python3 && \
    printf '%s\n' '#!/bin/bash' "exec ${ISAACSIM_ROOT_PATH}/python.sh -m pip \"\$@\"" \
        > /usr/local/bin/pip && chmod +x /usr/local/bin/pip && \
    cp /usr/local/bin/pip /usr/local/bin/pip3 && \
    echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> "${DOCKER_USER_HOME}/.bashrc" && \
    echo "export UHAS_PATH=${UHAS_PATH}" >> "${DOCKER_USER_HOME}/.bashrc" && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> "${DOCKER_USER_HOME}/.bashrc" && \
    echo "alias python=${ISAACSIM_ROOT_PATH}/python.sh" >> "${DOCKER_USER_HOME}/.bashrc" && \
    echo "alias python3=${ISAACSIM_ROOT_PATH}/python.sh" >> "${DOCKER_USER_HOME}/.bashrc" && \
    echo "alias pip='${ISAACSIM_ROOT_PATH}/python.sh -m pip'" >> "${DOCKER_USER_HOME}/.bashrc" && \
    echo "alias pip3='${ISAACSIM_ROOT_PATH}/python.sh -m pip'" >> "${DOCKER_USER_HOME}/.bashrc"

# Prefer the editable package over a repo-root folder of the same name.
COPY uhas_syspath.py uhas-sim.pth ${ISAACSIM_ROOT_PATH}/kit/python/lib/python3.10/site-packages/

COPY entrypoint.sh /usr/local/bin/uhas-entrypoint.sh
RUN chmod +x /usr/local/bin/uhas-entrypoint.sh

WORKDIR /workspace/UHAS_sim
ENTRYPOINT ["/usr/local/bin/uhas-entrypoint.sh"]
# Keep `compose up -d` alive so `compose exec uhas bash` has a running container.
CMD ["sleep", "infinity"]
