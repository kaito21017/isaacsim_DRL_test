# syntax=docker/dockerfile:1.7

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION=5.0.0
FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION}

SHELL ["/bin/bash", "-lc"]

ARG ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git
ARG ISAACLAB_REF=v2.2.1
ARG ISAACSIM_ROOT_PATH=/isaac-sim
ARG ISAACLAB_PATH=/workspace/IsaacLab
ARG PROJECT_PATH=/workspace/isaacsim_DRL_test
ARG DOCKER_USER_HOME=/root

ENV ACCEPT_EULA=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH} \
    ISAACLAB_PATH=${ISAACLAB_PATH} \
    PROJECT_PATH=${PROJECT_PATH} \
    DOCKER_USER_HOME=${DOCKER_USER_HOME} \
    IN_DOCKER=1 \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=${PROJECT_PATH}

USER root

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libglib2.0-0 \
        ncurses-term \
        wget && \
    apt-get -y autoremove && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${ISAACLAB_REF}" "${ISAACLAB_REPO}" "${ISAACLAB_PATH}" || \
    (rm -rf "${ISAACLAB_PATH}" && \
     git clone "${ISAACLAB_REPO}" "${ISAACLAB_PATH}" && \
     cd "${ISAACLAB_PATH}" && \
     git checkout "${ISAACLAB_REF}")

RUN chmod +x "${ISAACLAB_PATH}/isaaclab.sh" && \
    ln -sf "${ISAACSIM_ROOT_PATH}" "${ISAACLAB_PATH}/_isaac_sim"

RUN "${ISAACLAB_PATH}/isaaclab.sh" -p -m ensurepip --upgrade && \
    "${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install --upgrade pip && \
    "${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install --no-cache-dir "setuptools<70.0.0" wheel toml && \
    "${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install --no-cache-dir --no-build-isolation flatdict==4.0.1

RUN --mount=type=cache,target=/var/cache/apt \
    "${ISAACLAB_PATH}/isaaclab.sh" -p "${ISAACLAB_PATH}/tools/install_deps.py" apt "${ISAACLAB_PATH}/source" && \
    apt-get -y autoremove && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
        "${ISAACSIM_ROOT_PATH}/kit/cache" \
        "${DOCKER_USER_HOME}/.cache/ov" \
        "${DOCKER_USER_HOME}/.cache/pip" \
        "${DOCKER_USER_HOME}/.cache/nvidia/GLCache" \
        "${DOCKER_USER_HOME}/.nv/ComputeCache" \
        "${DOCKER_USER_HOME}/.nvidia-omniverse/logs" \
        "${DOCKER_USER_HOME}/.local/share/ov/data" \
        "${DOCKER_USER_HOME}/Documents" && \
    touch /bin/nvidia-smi \
        /bin/nvidia-debugdump \
        /bin/nvidia-persistenced \
        /bin/nvidia-cuda-mps-control \
        /bin/nvidia-cuda-mps-server \
        /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    "${ISAACLAB_PATH}/isaaclab.sh" --install && \
    "${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install --no-cache-dir wandb matplotlib && \
    ("${ISAACLAB_PATH}/isaaclab.sh" -p -m pip uninstall -y quadprog || true)

WORKDIR ${PROJECT_PATH}
COPY . ${PROJECT_PATH}

RUN chmod +x "${PROJECT_PATH}/isaacsim.sh" && \
    mkdir -p "${PROJECT_PATH}/logs" "${PROJECT_PATH}/wandb" && \
    echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> "${DOCKER_USER_HOME}/.bashrc" && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> "${DOCKER_USER_HOME}/.bashrc" && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> "${DOCKER_USER_HOME}/.bashrc" && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> "${DOCKER_USER_HOME}/.bashrc"

CMD ["/bin/bash"]
