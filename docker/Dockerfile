ARG ISAACSIM_VERSION=4.5.0
ARG ISAACLAB_VERSION=v2.2.1

FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

SHELL ["/bin/bash", "-lc"]

LABEL maintainer="kaito"
LABEL description="Double-pendulum RL environment built on Isaac Sim + Isaac Lab"

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV OMNI_KIT_ALLOW_ROOT=1
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACLAB_PATH=/workspace/IsaacLab
ENV PROJECT_PATH=/workspace/double_pendulum
ENV PYTHONPATH=/workspace/double_pendulum:${PYTHONPATH}

USER root

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    tmux \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${ISAACLAB_VERSION} \
    https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH}

RUN ln -sfn ${ISAACSIM_PATH} ${ISAACLAB_PATH}/_isaac_sim \
    && chmod +x ${ISAACLAB_PATH}/isaaclab.sh

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml
RUN ${ISAACLAB_PATH}/isaaclab.sh --install rl_games

COPY . ${PROJECT_PATH}

WORKDIR ${PROJECT_PATH}

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["train"]
