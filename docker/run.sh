#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-double-pendulum-isaaclab:latest}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${ISAACSIM_DOCKER_CACHE:-${HOME}/docker/isaac-sim}"

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${CACHE_ROOT}/cache/kit" \
    "${CACHE_ROOT}/cache/ov" \
    "${CACHE_ROOT}/cache/pip" \
    "${CACHE_ROOT}/cache/glcache" \
    "${CACHE_ROOT}/cache/computecache" \
    "${CACHE_ROOT}/logs" \
    "${CACHE_ROOT}/data" \
    "${CACHE_ROOT}/documents"

DOCKER_ARGS=(
    --rm
    -it
    --gpus all \
    --network host \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_MODE="${WANDB_MODE:-}" \
    -v "${PROJECT_DIR}/logs:/workspace/isaacsim_DRL_test/logs:rw" \
    -v "${CACHE_ROOT}/cache/kit:/isaac-sim/kit/cache:rw" \
    -v "${CACHE_ROOT}/cache/ov:/root/.cache/ov:rw" \
    -v "${CACHE_ROOT}/cache/pip:/root/.cache/pip:rw" \
    -v "${CACHE_ROOT}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "${CACHE_ROOT}/cache/computecache:/root/.nv/ComputeCache:rw" \
    -v "${CACHE_ROOT}/logs:/root/.nvidia-omniverse/logs:rw" \
    -v "${CACHE_ROOT}/data:/root/.local/share/ov/data:rw" \
    -v "${CACHE_ROOT}/documents:/root/Documents:rw"
)

if [[ -n "${DISPLAY:-}" ]]; then
    DOCKER_ARGS+=(-e "DISPLAY=${DISPLAY}" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw")
    XAUTHORITY_PATH="${XAUTHORITY:-${HOME}/.Xauthority}"
    if [[ -f "${XAUTHORITY_PATH}" ]]; then
        DOCKER_ARGS+=(-e XAUTHORITY=/root/.Xauthority -v "${XAUTHORITY_PATH}:/root/.Xauthority:ro")
    fi
fi

docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" "$@"
