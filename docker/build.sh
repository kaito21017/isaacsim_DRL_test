#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-double-pendulum-isaaclab:latest}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.0}"
ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"

docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "ISAACLAB_PATH=${ISAACLAB_PATH}" \
    -t "${IMAGE_NAME}" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
