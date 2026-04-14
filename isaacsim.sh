#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-/home/kaito/KainaIsaacLab}"
ISAACLAB_SH="${ISAACLAB_PATH}/isaaclab.sh"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] Isaac Lab launcher not found: ${ISAACLAB_SH}" >&2
    echo "[ERROR] Set ISAACLAB_PATH to your Isaac Lab directory." >&2
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" && -z "${IN_DOCKER:-}" && -x "/home/kaito/env_isaaclab/bin/python" ]]; then
    export VIRTUAL_ENV="/home/kaito/env_isaaclab"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

if [[ "${TERM:-}" == "dumb" ]]; then
    export TERM=xterm
fi

cd "${PROJECT_DIR}"
exec "${ISAACLAB_SH}" "$@"
