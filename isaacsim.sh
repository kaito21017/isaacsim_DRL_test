#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-/home/kaito/KainaIsaacLab}"
ISAACLAB_SH="${ISAACLAB_SH:-${ISAACLAB_PATH}/isaaclab.sh}"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] Isaac Lab launcher not found: ${ISAACLAB_SH}" >&2
    exit 1
fi

DEFAULT_VENV="${ISAACLAB_VENV:-/home/kaito/env_isaaclab}"
if [[ -z "${VIRTUAL_ENV:-}" && -x "${DEFAULT_VENV}/bin/python" ]]; then
    export VIRTUAL_ENV="${DEFAULT_VENV}"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

if [[ "${TERM:-}" == "dumb" ]]; then
    export TERM=xterm
fi

cd "${PROJECT_DIR}"
exec "${ISAACLAB_SH}" "$@"
