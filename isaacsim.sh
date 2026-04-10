#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_SH="/home/kaito/KainaIsaacLab/isaaclab.sh"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] Isaac Lab launcher not found: ${ISAACLAB_SH}" >&2
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" && -x "/home/kaito/env_isaaclab/bin/python" ]]; then
    export VIRTUAL_ENV="/home/kaito/env_isaaclab"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

if [[ "${TERM:-}" == "dumb" ]]; then
    export TERM=xterm
fi

cd "${PROJECT_DIR}"
exec "${ISAACLAB_SH}" "$@"
