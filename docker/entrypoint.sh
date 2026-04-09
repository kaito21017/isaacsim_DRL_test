#!/bin/bash

set -euo pipefail

cd "${PROJECT_PATH}"

ISAACLAB="${ISAACLAB_PATH}/isaaclab.sh"

case "${1:-train}" in
    train)
        shift || true
        exec "${ISAACLAB}" -p scripts/train.py \
            --task DoublePendulum-Direct-v0 \
            --headless \
            --num_envs 4096 \
            "$@"
        ;;
    eval)
        shift || true
        exec "${ISAACLAB}" -p scripts/eval.py \
            --task DoublePendulum-Direct-v0 \
            --headless \
            --use_last_checkpoint \
            "$@"
        ;;
    play)
        shift || true
        exec "${ISAACLAB}" -p scripts/play.py \
            --task DoublePendulum-Direct-v0 \
            --use_last_checkpoint \
            "$@"
        ;;
    keyboard)
        shift || true
        exec "${ISAACLAB}" -p scripts/keyboard_sim.py "$@"
        ;;
    shell)
        exec /bin/bash
        ;;
    *)
        exec "$@"
        ;;
esac
