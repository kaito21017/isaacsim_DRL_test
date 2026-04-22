#!/usr/bin/env bash
# =============================================================================
# isaacsim.sh — Isaac Lab の python.sh を経由してスクリプトを実行するランチャー
#
# ISAACLAB_PATH の検索順序:
#   1. 環境変数 ISAACLAB_PATH が既に設定されている場合はそれを使用
#   2. プロジェクト直下の .env ファイルに記載がある場合はそれを使用
#   3. プロジェクト直下の IsaacLab/ (setup.sh で自動クローンされる場所)
#
# 既存環境で使う場合は以下のいずれかで設定できます:
#   - export ISAACLAB_PATH=/path/to/IsaacLab してから実行
#   - プロジェクト直下に .env ファイルを作成:
#       echo "ISAACLAB_PATH=/home/kaito/KainaIsaacLab" > .env
# =============================================================================
set -euo pipefail

# プロジェクトルートディレクトリを取得
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# .env ファイルがあれば読み込む (ISAACLAB_PATH等を設定可能)
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # コメント行と空行を除いて export する
    set -a
    # shellcheck disable=SC1091
    source <(grep -v '^\s*#' "${PROJECT_DIR}/.env" | grep -v '^\s*$')
    set +a
fi

# ISAACLAB_PATH の自動検出
if [[ -z "${ISAACLAB_PATH:-}" ]]; then
    # プロジェクト直下の IsaacLab ディレクトリを探す (setup.sh で作成される)
    if [[ -x "${PROJECT_DIR}/IsaacLab/isaaclab.sh" ]]; then
        ISAACLAB_PATH="${PROJECT_DIR}/IsaacLab"
    else
        echo "[ERROR] Isaac Lab が見つかりません。" >&2
        echo "[ERROR] 以下のいずれかを実行してください:" >&2
        echo "  1. ./setup.sh を実行してセットアップする" >&2
        echo "  2. export ISAACLAB_PATH=/path/to/IsaacLab を設定する" >&2
        echo "  3. プロジェクト直下に .env ファイルを作成する:" >&2
        echo "     echo \"ISAACLAB_PATH=/path/to/IsaacLab\" > ${PROJECT_DIR}/.env" >&2
        exit 1
    fi
fi

export ISAACLAB_PATH

ISAACLAB_SH="${ISAACLAB_PATH}/isaaclab.sh"

# Isaac Lab ランチャーの存在確認
if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] Isaac Lab ランチャーが見つかりません: ${ISAACLAB_SH}" >&2
    echo "[ERROR] ISAACLAB_PATH を正しい Isaac Lab ディレクトリに設定してください。" >&2
    exit 1
fi

# ターミナル互換性の設定
if [[ "${TERM:-}" == "dumb" ]]; then
    export TERM=xterm
fi

# プロジェクトディレクトリに移動して Isaac Lab 経由で実行
cd "${PROJECT_DIR}"
exec "${ISAACLAB_SH}" "$@"
