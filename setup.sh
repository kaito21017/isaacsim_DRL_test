#!/usr/bin/env bash
# =============================================================================
# setup.sh — まっさらな Ubuntu PC にこのプロジェクトの実行環境を構築する
#
# 前提:
#   - Ubuntu 22.04 以上 (GLIBC 2.35+)
#   - NVIDIA GPU + ドライバがインストール済み
#
# 使い方:
#   chmod +x setup.sh
#   ./setup.sh
#
# 完了後:
#   source venv/bin/activate
#   ./isaacsim.sh -p scripts/train_upright_policy.py --headless
# =============================================================================
set -euo pipefail

# ---------- 設定 ----------
PYTHON_VERSION="3.11"
ISAACSIM_VERSION="5.0.0"
ISAACLAB_VERSION="v2.2.1"

# プロジェクトルートの絶対パス
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
ISAACLAB_DIR="${PROJECT_DIR}/IsaacLab"

# ---------- カラー出力 ----------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------- 前提チェック ----------
info "前提条件をチェックしています..."

# NVIDIA GPU ドライバの確認
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi が見つかりません。NVIDIA GPU ドライバをインストールしてください。"
    exit 1
fi

# GLIBC バージョン確認
GLIBC_VERSION=$(ldd --version 2>&1 | head -n1 | grep -oP '\d+\.\d+$' || echo "0.0")
GLIBC_MAJOR=$(echo "$GLIBC_VERSION" | cut -d. -f1)
GLIBC_MINOR=$(echo "$GLIBC_VERSION" | cut -d. -f2)
if [[ "$GLIBC_MAJOR" -lt 2 ]] || [[ "$GLIBC_MAJOR" -eq 2 && "$GLIBC_MINOR" -lt 35 ]]; then
    error "GLIBC ${GLIBC_VERSION} が検出されました。Isaac Sim ${ISAACSIM_VERSION} には GLIBC 2.35 以上が必要です (Ubuntu 22.04+)。"
    exit 1
fi

info "前提条件 OK (GLIBC=${GLIBC_VERSION}, GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1))"

# ---------- Python 3.11 の確認・インストール ----------
PYTHON_CMD=""

# python3.11 が既にあるか確認
if command -v python3.11 &>/dev/null; then
    PYTHON_CMD="python3.11"
    info "Python 3.11 が見つかりました: $(python3.11 --version)"
else
    info "Python 3.11 が見つかりません。インストールします..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    PYTHON_CMD="python3.11"
    info "Python 3.11 をインストールしました"
fi

# venv モジュールの確認
if ! "${PYTHON_CMD}" -m venv --help &>/dev/null; then
    info "python3.11-venv をインストールしています..."
    sudo apt-get install -y python3.11-venv
fi

# ---------- venv の作成 ----------
if [[ -d "${VENV_DIR}" ]]; then
    warn "venv が既に存在します: ${VENV_DIR}"
    warn "既存の venv を再利用します。クリーンインストールする場合は先に削除してください。"
else
    info "Python 仮想環境を作成しています: ${VENV_DIR}"
    "${PYTHON_CMD}" -m venv "${VENV_DIR}"
fi

# venv をアクティベート
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Python バージョン最終確認
ACTUAL_PY_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
if [[ "${ACTUAL_PY_VERSION}" != "${PYTHON_VERSION}" ]]; then
    error "Python ${ACTUAL_PY_VERSION} が検出されました。${PYTHON_VERSION} が必要です。"
    exit 1
fi

info "Python $(python --version) (${VENV_DIR}) を使用します"

# ---------- Isaac Sim のインストール (pip) ----------
info "Isaac Sim ${ISAACSIM_VERSION} を pip でインストールしています..."
info "  (初回は 7GB 以上のダウンロードがあるため、回線速度によっては時間がかかります)"

pip install --upgrade pip
pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

info "Isaac Sim ${ISAACSIM_VERSION} のインストールが完了しました"

# ---------- Isaac Lab のクローンとインストール ----------
if [[ -d "${ISAACLAB_DIR}" ]]; then
    warn "Isaac Lab ディレクトリが既に存在します: ${ISAACLAB_DIR}"
    warn "再インストールをスキップします。クリーンインストールする場合は先に削除してください。"
else
    info "Isaac Lab ${ISAACLAB_VERSION} をクローンしています..."
    git clone --depth 1 --branch "${ISAACLAB_VERSION}" \
        https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi

info "Isaac Lab の依存関係をインストールしています..."

# setuptools の互換性修正 (pkg_resources が必要なパッケージ向け)
pip install --no-cache-dir "setuptools<70.0.0" wheel toml
pip install --no-cache-dir --no-build-isolation flatdict==4.0.1

# Isaac Lab 本体のインストール
cd "${ISAACLAB_DIR}"
chmod +x isaaclab.sh
./isaaclab.sh --install

info "Isaac Lab ${ISAACLAB_VERSION} のインストールが完了しました"

# ---------- プロジェクト固有の依存関係 ----------
cd "${PROJECT_DIR}"

info "プロジェクト固有の依存パッケージをインストールしています..."
pip install --no-cache-dir wandb matplotlib

# quadprog の互換性問題を回避
pip uninstall -y quadprog 2>/dev/null || true

# ---------- .env ファイルの作成 ----------
info ".env ファイルを作成しています..."
cat > "${PROJECT_DIR}/.env" << ENV_EOF
# Isaac Lab のパス (setup.sh で自動生成)
ISAACLAB_PATH=${ISAACLAB_DIR}
ENV_EOF

# ---------- 動作確認 ----------
info "インストールの動作確認をしています..."

python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# ---------- 完了 ----------
echo ""
echo "=============================================="
info "セットアップが完了しました！"
echo "=============================================="
echo ""
echo "使い方:"
echo ""
echo "  # 1. venv をアクティベート (毎回必要)"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "  # 2. 学習を開始"
echo "  cd ${PROJECT_DIR}"
echo "  ./isaacsim.sh -p scripts/train_upright_policy.py --headless"
echo ""
echo "  # 3. その他のコマンド"
echo "  ./isaacsim.sh -p scripts/keyboard_sim.py           # キーボード操作"
echo "  ./isaacsim.sh -p scripts/urdf_import.py            # URDF 表示"
echo "  ./isaacsim.sh -p scripts/evaluate_upright_policy.py --headless  # 評価"
echo ""
