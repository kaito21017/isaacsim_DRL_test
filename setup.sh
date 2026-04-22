#!/usr/bin/env bash
# =============================================================================
# setup.sh — まっさらな Ubuntu PC にこのプロジェクトの実行環境を構築する
#
# 前提:
#   - Ubuntu 22.04 以上 (GLIBC 2.35+)
#   - NVIDIA GPU + ドライバがインストール済み
#   - conda (Miniconda / Anaconda) がインストール済み
#
# 使い方:
#   chmod +x setup.sh
#   ./setup.sh
#
# 完了後:
#   conda activate isaacsim
#   ./isaacsim.sh -p scripts/train_upright_policy.py --headless
# =============================================================================
set -euo pipefail

# ---------- 設定 ----------
PYTHON_VERSION="3.11"
CONDA_ENV_NAME="isaacsim"
ISAACSIM_VERSION="5.0.0"
ISAACLAB_VERSION="v2.2.1"

# プロジェクトルートの絶対パス
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${PROJECT_DIR}/IsaacLab"

# ---------- カラー出力 ----------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------- 前提チェック ----------
info "前提条件をチェックしています..."

# conda の存在確認
if ! command -v conda &>/dev/null; then
    error "conda が見つかりません。Miniconda または Anaconda をインストールしてください。"
    echo "  https://docs.anaconda.com/miniconda/"
    exit 1
fi

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

# ---------- conda 環境の作成 ----------
# conda のシェル初期化を有効にする
eval "$(conda shell.bash hook)"

if conda env list | grep -qE "^${CONDA_ENV_NAME}\s"; then
    warn "conda 環境 '${CONDA_ENV_NAME}' は既に存在します。再利用します。"
else
    info "conda 環境 '${CONDA_ENV_NAME}' を作成しています (Python ${PYTHON_VERSION})..."
    conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

info "conda 環境 '${CONDA_ENV_NAME}' をアクティベートしています..."
conda activate "${CONDA_ENV_NAME}"

# Python バージョン確認
ACTUAL_PY_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
if [[ "${ACTUAL_PY_VERSION}" != "${PYTHON_VERSION}" ]]; then
    error "Python ${ACTUAL_PY_VERSION} が検出されました。${PYTHON_VERSION} が必要です。"
    exit 1
fi

info "Python $(python --version) を使用します"

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

# ---------- 環境変数の設定 ----------
info "環境変数を設定しています..."

# ISAACLAB_PATH をこのプロジェクトのローカルクローンに設定
export ISAACLAB_PATH="${ISAACLAB_DIR}"

# conda 環境のアクティベート時に自動で設定されるようにする
CONDA_ENV_DIR="$(conda info --envs | grep "^${CONDA_ENV_NAME}" | awk '{print $NF}')"
ACTIVATE_DIR="${CONDA_ENV_DIR}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_ENV_DIR}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/isaacsim_env.sh" << ACTIVATE_EOF
#!/usr/bin/env bash
# Isaac Sim / Isaac Lab 環境変数 (自動生成)
export ISAACLAB_PATH="${ISAACLAB_DIR}"
export ACCEPT_EULA=Y
ACTIVATE_EOF

cat > "${DEACTIVATE_DIR}/isaacsim_env.sh" << DEACTIVATE_EOF
#!/usr/bin/env bash
# Isaac Sim / Isaac Lab 環境変数のクリーンアップ (自動生成)
unset ISAACLAB_PATH
unset ACCEPT_EULA
DEACTIVATE_EOF

chmod +x "${ACTIVATE_DIR}/isaacsim_env.sh" "${DEACTIVATE_DIR}/isaacsim_env.sh"

# ---------- 動作確認 ----------
info "インストールの動作確認をしています..."

python -c "
import isaaclab
print(f'  isaaclab version: {isaaclab.__version__}')
" 2>/dev/null || warn "isaaclab のバージョン確認をスキップしました"

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
echo "  # 1. conda 環境をアクティベート"
echo "  conda activate ${CONDA_ENV_NAME}"
echo ""
echo "  # 2. URDF を表示"
echo "  cd ${PROJECT_DIR}"
echo "  ./isaacsim.sh -p scripts/urdf_import.py"
echo ""
echo "  # 3. キーボードで手動操作"
echo "  ./isaacsim.sh -p scripts/keyboard_sim.py"
echo ""
echo "  # 4. DRL 学習 (ヘッドレス)"
echo "  ./isaacsim.sh -p scripts/train_upright_policy.py --headless"
echo ""
echo "  # 5. 学習済みポリシーの評価"
echo "  ./isaacsim.sh -p scripts/evaluate_upright_policy.py --headless"
echo ""
