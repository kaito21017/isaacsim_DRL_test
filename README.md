# Double Pendulum — Isaac Sim DRL テスト

二重振り子を Isaac Sim / Isaac Lab 上でシミュレーションし、先端ができるだけ真上を向くように PPO (rl-games) で強化学習するプロジェクトです。

## クイックスタート（まっさらなPCの場合）

### 前提条件

| 項目 | 要件 |
|------|------|
| OS | Ubuntu 22.04 以上 |
| GPU | NVIDIA GPU (RTX 20xx 以上を推奨) |
| ドライバ | NVIDIA GPU ドライバがインストール済み (`nvidia-smi` で確認) |
| Python | [Miniconda](https://docs.anaconda.com/miniconda/) または Anaconda がインストール済み |
| ストレージ | 15GB 以上の空き容量 |

### 3ステップセットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/<your-username>/isaacsim_DRL_test.git
cd isaacsim_DRL_test

# 2. セットアップスクリプトを実行（Isaac Sim, Isaac Lab, 依存パッケージをすべて自動インストール）
chmod +x setup.sh
./setup.sh

# 3. conda 環境をアクティベートして学習開始
conda activate isaacsim
./isaacsim.sh -p scripts/train_upright_policy.py --headless
```

`setup.sh` は以下を自動で行います:

- conda 環境 `isaacsim` (Python 3.11) の作成
- Isaac Sim 5.0.0 の pip インストール
- Isaac Lab v2.2.1 のクローンとインストール
- プロジェクト固有の依存パッケージ (wandb, matplotlib 等) のインストール
- `conda activate isaacsim` 時に必要な環境変数を自動設定

> **Note:** 初回の `setup.sh` 実行時は Isaac Sim のダウンロード (7GB以上) があるため、回線速度によっては時間がかかります。

---

## URDF

- `urdf/double_pendulum/double.urdf`

URDF 内のメッシュは `scale="0.001"` で mm から m に変換されています。台座はワールド固定で、二重振り子としてその場で回転運動だけを観察できます。

---

## 使い方

> 以下のコマンドはすべて `conda activate isaacsim` 後に実行してください。

### URDF をそのまま表示

```bash
./isaacsim.sh -p scripts/urdf_import.py
```

別のURDFを指定したい場合:

```bash
./isaacsim.sh -p scripts/urdf_import.py --urdf_path /absolute/path/to/file.urdf
```

### キーボードで手動操作

```bash
./isaacsim.sh -p scripts/keyboard_sim.py
```

操作:

- `Q / A`: `base_Revolute-1` に正 / 負トルク
- `W / S`: `link1_Revolute-2` に正 / 負トルク
- `R`: 初期状態へリセット
- `ESC`: 終了

トルク値を変える場合:

```bash
./isaacsim.sh -p scripts/keyboard_sim.py --torque_magnitude 0.5
```

初期角度を変えて、無入力時の振り子挙動を見る場合:

```bash
./isaacsim.sh -p scripts/keyboard_sim.py --initial_joint1 1.0 --initial_joint2 0.0
```

---

## DRL 学習

二重振り子ができるだけ真上を向くように、2関節トルクを出力する PPO ポリシーを学習します。
観測は `[q1, q2, dq1, dq2]`、報酬は推定した先端位置が高いほど高く、トルク使用量の総和にペナルティを与えます。
先端位置が最高点から1度相当以内の高さを5秒維持できた場合は特別報酬を与え、各エピソードは10秒で終了します。

### 基本的な学習

```bash
# ヘッドレス (推奨・高速)
./isaacsim.sh -p scripts/train_upright_policy.py --headless

# GUIで確認しながら
./isaacsim.sh -p scripts/train_upright_policy.py --num_envs 100
```

### ヘッドレスで速度優先

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --clone_in_fabric
```

### WandB に記録する場合

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --track --wandb-project-name double_pendulum_upright
```

`--use_wandb` と `--wandb_mode online/offline/disabled` も使えます。

### ポリシー保存間隔を指定

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --save_frequency 10
```

`save_frequency` は PPO epoch/update 単位です。デフォルトは50です。
保存名は `double_pendulum_upright_ep_<N>_rew_<R>.pth` です。
学習中の最新ポリシーと終了時点のポリシーは、常に `last.pth` に上書き保存されます。

### 軽く動作確認

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --num_envs 8 --max_iterations 1
```

ログとチェックポイントは `logs/rl_games/double_pendulum_upright/` に保存されます。

---

## 評価

保存済みポリシーを10秒評価し、関節角度、推定先端位置、使用トルクのCSVとグラフを保存します。

```bash
./isaacsim.sh -p scripts/evaluate_upright_policy.py --headless \
  --checkpoint logs/rl_games/double_pendulum_upright/<run>/nn/<checkpoint>.pth
```

`--checkpoint` を省略すると、`logs/rl_games/double_pendulum_upright/` 以下の最新 `.pth` を使います。
評価結果は `logs/eval/double_pendulum_upright/` に保存されます。

学習途中の全チェックポイントを順番に評価する場合:

```bash
./isaacsim.sh -p scripts/evaluate_upright_policy.py --headless \
  --evaluate_all --checkpoint logs/rl_games/double_pendulum_upright/<run>
```

GUIで実時間再生したい場合は `--headless` を外して `--real_time` を付けます。

---

## プロジェクト構成

```
isaacsim_DRL_test/
├── setup.sh                    # ワンコマンドセットアップスクリプト
├── isaacsim.sh                 # Isaac Lab 経由の実行ランチャー
├── agents/
│   └── rl_games_upright_ppo_cfg.yaml   # rl-games PPO ハイパーパラメータ
├── config/
│   ├── double_pendulum_cfg.py          # URDF → ArticulationCfg
│   └── double_pendulum_upright_env_cfg.py  # DirectRLEnvCfg
├── envs/
│   └── double_pendulum_upright_env.py  # DirectRLEnv 実装
├── scripts/
│   ├── train_upright_policy.py         # rl-games PPO 学習
│   ├── evaluate_upright_policy.py      # 学習済みポリシー評価
│   ├── keyboard_sim.py                # キーボード手動操作
│   └── urdf_import.py                 # URDF 表示確認
├── urdf/
│   └── double_pendulum/
│       ├── double.urdf
│       └── meshes/
│           ├── base.stl
│           ├── link1.stl
│           └── link2.stl
├── docker/                     # Docker 環境 (オプション)
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   ├── .env.example
│   └── README.md
└── IsaacLab/                   # setup.sh で自動クローン (git管理外)
```

---

## 動作要件

| ソフトウェア | バージョン |
|-------------|-----------|
| Isaac Sim   | 5.0.0     |
| Isaac Lab   | v2.2.1    |
| Python      | 3.11      |
| PyTorch     | 2.7+      |
| CUDA        | 12.x      |

---

## Docker (オプション)

Isaac Sim 公式イメージをベースに Isaac Lab とこのリポジトリを入れた Docker イメージも作れます。

```bash
# ビルド
docker build -f docker/Dockerfile -t isaacsim-drl-double-pendulum:latest .

# 学習
cd docker
docker compose run --rm double-pendulum \
  ./isaacsim.sh -p scripts/train_upright_policy.py --headless --num_envs 100
```

詳細は `docker/README.md` を参照してください。
完成イメージを第三者へ再配布する場合は、Isaac Sim ベースイメージを含むため NVIDIA 側のライセンス条件を確認してください。
