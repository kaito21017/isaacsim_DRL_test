# Double Pendulum Isaac Sim Test

このリポジトリに含まれるURDF

- `urdf/double_pendulum/double.urdf`

を、そのまま Isaac Sim / Isaac Lab 上で読み込んで確認するための最小構成です。  
実装方針は `/home/kaito/KainaIsaacLab` の `UrdfFileCfg + ArticulationCfg + InteractiveScene` 構成に合わせています。

## 追加したもの

- `config/double_pendulum_cfg.py`
  - URDF の絶対パス解決
  - 固定台座 (`fix_base=True`)
  - `base_Revolute-1`, `link1_Revolute-2` のトルク制御設定
- `scripts/urdf_import.py`
  - URDF を読み込んで表示だけ行う最小スクリプト
- `scripts/keyboard_sim.py`
  - `Q/A`, `W/S` で2関節にトルクを与える手動シミュレーション

## 前提

- Isaac Sim `5.0.x`
- Isaac Lab `v2.3.x` 相当
- Docker実行時は NVIDIA Container Toolkit

## 実行

### 1. URDF をそのまま表示

```bash
./isaacsim.sh -p scripts/urdf_import.py
```

別のURDFを指定したい場合:

```bash
./isaacsim.sh -p scripts/urdf_import.py --urdf_path /absolute/path/to/file.urdf
```

### 2. キーボードで手動操作

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

## 補足

- URDF 内のメッシュは `scale="0.001"` で mm から m に変換されているため、Isaac Lab 側では追加スケールをかけていません。
- 台座はワールド固定です。二重振り子としてその場で回転運動だけを観察できます。

## DRL 学習

二重振り子ができるだけ真上を向くように、2関節トルクを出力する PPO ポリシーを学習します。
観測は `[q1, q2, dq1, dq2]`、報酬は推定した先端位置が高いほど高く、トルク使用量、角速度、先端上下速度、アクション変化量にペナルティを与えます。
先端位置が最高点から1度相当以内の高さで、かつ両関節速度が小さい状態を5秒維持できた場合は特別報酬を与え、各エピソードは10秒で終了します。

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless
```

デフォルトでは100並列環境で学習し、25 PPO epochごとにcheckpointを保存します。保存間隔を変える場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --save_frequency 10
```

GUIで複数環境を見ながら確認する場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --num_envs 100
```

ヘッドレスで速度優先にする場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --clone_in_fabric
```

軽く動作確認する場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --num_envs 8 --max_iterations 1
```

ログとチェックポイントは `logs/rl_games/double_pendulum_upright/` に保存されます。

WandBに学習曲線を保存する場合:

```bash
wandb login
./isaacsim.sh -p scripts/train_upright_policy.py --headless --track --wandb_project_name double_pendulum_upright
```

オフライン保存だけ行う場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --track --wandb_mode offline
```

学習済みポリシーを実行する場合:

```bash
./isaacsim.sh -p scripts/evaluate.py --num_envs 1
```

特定のcheckpointを実行する場合:

```bash
./isaacsim.sh -p scripts/evaluate.py --checkpoint logs/rl_games/double_pendulum_upright/<run>/nn/double_pendulum_upright.pth
```

定期保存された最新checkpointを実行する場合:

```bash
./isaacsim.sh -p scripts/evaluate.py --use_last_checkpoint
```

## Docker

Isaac Lab入りのベースイメージに、このリポジトリだけを重ねる構成です。デフォルトのベースイメージは `nvcr.io/nvidia/isaac-lab:2.3.0` です。

```bash
docker/build.sh
```

学習をDockerで実行:

```bash
docker/run.sh -p scripts/train_upright_policy.py --headless --num_envs 100 --track
```

評価をDockerで実行:

```bash
docker/run.sh -p scripts/evaluate.py --num_envs 1 --use_last_checkpoint
```

Docker Composeを使う場合:

```bash
docker compose up --build
```
