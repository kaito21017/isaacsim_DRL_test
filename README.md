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

- Isaac Sim `4.5.x`
- Isaac Lab `v2.2.x`

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
観測は `[q1, q2, dq1, dq2]`、報酬は推定した先端位置が高いほど高く、トルク使用量の総和にペナルティを与えます。
先端位置が最高点から1度相当以内の高さを5秒維持できた場合は特別報酬を与え、各エピソードは10秒で終了します。

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless
```

デフォルトでは100並列環境で学習します。GUIで複数環境を見ながら確認する場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --num_envs 100
```

ヘッドレスで速度優先にする場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --clone_in_fabric
```

WandB に記録する場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --track --wandb-project-name double_pendulum_upright
```

KainaIsaacLab 側の実行形式に合わせて、`--use_wandb` と `--wandb_mode online/offline/disabled` も使えます。

ポリシー保存間隔を指定する場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --save_frequency 10
```

`save_frequency` は PPO epoch/update 単位です。デフォルトは50です。
保存名は `double_pendulum_upright_ep_<N>_rew_<R>.pth` です。
学習中の最新ポリシーと終了時点のポリシーは、常に `last.pth` に上書き保存されます。

rl-games が内部で作る `last_...pth` は最後に削除し、`last.pth` だけを残します。
best用の `double_pendulum_upright.pth` も新しいrunでは保存しません。

保存済みポリシーを10秒評価し、関節角度、推定先端位置、使用トルクのCSVとグラフを保存する場合:

```bash
./isaacsim.sh -p scripts/evaluate_upright_policy.py --headless --checkpoint logs/rl_games/double_pendulum_upright/<run>/nn/<checkpoint>.pth
```

`--checkpoint` を省略すると、`logs/rl_games/double_pendulum_upright/` 以下の最新 `.pth` を使います。
評価結果は `logs/eval/double_pendulum_upright/` に保存されます。
学習途中の全チェックポイントを順番に評価する場合:

```bash
./isaacsim.sh -p scripts/evaluate_upright_policy.py --headless --evaluate_all --checkpoint logs/rl_games/double_pendulum_upright/<run>
```

GUIで実時間再生したい場合は `--headless` を外して `--real_time` を付けます。

軽く動作確認する場合:

```bash
./isaacsim.sh -p scripts/train_upright_policy.py --headless --num_envs 8 --max_iterations 1
```

ログとチェックポイントは `logs/rl_games/double_pendulum_upright/` に保存されます。
