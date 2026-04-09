# 二重振り子 DRL (Isaac Sim / Isaac Lab)

Isaac Sim + Isaac Lab 上で、二重振り子をトルク制御しながら swing-up して真上で維持する PPO タスクです。  
ローカル実行、学習済みポリシーの可視化、定量評価、Docker イメージ化まで一通り入っています。

## 構成

```text
isaacsim_DRL_test/
├── agents/
│   └── rl_games_ppo_cfg.yaml
├── config/
│   ├── double_pendulum_cfg.py
│   └── double_pendulum_env_cfg.py
├── envs/
│   ├── __init__.py
│   └── double_pendulum_env.py
├── scripts/
│   ├── common.py
│   ├── eval.py
│   ├── keyboard_sim.py
│   ├── play.py
│   └── train.py
├── utils/
│   └── double_pendulum.py
├── urdf/
│   └── double_pendulum.urdf
└── docker/
    ├── .env
    ├── Dockerfile
    ├── docker-compose.dev.yaml
    ├── docker-compose.yaml
    └── entrypoint.sh
```

## 対応バージョン

- Isaac Sim `4.5.x`
- Isaac Lab `v2.2.x`
- RL ライブラリ: `rl_games`

Isaac Lab の公式互換表では `v2.2.x` は Isaac Sim `4.5 / 5.0` 系対応です。

## タスク定義

- アセット: 固定台座 + 2 自由度回転関節 + 2 本リンク
- 行動: `joint1`, `joint2` へのトルク指令 `[τ1, τ2]`
- 観測: `[sin(q1), cos(q1), sin(q2), cos(q2), dq1, dq2]`
- 目標: 下向き近傍から振り上げて両リンクを真上に保つ
- 報酬: 倒立報酬 + 角度誤差ペナルティ + 角速度ペナルティ + トルクペナルティ + 生存報酬 + 成功ボーナス
- 成功判定: 両リンクがほぼ真上、かつ角速度が十分小さい状態を一定時間以上維持

## ローカル実行

### 1. 手動シミュレーション

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/keyboard_sim.py
```

キー操作:

- `Q / A`: 関節1 に正 / 負トルク
- `W / S`: 関節2 に正 / 負トルク
- `R`: リセット
- `ESC`: 終了

### 2. 学習

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/train.py --task DoublePendulum-Direct-v0 --headless
```

よく使うオプション:

```bash
# 並列環境数を変更
/path/to/IsaacLab/isaaclab.sh -p scripts/train.py --headless --num_envs 2048

# 途中再開
/path/to/IsaacLab/isaaclab.sh -p scripts/train.py --headless --checkpoint logs/rl_games/double_pendulum_direct/<RUN>/nn/last_double_pendulum_direct_ep_500_rew_xx.pth

# 学習動画も保存
/path/to/IsaacLab/isaaclab.sh -p scripts/train.py --headless --video --video_length 600
```

### 3. 可視化再生

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/play.py --checkpoint logs/rl_games/double_pendulum_direct/<RUN>/nn/double_pendulum_direct.pth
```

チェックポイントを省略して最新を使う場合:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/play.py --use_last_checkpoint
```

### 4. 定量評価

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/eval.py --headless --use_last_checkpoint --num_envs 64 --num_episodes 128
```

出力:

- 平均エピソード報酬
- 平均倒立維持率
- 平均最長倒立維持時間
- 成功率
- JSON レポート (`logs/rl_games/.../eval/<timestamp>.json`)

## Docker

### 1. ビルド

NVIDIA NGC ベースイメージを使うので、必要なら事前に `docker login nvcr.io` を実施してください。

```bash
docker compose -f docker/docker-compose.yaml build
```

### 2. 学習

```bash
docker compose -f docker/docker-compose.yaml run --rm train
```

カスタム引数:

```bash
docker compose -f docker/docker-compose.yaml run --rm train -- --num_envs 2048 --max_iterations 1000
```

### 3. 評価

```bash
docker compose -f docker/docker-compose.yaml run --rm eval -- --num_episodes 128
```

### 4. 可視化再生

ヘッドレスならそのまま、GUI を出したい場合は X11 を追加してください。

```bash
# headless replay
docker compose -f docker/docker-compose.yaml run --rm play -- --headless

# GUI replay (Linux/X11)
xhost +local:root
docker compose -f docker/docker-compose.yaml run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  play
```

### 5. 開発用 bind mount

ソースをホスト側から差し替えたい場合は dev override を重ねます。

```bash
docker compose \
  -f docker/docker-compose.yaml \
  -f docker/docker-compose.dev.yaml \
  run --rm train -- --num_envs 512
```

## イメージ配布

### レジストリに push

```bash
docker tag double-pendulum-drl:latest <registry>/double-pendulum-drl:latest
docker push <registry>/double-pendulum-drl:latest
```

### tar として配布

```bash
docker save double-pendulum-drl:latest | gzip > double-pendulum-drl.tar.gz
```

受け取り側:

```bash
gunzip -c double-pendulum-drl.tar.gz | docker load
docker compose -f docker/docker-compose.yaml run --rm eval
```

## 実装メモ

- 環境は Isaac Lab の `DirectRLEnv` パターンで実装
- `train.py` / `play.py` は Isaac Lab 公式の `rl_games` ワークフローに沿って構成
- `eval.py` は `rl_games` player API を用いて deterministic policy を直接評価
- Docker は self-contained image を基準にし、開発時だけ `docker-compose.dev.yaml` で bind mount
