# Docker

この Docker 構成は Isaac Sim 公式イメージをベースに Isaac Lab `v2.2.1` を入れ、このリポジトリを `/workspace/isaacsim_DRL_test` に配置します。

## 前提

- NVIDIA Driver と NVIDIA Container Toolkit が入っていること
- `nvcr.io/nvidia/isaac-sim:5.0.0` を pull できること
- NVIDIA Isaac Sim / Omniverse の EULA と配布条件を満たすこと

完成イメージを第三者へ再配布する場合、Isaac Sim ベースイメージを含むため NVIDIA 側のライセンス条件を確認してください。安全なのは、このリポジトリと Dockerfile を配布し、利用者側で build してもらう形です。

## Build

```bash
cd /home/kaito/isaacsim_DRL_test
docker build -f docker/Dockerfile -t isaacsim-drl-double-pendulum:latest .
```

Compose を使う場合:

```bash
cd /home/kaito/isaacsim_DRL_test/docker
cp .env.example .env
docker compose build
```

## 学習

```bash
cd /home/kaito/isaacsim_DRL_test/docker
docker compose run --rm double-pendulum \
  ./isaacsim.sh -p scripts/train_upright_policy.py --headless --num_envs 100
```

WandB を使う場合は `.env` に `WANDB_API_KEY` を設定してから実行します。

```bash
docker compose run --rm double-pendulum \
  ./isaacsim.sh -p scripts/train_upright_policy.py --headless --num_envs 100 --track
```

## 評価

```bash
cd /home/kaito/isaacsim_DRL_test/docker
docker compose run --rm double-pendulum \
  ./isaacsim.sh -p scripts/evaluate_upright_policy.py --headless
```

特定 checkpoint を評価する場合:

```bash
docker compose run --rm double-pendulum \
  ./isaacsim.sh -p scripts/evaluate_upright_policy.py --headless \
  --checkpoint logs/rl_games/double_pendulum_upright/<run>/nn/last.pth
```

## GUI 実行

GUI を出す場合はホスト側で X11 を許可してから `--headless` を外します。

```bash
xhost +local:root
cd /home/kaito/isaacsim_DRL_test/docker
docker compose run --rm double-pendulum \
  ./isaacsim.sh -p scripts/keyboard_sim.py
```

## Push

```bash
docker tag isaacsim-drl-double-pendulum:latest <registry>/<namespace>/isaacsim-drl-double-pendulum:latest
docker push <registry>/<namespace>/isaacsim-drl-double-pendulum:latest
```

## tar で配布

レジストリを使わずに渡す場合:

```bash
docker save isaacsim-drl-double-pendulum:latest | gzip > isaacsim-drl-double-pendulum_latest.tar.gz
```

受け取った側:

```bash
gunzip -c isaacsim-drl-double-pendulum_latest.tar.gz | docker load
```
