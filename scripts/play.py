"""Visualize a trained PPO policy on the double-pendulum task."""

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="二重振り子の推論スクリプト")
parser.add_argument("--num_envs", type=int, default=4, help="並列環境数")
parser.add_argument("--task", type=str, default="DoublePendulum-Direct-v0", help="タスク名")
parser.add_argument("--checkpoint", type=str, default=None, help="チェックポイントファイルのパス")
parser.add_argument("--use_last_checkpoint", action="store_true", help="最新チェックポイントを自動選択する")
parser.add_argument("--seed", type=int, default=None, help="乱数シード (-1 でランダム)")
parser.add_argument("--video", action="store_true", default=False, help="再生動画を保存する")
parser.add_argument("--video_length", type=int, default=600, help="保存する動画長 (ステップ数)")
parser.add_argument("--real_time", action="store_true", help="可能な範囲で実時間に合わせて再生する")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from common import (
    create_rl_games_env,
    create_runner,
    load_agent_cfg,
    load_env_cfg,
    resolve_checkpoint,
    resolve_seed,
    apply_agent_overrides,
    unwrap_obs,
)


def main():
    """Play a trained checkpoint."""
    agent_cfg = load_agent_cfg()
    seed = resolve_seed(args_cli.seed, agent_cfg["params"]["seed"])
    checkpoint_path = resolve_checkpoint(
        agent_cfg,
        checkpoint=args_cli.checkpoint,
        use_last_checkpoint=args_cli.use_last_checkpoint or args_cli.checkpoint is None,
    )

    env_cfg = load_env_cfg(
        num_envs=args_cli.num_envs,
        device=getattr(args_cli, "device", None),
        seed=seed,
    )
    apply_agent_overrides(
        agent_cfg,
        device=getattr(args_cli, "device", None),
        seed=seed,
        checkpoint=checkpoint_path,
    )

    video_dir = Path(checkpoint_path).resolve().parents[1] / "videos" / "play" if args_cli.video else None

    base_env, env = create_rl_games_env(
        args_cli.task,
        env_cfg,
        agent_cfg,
        video=args_cli.video,
        video_folder=video_dir,
        video_length=args_cli.video_length,
        video_once=True,
    )
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = create_runner(agent_cfg, with_observer=False)
    player = runner.create_player()
    player.restore(checkpoint_path)
    player.reset()

    print(f"[INFO] チェックポイント読み込み: {checkpoint_path}")

    obs = unwrap_obs(env.reset())
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    timestep = 0
    dt = base_env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs_tensor = player.obs_to_torch(obs)
            actions = player.get_action(obs_tensor, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            obs = unwrap_obs(obs)

            if player.is_rnn and player.states is not None and len(dones) > 0:
                done_ids = torch.nonzero(torch.as_tensor(dones), as_tuple=False).squeeze(-1)
                if done_ids.numel() > 0:
                    for state in player.states:
                        state[:, done_ids, :] = 0.0

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
