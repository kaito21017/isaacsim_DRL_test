"""Evaluate a trained PPO policy on the double-pendulum task."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="二重振り子の評価スクリプト")
parser.add_argument("--num_envs", type=int, default=32, help="並列環境数")
parser.add_argument("--num_episodes", type=int, default=64, help="評価エピソード数")
parser.add_argument("--task", type=str, default="DoublePendulum-Direct-v0", help="タスク名")
parser.add_argument("--checkpoint", type=str, default=None, help="チェックポイントファイルのパス")
parser.add_argument("--use_last_checkpoint", action="store_true", help="最新チェックポイントを自動選択する")
parser.add_argument("--seed", type=int, default=None, help="乱数シード (-1 でランダム)")
parser.add_argument("--video", action="store_true", default=False, help="評価動画を保存する")
parser.add_argument("--video_length", type=int, default=600, help="保存する動画長 (ステップ数)")
parser.add_argument("--success_hold_steps", type=int, default=60, help="成功とみなす最小倒立維持ステップ数")
parser.add_argument("--report_json", type=str, default=None, help="評価結果 JSON の保存先")

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
from utils.double_pendulum import compute_upright_mask


def main():
    """Evaluate a trained checkpoint and save summary metrics."""
    agent_cfg = load_agent_cfg()
    seed = resolve_seed(args_cli.seed, agent_cfg["params"]["seed"])
    checkpoint_path = resolve_checkpoint(
        agent_cfg,
        checkpoint=args_cli.checkpoint,
        use_last_checkpoint=args_cli.use_last_checkpoint or args_cli.checkpoint is None,
    )
    checkpoint_path = Path(checkpoint_path).resolve()

    env_cfg = load_env_cfg(
        num_envs=args_cli.num_envs,
        device=getattr(args_cli, "device", None),
        seed=seed,
    )
    apply_agent_overrides(
        agent_cfg,
        device=getattr(args_cli, "device", None),
        seed=seed,
        checkpoint=str(checkpoint_path),
    )

    report_path = (
        Path(args_cli.report_json).resolve()
        if args_cli.report_json is not None
        else checkpoint_path.parents[1] / "eval" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    video_dir = checkpoint_path.parents[1] / "videos" / "eval" if args_cli.video else None

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
    player.restore(str(checkpoint_path))
    player.reset()

    base_task = base_env.unwrapped
    device = base_task.device
    dt = base_task.step_dt

    episode_returns = torch.zeros(base_task.num_envs, dtype=torch.float, device=device)
    episode_lengths = torch.zeros(base_task.num_envs, dtype=torch.long, device=device)
    upright_steps = torch.zeros(base_task.num_envs, dtype=torch.long, device=device)
    current_streak = torch.zeros(base_task.num_envs, dtype=torch.long, device=device)
    max_streak = torch.zeros(base_task.num_envs, dtype=torch.long, device=device)

    returns_history = []
    lengths_history = []
    upright_ratio_history = []
    max_streak_history = []
    success_history = []

    obs = unwrap_obs(env.reset())
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    while simulation_app.is_running() and len(returns_history) < args_cli.num_episodes:
        with torch.inference_mode():
            obs_tensor = player.obs_to_torch(obs)
            actions = player.get_action(obs_tensor, is_deterministic=True)
            obs, rewards, dones, _ = env.step(actions)
            obs = unwrap_obs(obs)

        reward_tensor = torch.as_tensor(rewards, device=device, dtype=torch.float).view(-1)
        done_tensor = torch.as_tensor(dones, device=device, dtype=torch.bool).view(-1)

        joint1_pos = base_task.joint_pos[:, base_task._joint1_dof_idx[0]]
        joint2_pos = base_task.joint_pos[:, base_task._joint2_dof_idx[0]]
        joint1_vel = base_task.joint_vel[:, base_task._joint1_dof_idx[0]]
        joint2_vel = base_task.joint_vel[:, base_task._joint2_dof_idx[0]]
        upright_mask = compute_upright_mask(
            joint1_pos,
            joint2_pos,
            joint1_vel,
            joint2_vel,
            angle_threshold=base_task.cfg.success_angle_threshold,
            velocity_threshold=base_task.cfg.success_velocity_threshold,
        )

        episode_returns += reward_tensor
        episode_lengths += 1
        upright_steps += upright_mask.long()
        current_streak = torch.where(upright_mask, current_streak + 1, torch.zeros_like(current_streak))
        max_streak = torch.maximum(max_streak, current_streak)

        done_ids = torch.nonzero(done_tensor, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            for env_id in done_ids.tolist():
                if len(returns_history) >= args_cli.num_episodes:
                    break

                ep_length = int(episode_lengths[env_id].item())
                ep_return = float(episode_returns[env_id].item())
                ep_upright_ratio = float(upright_steps[env_id].item()) / max(ep_length, 1)
                ep_max_streak = int(max_streak[env_id].item())
                ep_success = ep_max_streak >= args_cli.success_hold_steps

                returns_history.append(ep_return)
                lengths_history.append(ep_length)
                upright_ratio_history.append(ep_upright_ratio)
                max_streak_history.append(ep_max_streak)
                success_history.append(ep_success)

            episode_returns[done_ids] = 0.0
            episode_lengths[done_ids] = 0
            upright_steps[done_ids] = 0
            current_streak[done_ids] = 0
            max_streak[done_ids] = 0

            if player.is_rnn and player.states is not None:
                for state in player.states:
                    state[:, done_ids, :] = 0.0

    summary = {
        "checkpoint": str(checkpoint_path),
        "num_episodes": len(returns_history),
        "num_envs": int(base_task.num_envs),
        "step_dt": float(dt),
        "success_hold_steps": int(args_cli.success_hold_steps),
        "success_hold_seconds": float(args_cli.success_hold_steps * dt),
        "mean_return": float(sum(returns_history) / max(len(returns_history), 1)),
        "mean_episode_length": float(sum(lengths_history) / max(len(lengths_history), 1)),
        "mean_upright_ratio": float(sum(upright_ratio_history) / max(len(upright_ratio_history), 1)),
        "mean_max_upright_streak": float(sum(max_streak_history) / max(len(max_streak_history), 1)),
        "mean_max_upright_seconds": float(sum(max_streak_history) / max(len(max_streak_history), 1) * dt),
        "success_rate": float(sum(1.0 for flag in success_history if flag) / max(len(success_history), 1)),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO] チェックポイント: {checkpoint_path}")
    print(f"[INFO] 評価結果JSON: {report_path}")
    print(
        "[INFO] "
        f"episodes={summary['num_episodes']} "
        f"return={summary['mean_return']:.3f} "
        f"upright_ratio={summary['mean_upright_ratio']:.3f} "
        f"max_upright={summary['mean_max_upright_seconds']:.3f}s "
        f"success_rate={summary['success_rate']:.3f}"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
