"""Train a PPO policy for the double-pendulum swing-up task."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="二重振り子のRL学習スクリプト")
parser.add_argument("--video", action="store_true", default=False, help="学習中の動画を記録する")
parser.add_argument("--video_length", type=int, default=200, help="動画の長さ (ステップ数)")
parser.add_argument("--video_interval", type=int, default=2000, help="動画記録の間隔 (ステップ数)")
parser.add_argument("--num_envs", type=int, default=None, help="並列環境数")
parser.add_argument("--task", type=str, default="DoublePendulum-Direct-v0", help="タスク名")
parser.add_argument("--seed", type=int, default=None, help="乱数シード (-1 でランダム)")
parser.add_argument("--checkpoint", type=str, default=None, help="途中再開用のチェックポイント")
parser.add_argument("--sigma", type=float, default=None, help="初期方策標準偏差")
parser.add_argument("--max_iterations", type=int, default=None, help="学習イテレーション数")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.assets import retrieve_file_path

from common import (
    create_rl_games_env,
    create_runner,
    dump_configs,
    load_agent_cfg,
    load_env_cfg,
    prepare_log_dir,
    resolve_seed,
    apply_agent_overrides,
    tune_ppo_batch_config,
)


def main():
    """Train the rl-games PPO policy."""
    agent_cfg = load_agent_cfg()
    seed = resolve_seed(args_cli.seed, agent_cfg["params"]["seed"])

    env_cfg = load_env_cfg(
        num_envs=args_cli.num_envs,
        device=getattr(args_cli, "device", None),
        seed=seed,
    )

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint is not None else None

    apply_agent_overrides(
        agent_cfg,
        device=getattr(args_cli, "device", None),
        seed=seed,
        checkpoint=resume_path,
        max_iterations=args_cli.max_iterations,
    )
    tune_ppo_batch_config(agent_cfg, env_cfg.scene.num_envs)

    log_root_path, log_dir = prepare_log_dir(agent_cfg)
    dump_configs(log_root_path, log_dir, env_cfg, agent_cfg)
    print(f"[INFO] ログディレクトリ: {(log_root_path / log_dir).resolve()}")
    print(f"[INFO] minibatch_size={agent_cfg['params']['config']['minibatch_size']}")

    _, env = create_rl_games_env(
        args_cli.task,
        env_cfg,
        agent_cfg,
        video=args_cli.video,
        video_folder=log_root_path / log_dir / "videos" / "train",
        video_length=args_cli.video_length,
        video_interval=args_cli.video_interval,
    )
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = create_runner(agent_cfg, with_observer=True)
    runner.reset()

    run_args = {"train": True, "play": False, "sigma": args_cli.sigma}
    if resume_path is not None:
        run_args["checkpoint"] = resume_path
        print(f"[INFO] チェックポイントから再開: {resume_path}")
    runner.run(run_args)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
