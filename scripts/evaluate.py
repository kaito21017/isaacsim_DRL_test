"""Evaluate a trained rl-games policy for the local double pendulum."""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a trained double-pendulum upright torque policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="DoublePendulum-Upright-Direct-v0", help="Gymnasium task name.")
parser.add_argument("--seed", type=int, default=None, help="Random seed. Use -1 for random.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to play.")
parser.add_argument("--run_dir", type=str, default=None, help="Run directory under logs/rl_games/double_pendulum_upright.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="Use the latest periodic checkpoint instead of the best checkpoint when --checkpoint is omitted.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record a play video.")
parser.add_argument("--video_length", type=int, default=600, help="Video length in env steps.")
parser.add_argument("--real_time", action="store_true", default=True, help="Throttle simulation to real time.")
parser.add_argument("--no_real_time", action="store_false", dest="real_time", help="Run as fast as possible.")
parser.add_argument(
    "--clone_in_fabric",
    action="store_true",
    help="Use Fabric cloning for faster headless play. Disable this when visually inspecting many envs.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: F401
from config.double_pendulum_upright_env_cfg import DoublePendulumUprightEnvCfg

AGENT_CFG_PATH = PROJECT_ROOT / "agents" / "rl_games_upright_ppo_cfg.yaml"
LAST_CHECKPOINT_PATTERN = re.compile(r"_ep_([0-9]+)_")


def resolve_seed(requested_seed: int | None, fallback_seed: int) -> int:
    if requested_seed == -1:
        return random.randint(0, 10000)
    if requested_seed is None:
        return fallback_seed
    return requested_seed


def load_agent_cfg() -> dict:
    with AGENT_CFG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_env_cfg(seed: int):
    env_cfg = DoublePendulumUprightEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.clone_in_fabric = args_cli.clone_in_fabric
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = seed
    return env_cfg


def find_checkpoint(agent_cfg: dict) -> Path:
    if args_cli.checkpoint is not None:
        return Path(retrieve_file_path(args_cli.checkpoint)).resolve()

    config_name = agent_cfg["params"]["config"]["name"]
    log_root = PROJECT_ROOT / "logs" / "rl_games" / config_name
    if args_cli.run_dir is not None:
        run_path = log_root / args_cli.run_dir
    else:
        run_paths = sorted(path for path in log_root.glob("*") if (path / "nn").is_dir())
        if not run_paths:
            raise FileNotFoundError(f"No rl-games runs found under: {log_root}")
        run_path = run_paths[-1]

    nn_dir = run_path / "nn"
    if args_cli.use_last_checkpoint:
        checkpoints = sorted(nn_dir.glob("last_*.pth"), key=checkpoint_epoch)
        if not checkpoints:
            raise FileNotFoundError(f"No periodic checkpoints found under: {nn_dir}")
        return checkpoints[-1].resolve()

    best_checkpoint = nn_dir / f"{config_name}.pth"
    if best_checkpoint.exists():
        return best_checkpoint.resolve()

    checkpoints = sorted(nn_dir.glob("last_*.pth"), key=checkpoint_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under: {nn_dir}")
    return checkpoints[-1].resolve()


def checkpoint_epoch(path: Path) -> int:
    match = LAST_CHECKPOINT_PATTERN.search(path.name)
    if match is None:
        return -1
    return int(match.group(1))


def create_rl_games_env(task: str, env_cfg, agent_cfg: dict, checkpoint: Path):
    render_mode = "rgb_array" if args_cli.video else None
    base_env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
    gym_env = base_env

    if args_cli.video:
        video_folder = checkpoint.parents[1] / "videos" / "play"
        gym_env = gym.wrappers.RecordVideo(
            gym_env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    wrapped_env = RlGamesVecEnvWrapper(gym_env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env},
    )
    return wrapped_env


def main() -> None:
    agent_cfg = load_agent_cfg()
    seed = resolve_seed(args_cli.seed, agent_cfg["params"]["seed"])
    env_cfg = load_env_cfg(seed)
    checkpoint = find_checkpoint(agent_cfg)

    if getattr(args_cli, "device", None) is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = seed
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs}")
    print(f"[INFO] checkpoint={checkpoint}")

    env = create_rl_games_env(args_cli.task, env_cfg, agent_cfg, checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(checkpoint))
    agent.reset()

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    step = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for state in agent.states:
                    state[:, dones, :] = 0.0

        step += 1
        if args_cli.video and step >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
