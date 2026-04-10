"""Train an rl-games PPO policy to point the double pendulum upright."""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime
from math import gcd
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train the double-pendulum upright torque policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="DoublePendulum-Upright-Direct-v0", help="Gymnasium task name.")
parser.add_argument("--seed", type=int, default=None, help="Random seed. Use -1 for random.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO training epochs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--sigma", type=float, default=None, help="Initial policy standard deviation override.")
parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
parser.add_argument("--video_length", type=int, default=300, help="Training video length in env steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Training video interval in env steps.")
parser.add_argument(
    "--clone_in_fabric",
    action="store_true",
    help="Use Fabric cloning for faster headless training. Disable this when visually inspecting many envs.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: F401
from config.double_pendulum_upright_env_cfg import DoublePendulumUprightEnvCfg

AGENT_CFG_PATH = PROJECT_ROOT / "agents" / "rl_games_upright_ppo_cfg.yaml"


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
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.clone_in_fabric = args_cli.clone_in_fabric
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = seed
    return env_cfg


def apply_agent_overrides(agent_cfg: dict, seed: int, checkpoint: str | None) -> None:
    device = getattr(args_cli, "device", None)
    if device is not None:
        agent_cfg["params"]["config"]["device"] = device
        agent_cfg["params"]["config"]["device_name"] = device
    agent_cfg["params"]["seed"] = seed
    if checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint
    if args_cli.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations


def tune_ppo_batch_config(agent_cfg: dict, num_envs: int) -> None:
    config = agent_cfg["params"]["config"]
    batch_size = int(num_envs) * int(config["horizon_length"])
    minibatch_size = min(int(config["minibatch_size"]), batch_size)
    if batch_size % minibatch_size != 0:
        minibatch_size = gcd(batch_size, minibatch_size)
    config["minibatch_size"] = max(minibatch_size, 1)


def prepare_log_dir(agent_cfg: dict) -> tuple[Path, str]:
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = PROJECT_ROOT / "logs" / "rl_games" / config_name
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["params"]["config"]["train_dir"] = str(log_root_path.resolve())
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    return log_root_path, log_dir


def dump_configs(log_root_path: Path, log_dir: str, env_cfg, agent_cfg: dict) -> None:
    params_dir = log_root_path / log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
    dump_pickle(str(params_dir / "env.pkl"), env_cfg)
    dump_pickle(str(params_dir / "agent.pkl"), agent_cfg)


def create_rl_games_env(task: str, env_cfg, agent_cfg: dict, log_root_path: Path, log_dir: str):
    render_mode = "rgb_array" if args_cli.video else None
    base_env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
    gym_env = base_env

    if args_cli.video:
        video_folder = log_root_path / log_dir / "videos" / "train"
        gym_env = gym.wrappers.RecordVideo(
            gym_env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
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
    checkpoint = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint is not None else None

    env_cfg = load_env_cfg(seed)
    apply_agent_overrides(agent_cfg, seed, checkpoint)
    tune_ppo_batch_config(agent_cfg, env_cfg.scene.num_envs)
    log_root_path, log_dir = prepare_log_dir(agent_cfg)
    dump_configs(log_root_path, log_dir, env_cfg, agent_cfg)

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs}")
    print(f"[INFO] log_dir={(log_root_path / log_dir).resolve()}")
    print(f"[INFO] minibatch_size={agent_cfg['params']['config']['minibatch_size']}")

    env = create_rl_games_env(args_cli.task, env_cfg, agent_cfg, log_root_path, log_dir)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    run_args = {"train": True, "play": False, "sigma": args_cli.sigma}
    if checkpoint is not None:
        run_args["checkpoint"] = checkpoint
        print(f"[INFO] resume checkpoint={checkpoint}")
    runner.run(run_args)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
