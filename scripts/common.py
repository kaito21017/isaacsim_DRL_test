"""Shared helpers for the standalone Isaac Lab scripts."""

from __future__ import annotations

import math
import random
import sys
from datetime import datetime
from math import gcd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import envs  # noqa: F401


DEFAULT_TASK = "DoublePendulum-Direct-v0"
AGENT_CFG_PATH = PROJECT_ROOT / "agents" / "rl_games_ppo_cfg.yaml"


def resolve_seed(requested_seed: int | None, fallback_seed: int) -> int:
    """Resolve the runtime seed, supporting rl-games' `-1 => random` convention."""
    if requested_seed == -1:
        return random.randint(0, 10000)
    if requested_seed is None:
        return fallback_seed
    return requested_seed


def load_env_cfg(num_envs: int | None = None, device: str | None = None, seed: int | None = None):
    """Create the environment config and apply CLI overrides."""
    from config.double_pendulum_env_cfg import DoublePendulumEnvCfg

    env_cfg = DoublePendulumEnvCfg()
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    if device is not None:
        env_cfg.sim.device = device
    if seed is not None:
        env_cfg.seed = seed
    return env_cfg


def load_agent_cfg() -> dict:
    """Load the rl-games PPO configuration."""
    with AGENT_CFG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_agent_overrides(
    agent_cfg: dict,
    *,
    device: str | None = None,
    seed: int | None = None,
    checkpoint: str | None = None,
    max_iterations: int | None = None,
) -> None:
    """Apply common runtime overrides to the rl-games config."""
    if device is not None:
        agent_cfg["params"]["config"]["device"] = device
        agent_cfg["params"]["config"]["device_name"] = device
    if seed is not None:
        agent_cfg["params"]["seed"] = seed
    if checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint
    if max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = max_iterations


def tune_ppo_batch_config(agent_cfg: dict, num_envs: int) -> None:
    """Ensure rl-games batch settings are valid for the requested environment count."""
    config = agent_cfg["params"]["config"]
    horizon_length = int(config["horizon_length"])
    batch_size = int(num_envs) * horizon_length
    minibatch_size = int(config["minibatch_size"])

    if minibatch_size > batch_size:
        minibatch_size = batch_size
    if batch_size % minibatch_size != 0:
        minibatch_size = gcd(batch_size, minibatch_size)
    if minibatch_size <= 0:
        minibatch_size = batch_size

    config["minibatch_size"] = minibatch_size


def prepare_log_dir(agent_cfg: dict) -> tuple[Path, str]:
    """Allocate a timestamped run directory and update the rl-games config."""
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = PROJECT_ROOT / "logs" / "rl_games" / config_name
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    agent_cfg["params"]["config"]["train_dir"] = str(log_root_path.resolve())
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    return log_root_path, log_dir


def dump_configs(log_root_path: Path, log_dir: str, env_cfg, agent_cfg: dict) -> None:
    """Persist the runtime configs next to the training outputs."""
    params_dir = log_root_path / log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
    dump_pickle(str(params_dir / "env.pkl"), env_cfg)
    dump_pickle(str(params_dir / "agent.pkl"), agent_cfg)


def resolve_checkpoint(
    agent_cfg: dict,
    checkpoint: str | None = None,
    *,
    use_last_checkpoint: bool = False,
) -> str:
    """Resolve a checkpoint path from an explicit path or the local log directory."""
    if checkpoint is not None:
        return retrieve_file_path(checkpoint)

    config_name = agent_cfg["params"]["config"]["name"]
    checkpoint_root = PROJECT_ROOT / "logs" / "rl_games" / config_name
    candidates = sorted(
        checkpoint_root.glob("*/nn/*.pth"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {checkpoint_root}. Run training first or pass --checkpoint."
        )

    if not use_last_checkpoint:
        best_name = f"{config_name}.pth"
        best_candidates = [path for path in candidates if path.name == best_name]
        if best_candidates:
            return str(best_candidates[-1].resolve())

    return str(candidates[-1].resolve())


def create_rl_games_env(
    task: str,
    env_cfg,
    agent_cfg: dict,
    *,
    video: bool = False,
    video_folder: Path | None = None,
    video_length: int = 200,
    video_interval: int = 2000,
    video_once: bool = False,
):
    """Create the Isaac Lab env and wrap it for rl-games."""
    render_mode = "rgb_array" if video else None
    base_env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
    gym_env = base_env

    if video:
        if video_folder is None:
            raise ValueError("video_folder must be provided when video recording is enabled.")
        video_kwargs = {
            "video_folder": str(video_folder),
            "step_trigger": (lambda step: step == 0)
            if video_once
            else (lambda step: step % video_interval == 0),
            "video_length": video_length,
            "disable_logger": True,
        }
        gym_env = gym.wrappers.RecordVideo(gym_env, **video_kwargs)

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

    return base_env, wrapped_env


def create_runner(agent_cfg: dict, *, with_observer: bool) -> Runner:
    """Instantiate and load an rl-games runner."""
    runner = Runner(IsaacAlgoObserver()) if with_observer else Runner()
    runner.load(agent_cfg)
    return runner


def unwrap_obs(obs):
    """Extract the policy observation tensor from rl-games wrapper outputs."""
    if isinstance(obs, dict):
        return obs.get("obs", obs)
    return obs
