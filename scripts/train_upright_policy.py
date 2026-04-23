"""Train an rl-games PPO policy to point the double pendulum upright."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import datetime
from distutils.util import strtobool
from math import gcd
from pathlib import Path

# --livestream 3 のインターセプト (Native Stream)
use_native_stream = False
if "--livestream" in sys.argv:
    idx = sys.argv.index("--livestream")
    if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "3":
        use_native_stream = True
        sys.argv.pop(idx)  # remove --livestream
        sys.argv.pop(idx)  # remove "3"
        if "--headless" not in sys.argv:
            sys.argv.append("--headless")
        if "--enable_cameras" not in sys.argv:
            sys.argv.append("--enable_cameras")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train the double-pendulum upright torque policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="DoublePendulum-Upright-Direct-v0", help="Gymnasium task name.")
parser.add_argument("--seed", type=int, default=None, help="Random seed. Use -1 for random.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO training epochs.")
parser.add_argument("--save_frequency", type=int, default=None, help="Checkpoint save interval in PPO epochs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--sigma", type=float, default=None, help="Initial policy standard deviation override.")
parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
parser.add_argument("--video_length", type=int, default=300, help="Training video length in env steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Training video interval in env steps.")
parser.add_argument(
    "--track",
    type=lambda value: bool(strtobool(value)),
    default=False,
    nargs="?",
    const=True,
    help="Track this experiment with Weights & Biases.",
)
parser.add_argument("--use_wandb", action="store_true", dest="track", help="Alias for --track.")
parser.add_argument(
    "--wandb_project_name",
    "--wandb-project-name",
    type=str,
    default=None,
    help="Weights & Biases project name.",
)
parser.add_argument(
    "--wandb_entity",
    "--wandb-entity",
    type=str,
    default=None,
    help="Weights & Biases entity or team name.",
)
parser.add_argument("--wandb_name", "--wandb-name", type=str, default=None, help="Weights & Biases run name.")
parser.add_argument("--wandb_group", "--wandb-group", type=str, default=None, help="Weights & Biases group name.")
parser.add_argument(
    "--wandb_tags",
    "--wandb-tags",
    type=str,
    default="",
    help="Comma-separated Weights & Biases tags.",
)
parser.add_argument(
    "--wandb_mode",
    "--wandb-mode",
    choices=["online", "offline", "disabled"],
    default="online",
    help="Weights & Biases mode.",
)
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

if use_native_stream:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.kit.livestream.native")

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


class PolicyCheckpointObserver(IsaacAlgoObserver):
    """Save periodic policy files without rl-games' default 'last_' prefix."""

    def __init__(self, save_frequency: int):
        super().__init__()
        self.save_frequency = max(int(save_frequency), 0)
        self.algo = None

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        if self.save_frequency <= 0 or self.algo.global_rank != 0:
            return
        if epoch_num % self.save_frequency != 0 or self.algo.game_rewards.current_size <= 0:
            return

        mean_rewards = self.algo.game_rewards.get_mean()
        reward = float(mean_rewards[0].item())
        checkpoint_name = f"{self.algo.config['name']}_ep_{epoch_num}_rew_{format_checkpoint_value(reward)}"
        checkpoint_path = Path(self.algo.nn_dir) / checkpoint_name
        self.algo.save(str(checkpoint_path))
        self.save_latest_policy()
        print(f"[INFO] saved policy checkpoint: {checkpoint_path}.pth")

    def save_final_policy(self) -> None:
        self.save_latest_policy()

    def save_latest_policy(self) -> None:
        if self.algo is None or self.algo.global_rank != 0:
            return
        checkpoint_path = Path(self.algo.nn_dir) / "last"
        self.algo.save(str(checkpoint_path))
        print(f"[INFO] saved latest policy: {checkpoint_path}.pth")

    def remove_rl_games_last_prefix_files(self) -> None:
        if self.algo is None or self.algo.global_rank != 0:
            return
        for checkpoint_path in Path(self.algo.nn_dir).glob("last_*.pth"):
            checkpoint_path.unlink()
            print(f"[INFO] removed rl-games prefixed checkpoint: {checkpoint_path}")


def format_checkpoint_value(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


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
    if args_cli.save_frequency is not None:
        agent_cfg["params"]["config"]["save_frequency"] = args_cli.save_frequency


def configure_policy_saving(agent_cfg: dict) -> int:
    config = agent_cfg["params"]["config"]
    save_frequency = int(config.get("save_frequency", 0))
    config["policy_save_frequency"] = save_frequency
    config["save_frequency"] = 0
    config["save_best_after"] = 10**12
    return save_frequency


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


def maybe_init_wandb(config_name: str, log_dir: str, env_cfg, agent_cfg: dict):
    global_rank = int(os.getenv("RANK", "0"))
    if not args_cli.track or args_cli.wandb_mode == "disabled" or global_rank != 0:
        return None

    import wandb

    wandb_project = args_cli.wandb_project_name or config_name
    wandb_name = args_cli.wandb_name or log_dir
    wandb_tags = [tag.strip() for tag in args_cli.wandb_tags.split(",") if tag.strip()]
    wandb_tags += [f"envs_{env_cfg.scene.num_envs}", "rl_games", "PPO"]

    init_kwargs = {
        "project": wandb_project,
        "name": wandb_name,
        "group": args_cli.wandb_group,
        "tags": wandb_tags,
        "mode": args_cli.wandb_mode,
        "sync_tensorboard": True,
        "monitor_gym": True,
        "save_code": True,
        "config": {
            "seed": agent_cfg["params"]["seed"],
            "task": args_cli.task,
            "num_envs": env_cfg.scene.num_envs,
            "max_iterations": agent_cfg["params"]["config"]["max_epochs"],
            "agent_cfg": agent_cfg,
            "env_cfg": env_cfg.to_dict(),
        },
    }
    if args_cli.wandb_entity is not None:
        init_kwargs["entity"] = args_cli.wandb_entity

    run = wandb.init(**init_kwargs)
    print(f"[INFO] wandb enabled: project={wandb_project}, name={wandb_name}, mode={args_cli.wandb_mode}")
    return run


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
    policy_save_frequency = configure_policy_saving(agent_cfg)
    tune_ppo_batch_config(agent_cfg, env_cfg.scene.num_envs)
    log_root_path, log_dir = prepare_log_dir(agent_cfg)
    dump_configs(log_root_path, log_dir, env_cfg, agent_cfg)
    config_name = agent_cfg["params"]["config"]["name"]

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs}")
    print(f"[INFO] log_dir={(log_root_path / log_dir).resolve()}")
    print(f"[INFO] minibatch_size={agent_cfg['params']['config']['minibatch_size']}")
    print(f"[INFO] policy_save_frequency={policy_save_frequency}")

    env = create_rl_games_env(args_cli.task, env_cfg, agent_cfg, log_root_path, log_dir)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    checkpoint_observer = PolicyCheckpointObserver(policy_save_frequency)
    runner = Runner(checkpoint_observer)
    runner.load(agent_cfg)
    runner.reset()

    wandb_run = maybe_init_wandb(config_name, log_dir, env_cfg, agent_cfg)

    run_args = {"train": True, "play": False, "sigma": args_cli.sigma}
    if checkpoint is not None:
        run_args["checkpoint"] = checkpoint
        print(f"[INFO] resume checkpoint={checkpoint}")
    try:
        runner.run(run_args)
    finally:
        checkpoint_observer.save_final_policy()
        checkpoint_observer.remove_rl_games_last_prefix_files()
        if wandb_run is not None:
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
