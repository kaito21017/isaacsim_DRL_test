"""Evaluate a trained rl-games policy and save double-pendulum time series."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a trained double-pendulum policy.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to latest saved .pth.")
parser.add_argument("--checkpoint_glob", type=str, default=None, help="Glob pattern for evaluating multiple checkpoints.")
parser.add_argument("--evaluate_all", action="store_true", default=False, help="Evaluate all saved policies in order.")
parser.add_argument("--task", type=str, default="DoublePendulum-Upright-Direct-v0", help="Gymnasium task name.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed. Use -1 for random.")
parser.add_argument("--duration", type=float, default=10.0, help="Evaluation duration [s].")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save CSV and plots.")
parser.add_argument("--initial_joint1", type=float, default=0.0, help="Initial first joint angle [rad].")
parser.add_argument("--initial_joint2", type=float, default=0.0, help="Initial second joint angle [rad].")
parser.add_argument("--initial_velocity", type=float, default=0.0, help="Initial joint velocity [rad/s].")
parser.add_argument(
    "--clone_in_fabric",
    action="store_true",
    help="Use Fabric cloning for faster headless evaluation. Disable this for visual inspection.",
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run at real-time speed if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: F401
from config.double_pendulum_upright_env_cfg import DoublePendulumUprightEnvCfg
from envs.double_pendulum_upright_env import compute_tip_position, wrap_to_pi

AGENT_CFG_PATH = PROJECT_ROOT / "agents" / "rl_games_upright_ppo_cfg.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_seed(requested_seed: int) -> int:
    if requested_seed == -1:
        return random.randint(0, 10000)
    return requested_seed


def checkpoint_sort_key(path: Path) -> tuple[int, int, float, str]:
    if path.stem == "last":
        return 2, 10**12, path.stat().st_mtime, path.name
    episode_match = re.search(r"episodes_(\d+)", path.stem)
    epoch_match = re.search(r"_ep(?:och)?_(\d+)", path.stem)
    episode = int(episode_match.group(1)) if episode_match else -1
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    if episode >= 0:
        return 0, episode, path.stat().st_mtime, path.name
    if epoch >= 0:
        return 1, epoch, path.stat().st_mtime, path.name
    return 3, -1, path.stat().st_mtime, path.name


def find_latest_run_dir(config_name: str) -> Path:
    log_root = PROJECT_ROOT / "logs" / "rl_games" / config_name
    if not log_root.exists():
        raise FileNotFoundError(f"No training log directory found at {log_root}.")
    run_dirs = sorted((path for path in log_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not run_dirs:
        raise FileNotFoundError(f"No training runs found under {log_root}.")
    return run_dirs[-1]


def find_latest_checkpoint(config_name: str) -> Path:
    log_root = PROJECT_ROOT / "logs" / "rl_games" / config_name
    checkpoints = sorted(log_root.glob("*/nn/*.pth"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {log_root}. Pass --checkpoint explicitly.")
    return checkpoints[-1]


def resolve_path_pattern(pattern: str) -> list[Path]:
    pattern_path = Path(pattern).expanduser()
    if not pattern_path.is_absolute():
        pattern = str((PROJECT_ROOT / pattern_path).resolve())
    paths = [Path(path).resolve() for path in glob.glob(pattern)]
    return sorted((path for path in paths if path.is_file()), key=checkpoint_sort_key)


def checkpoints_in_run(run_dir: Path) -> list[Path]:
    nn_dir = run_dir / "nn"
    checkpoints = [
        path
        for path in nn_dir.glob("*.pth")
        if path.stem == "last" or re.search(r"_ep(?:och)?_\d+", path.stem)
    ]
    if checkpoints:
        return sorted(checkpoints, key=checkpoint_sort_key)
    return sorted(nn_dir.glob("*.pth"), key=checkpoint_sort_key)


def run_dir_for_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "episode_checkpoints":
        return checkpoint_path.parents[2]
    if checkpoint_path.parent.name == "nn":
        return checkpoint_path.parents[1]
    return checkpoint_path.parent


def resolve_checkpoint_paths(agent_cfg: dict) -> list[Path]:
    if args_cli.checkpoint_glob is not None:
        checkpoints = resolve_path_pattern(args_cli.checkpoint_glob)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints matched: {args_cli.checkpoint_glob}")
        return checkpoints

    if args_cli.checkpoint is not None:
        checkpoint_path = Path(args_cli.checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        if any(char in str(checkpoint_path) for char in "*?[]"):
            checkpoints = resolve_path_pattern(str(checkpoint_path))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints matched: {args_cli.checkpoint}")
            return checkpoints
        if checkpoint_path.is_dir():
            if (checkpoint_path / "nn").is_dir():
                checkpoints = checkpoints_in_run(checkpoint_path)
            elif (checkpoint_path / "episode_checkpoints").is_dir():
                checkpoints = checkpoints_in_run(checkpoint_path.parent)
            else:
                checkpoints = sorted(checkpoint_path.rglob("*.pth"), key=checkpoint_sort_key)
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found under {checkpoint_path}.")
            return checkpoints if args_cli.evaluate_all else [checkpoints[-1]]
        checkpoint_path = Path(retrieve_file_path(str(checkpoint_path))).resolve()
        if args_cli.evaluate_all:
            run_dir = run_dir_for_checkpoint(checkpoint_path)
            checkpoints = checkpoints_in_run(run_dir)
            if checkpoints:
                return checkpoints
        return [checkpoint_path]

    config_name = agent_cfg["params"]["config"]["name"]
    if args_cli.evaluate_all:
        return [path.resolve() for path in checkpoints_in_run(find_latest_run_dir(config_name))]
    return [find_latest_checkpoint(config_name).resolve()]


def load_agent_cfg_for_checkpoint(checkpoint_path: Path) -> dict:
    run_agent_cfg = run_dir_for_checkpoint(checkpoint_path) / "params" / "agent.yaml"
    if run_agent_cfg.exists():
        return load_yaml(run_agent_cfg)
    return load_yaml(AGENT_CFG_PATH)


def load_env_cfg(seed: int) -> DoublePendulumUprightEnvCfg:
    env_cfg = DoublePendulumUprightEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.clone_in_fabric = args_cli.clone_in_fabric
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = seed
    env_cfg.initial_joint1_angle_range = [args_cli.initial_joint1, args_cli.initial_joint1]
    env_cfg.initial_joint2_angle_range = [args_cli.initial_joint2, args_cli.initial_joint2]
    env_cfg.initial_joint_velocity_range = [args_cli.initial_velocity, args_cli.initial_velocity]
    env_cfg.episode_length_s = max(args_cli.duration, env_cfg.episode_length_s)
    return env_cfg


def prepare_agent_cfg(agent_cfg: dict, checkpoint_path: Path, seed: int) -> None:
    device = getattr(args_cli, "device", None)
    if device is not None:
        agent_cfg["params"]["config"]["device"] = device
        agent_cfg["params"]["config"]["device_name"] = device
    agent_cfg["params"]["seed"] = seed
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint_path)


def create_batch_output_root(config_name: str, multiple_checkpoints: bool) -> Path | None:
    if not multiple_checkpoints:
        return None
    if args_cli.output_dir is not None:
        output_root = Path(args_cli.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_root = PROJECT_ROOT / "logs" / "eval" / config_name / f"{timestamp}_all_checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root.resolve()


def create_output_dir(
    checkpoint_path: Path,
    config_name: str,
    multiple_checkpoints: bool,
    batch_output_root: Path | None,
) -> Path:
    # チェックポイントの親ディレクトリ名（ラン名）とファイル名を結合してわかりやすくする
    # 例: 2026-04-23_03-13-01_last
    run_dir_name = checkpoint_path.parent.parent.name if checkpoint_path.parent.name == "nn" else checkpoint_path.parent.name
    policy_name = f"{run_dir_name}_{checkpoint_path.stem}"

    if args_cli.output_dir is not None:
        output_dir = Path(args_cli.output_dir)
        if multiple_checkpoints:
            output_dir = output_dir / policy_name
    elif batch_output_root is not None:
        output_dir = batch_output_root / policy_name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = PROJECT_ROOT / "logs" / "eval" / config_name / f"{timestamp}_{policy_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def create_env(env_cfg: DoublePendulumUprightEnvCfg, agent_cfg: dict):
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    wrapped_env = RlGamesVecEnvWrapper(base_env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env},
    )
    return wrapped_env


def first_scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().flatten()[0].cpu())
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def first_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().flatten()[0].cpu())
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


def collect_sample(env, time_s: float, reward, done) -> dict[str, float | bool]:
    base_env = env.unwrapped
    base_env._update_joint_state_cache()

    q1 = base_env.joint_pos[0, base_env._joint1_dof_idx[0]]
    q2 = base_env.joint_pos[0, base_env._joint2_dof_idx[0]]
    dq1 = base_env.joint_vel[0, base_env._joint1_dof_idx[0]]
    dq2 = base_env.joint_vel[0, base_env._joint2_dof_idx[0]]
    q1_wrapped = wrap_to_pi(q1)
    q2_wrapped = wrap_to_pi(q2)
    tip_horizontal, tip_height = compute_tip_position(
        q1,
        q2,
        base_env.cfg.link1_length,
        base_env.cfg.link2_length,
    )
    max_tip_height = base_env.cfg.link1_length + base_env.cfg.link2_length
    tip_height_ratio = torch.clamp((tip_height + max_tip_height) / (2.0 * max_tip_height), 0.0, 1.0)
    torque = base_env.actions[0]

    # 0〜360度（0〜2π）の範囲に変換して記録
    q1_rad = float(q1.detach().cpu()) % (2 * math.pi)
    q2_rad = float(q2.detach().cpu()) % (2 * math.pi)
    return {
        "time_s": time_s,
        "q1_rad": q1_rad,
        "q2_rad": q2_rad,
        "q1_deg": math.degrees(q1_rad),
        "q2_deg": math.degrees(q2_rad),
        "dq1_rad_s": float(dq1.detach().cpu()),
        "dq2_rad_s": float(dq2.detach().cpu()),
        "tip_horizontal_m": float(tip_horizontal.detach().cpu()),
        "tip_height_m": float(tip_height.detach().cpu()),
        "tip_height_ratio": float(tip_height_ratio.detach().cpu()),
        "torque1_nm": float(torque[0].detach().cpu()),
        "torque2_nm": float(torque[1].detach().cpu()),
        "reward": first_scalar(reward),
        "done": first_bool(done),
    }


def save_csv(rows: list[dict[str, float | bool]], output_path: Path) -> None:
    if not rows:
        raise RuntimeError("No evaluation rows were collected.")
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_time_series(rows: list[dict[str, float | bool]], output_dir: Path) -> None:
    time_s = [float(row["time_s"]) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(time_s, [float(row["q1_deg"]) for row in rows], label="joint1")
    plt.plot(time_s, [float(row["q2_deg"]) for row in rows], label="joint2")
    plt.xlabel("time [s]")
    plt.ylabel("joint angle [deg]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "joint_angles.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_s, [float(row["tip_horizontal_m"]) for row in rows], label="tip horizontal")
    plt.plot(time_s, [float(row["tip_height_m"]) for row in rows], label="tip height")
    plt.xlabel("time [s]")
    plt.ylabel("tip position [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tip_position.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_s, [float(row["torque1_nm"]) for row in rows], label="joint1 torque")
    plt.plot(time_s, [float(row["torque2_nm"]) for row in rows], label="joint2 torque")
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "torques.png", dpi=160)
    plt.close()


def evaluate_checkpoint(
    env,
    agent,
    checkpoint_path: Path,
    config_name: str,
    multiple_checkpoints: bool,
    batch_output_root: Path | None,
) -> None:
    output_dir = create_output_dir(checkpoint_path, config_name, multiple_checkpoints, batch_output_root)

    print(f"[INFO] checkpoint={checkpoint_path}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] duration={args_cli.duration:.3f}s")

    agent.restore(str(checkpoint_path))
    agent.reset()

    dt = env.unwrapped.step_dt
    num_steps = int(round(args_cli.duration / dt))
    rows = []
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    for step in range(num_steps):
        start_time = time.time()
        with torch.inference_mode():
            agent_obs = obs["obs"] if isinstance(obs, dict) else obs
            agent_obs = agent.obs_to_torch(agent_obs)
            actions = agent.get_action(agent_obs, is_deterministic=True)
        obs, rewards, dones, _ = env.step(actions.clone())
        rows.append(collect_sample(env, (step + 1) * dt, rewards, dones))
        if first_bool(dones) and (step + 1) < num_steps:
            print(f"[WARN] Environment ended early at t={(step + 1) * dt:.3f}s.")
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    csv_path = output_dir / "evaluation.csv"
    save_csv(rows, csv_path)
    plot_time_series(rows, output_dir)
    print(f"[INFO] saved csv={csv_path}")
    print(f"[INFO] saved plots={output_dir / 'joint_angles.png'}, {output_dir / 'tip_position.png'}, {output_dir / 'torques.png'}")


def main() -> None:
    default_agent_cfg = load_yaml(AGENT_CFG_PATH)
    checkpoint_paths = resolve_checkpoint_paths(default_agent_cfg)
    multiple_checkpoints = len(checkpoint_paths) > 1
    seed = resolve_seed(args_cli.seed)
    agent_cfg = load_agent_cfg_for_checkpoint(checkpoint_paths[0])
    env_cfg = load_env_cfg(seed)
    prepare_agent_cfg(agent_cfg, checkpoint_paths[0], seed)
    config_name = agent_cfg["params"]["config"]["name"]
    batch_output_root = create_batch_output_root(config_name, multiple_checkpoints)
    print(f"[INFO] evaluating {len(checkpoint_paths)} checkpoint(s)")
    if batch_output_root is not None:
        print(f"[INFO] batch_output_root={batch_output_root}")

    env = create_env(env_cfg, agent_cfg)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()

    try:
        for checkpoint_path in checkpoint_paths:
            evaluate_checkpoint(env, agent, checkpoint_path, config_name, multiple_checkpoints, batch_output_root)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
