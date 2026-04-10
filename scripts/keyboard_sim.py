"""Manual torque control for the local double-pendulum URDF in Isaac Sim."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Manual torque control for the double pendulum.")
parser.add_argument("--torque_magnitude", type=float, default=0.3, help="Torque applied while a key is held.")
parser.add_argument("--initial_joint1", type=float, default=0.0, help="Initial first joint angle [rad].")
parser.add_argument("--initial_joint2", type=float, default=0.0, help="Initial second joint angle [rad].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import carb.input
import omni.appwindow
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from config.double_pendulum_cfg import DOUBLE_PENDULUM_CFG


JOINT1_NAME_PATTERNS = ["base_Revolute[-_]1", "joint1"]
JOINT2_NAME_PATTERNS = ["link1_Revolute[-_]2", "joint2"]


class KeyboardController:
    """Translate keyboard input to joint torques."""

    def __init__(self, torque_magnitude: float) -> None:
        self.torque_magnitude = torque_magnitude
        self._key_state = {"Q": False, "A": False, "W": False, "S": False}
        self._reset_requested = False

        app_window = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = app_window.get_keyboard()
        self._keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, self._on_keyboard_event)

        print("Double pendulum keyboard sim")
        print("  Q / A : first revolute joint +/- torque")
        print("  W / S : second revolute joint +/- torque")
        print("  R     : reset")
        print("  ESC   : close Isaac Sim")
        print(f"  torque magnitude = {self.torque_magnitude:.2f} Nm")

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        key_name = event.input.name

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in self._key_state:
                self._key_state[key_name] = True
            elif key_name == "R":
                self._reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE and key_name in self._key_state:
            self._key_state[key_name] = False

        return True

    def get_torques(self) -> tuple[float, float]:
        torque1 = 0.0
        torque2 = 0.0

        if self._key_state["Q"]:
            torque1 += self.torque_magnitude
        if self._key_state["A"]:
            torque1 -= self.torque_magnitude
        if self._key_state["W"]:
            torque2 += self.torque_magnitude
        if self._key_state["S"]:
            torque2 -= self.torque_magnitude

        return torque1, torque2

    def consume_reset(self) -> bool:
        if not self._reset_requested:
            return False
        self._reset_requested = False
        return True


def _find_first_joint(robot: Articulation, name_patterns: list[str]) -> list[int]:
    """Find one joint, allowing Isaac's URDF name sanitization from '-' to '_'."""
    for name_pattern in name_patterns:
        try:
            joint_ids, joint_names = robot.find_joints(name_pattern)
        except ValueError:
            continue
        if joint_ids:
            print(f"[INFO] matched joint pattern '{name_pattern}': {joint_names}")
            return joint_ids
    raise RuntimeError(f"Could not find any joint matching {name_patterns}. Available joints: {robot.joint_names}")


def _reset_robot(robot: Articulation, joint1_idx: list[int], joint2_idx: list[int]) -> None:
    """Reset the robot to the requested initial joint angles."""
    default_root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint1_idx] = args_cli.initial_joint1
    joint_pos[:, joint2_idx] = args_cli.initial_joint2

    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def main() -> None:
    """Run the manual simulation loop."""
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.0, 1.8], target=[0.0, 0.0, 1.0])

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(DOUBLE_PENDULUM_CFG.replace(prim_path="/World/Robot"))

    sim.reset()

    joint1_idx = _find_first_joint(robot, JOINT1_NAME_PATTERNS)
    joint2_idx = _find_first_joint(robot, JOINT2_NAME_PATTERNS)
    print(f"[INFO] joint names: {robot.joint_names}")
    print(f"[INFO] controlled joint ids: joint1={joint1_idx}, joint2={joint2_idx}")
    print(f"[INFO] initial joint angles: ({args_cli.initial_joint1:.3f}, {args_cli.initial_joint2:.3f}) rad")
    _reset_robot(robot, joint1_idx, joint2_idx)

    keyboard = KeyboardController(torque_magnitude=args_cli.torque_magnitude)

    sim_dt = sim.get_physics_dt()
    step_count = 0

    while simulation_app.is_running():
        if keyboard.consume_reset():
            _reset_robot(robot, joint1_idx, joint2_idx)
            print("[INFO] robot reset")

        torque1, torque2 = keyboard.get_torques()
        robot.set_joint_effort_target(torch.tensor([[torque1]], device=sim.device), joint_ids=joint1_idx)
        robot.set_joint_effort_target(torch.tensor([[torque2]], device=sim.device), joint_ids=joint2_idx)

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        step_count += 1
        if step_count % 60 == 0:
            theta1 = robot.data.joint_pos[0, joint1_idx[0]].item()
            theta2 = robot.data.joint_pos[0, joint2_idx[0]].item()
            omega1 = robot.data.joint_vel[0, joint1_idx[0]].item()
            omega2 = robot.data.joint_vel[0, joint2_idx[0]].item()
            print(
                f"[STATE] "
                f"q1={math.degrees(theta1):+7.1f} deg "
                f"q2={math.degrees(theta2):+7.1f} deg "
                f"dq1={omega1:+6.2f} rad/s "
                f"dq2={omega2:+6.2f} rad/s "
                f"tau=({torque1:+4.1f}, {torque2:+4.1f})"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
