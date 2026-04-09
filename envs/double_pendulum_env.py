"""Double-pendulum DirectRLEnv task."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.double_pendulum_env_cfg import DoublePendulumEnvCfg


class DoublePendulumEnv(DirectRLEnv):
    """Swing-up and balance task for a torque-controlled double pendulum."""

    cfg: DoublePendulumEnvCfg

    def __init__(self, cfg: DoublePendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint1_dof_idx, _ = self.robot.find_joints(self.cfg.joint1_dof_name)
        self._joint2_dof_idx, _ = self.robot.find_joints(self.cfg.joint2_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._episode_sums = {
            "upright": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "angle_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "angular_vel": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "torque": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "alive": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "success_bonus": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _update_joint_state_cache(self) -> None:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.cfg.action_scale * torch.clamp(actions.clone(), -1.0, 1.0)

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(
            self.actions[:, 0].unsqueeze(dim=1),
            joint_ids=self._joint1_dof_idx,
        )
        self.robot.set_joint_effort_target(
            self.actions[:, 1].unsqueeze(dim=1),
            joint_ids=self._joint2_dof_idx,
        )

    def _get_observations(self) -> dict:
        self._update_joint_state_cache()

        theta1 = torch.nan_to_num(self.joint_pos[:, self._joint1_dof_idx[0]], nan=0.0, posinf=0.0, neginf=0.0)
        theta2 = torch.nan_to_num(self.joint_pos[:, self._joint2_dof_idx[0]], nan=0.0, posinf=0.0, neginf=0.0)
        omega1 = torch.nan_to_num(self.joint_vel[:, self._joint1_dof_idx[0]], nan=0.0, posinf=0.0, neginf=0.0)
        omega2 = torch.nan_to_num(self.joint_vel[:, self._joint2_dof_idx[0]], nan=0.0, posinf=0.0, neginf=0.0)
        omega1 = torch.clamp(omega1, -self.cfg.max_joint_velocity, self.cfg.max_joint_velocity)
        omega2 = torch.clamp(omega2, -self.cfg.max_joint_velocity, self.cfg.max_joint_velocity)

        obs = torch.stack(
            (
                torch.sin(theta1),
                torch.cos(theta1),
                torch.sin(theta2),
                torch.cos(theta2),
                omega1,
                omega2,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._update_joint_state_cache()

        joint1_pos = self.joint_pos[:, self._joint1_dof_idx[0]]
        joint2_pos = self.joint_pos[:, self._joint2_dof_idx[0]]
        joint1_vel = self.joint_vel[:, self._joint1_dof_idx[0]]
        joint2_vel = self.joint_vel[:, self._joint2_dof_idx[0]]

        (
            total_reward,
            rew_upright,
            rew_angle_error,
            rew_angular_vel,
            rew_torque,
            rew_alive,
            rew_success_bonus,
        ) = compute_rewards(
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_angle_error,
            self.cfg.rew_scale_angular_vel,
            self.cfg.rew_scale_torque,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_success_bonus,
            self.cfg.success_angle_threshold,
            self.cfg.success_velocity_threshold,
            joint1_pos,
            joint2_pos,
            joint1_vel,
            joint2_vel,
            self.actions,
        )

        self._episode_sums["upright"] += rew_upright
        self._episode_sums["angle_error"] += rew_angle_error
        self._episode_sums["angular_vel"] += rew_angular_vel
        self._episode_sums["torque"] += rew_torque
        self._episode_sums["alive"] += rew_alive
        self._episode_sums["success_bonus"] += rew_success_bonus

        return torch.nan_to_num(total_reward, nan=-100.0, posinf=-100.0, neginf=-100.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_joint_state_cache()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        invalid_state = torch.any(torch.isnan(self.joint_pos) | torch.isnan(self.joint_vel), dim=1)
        invalid_state |= torch.any(torch.isinf(self.joint_pos) | torch.isinf(self.joint_vel), dim=1)
        invalid_state |= torch.any(torch.abs(self.joint_vel) > self.cfg.max_joint_velocity, dim=1)
        invalid_state |= torch.any(torch.abs(self.joint_pos) > self.cfg.max_joint_angle, dim=1)
        return invalid_state, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if len(env_ids) > 0:
            episode_log = {}
            for name, values in self._episode_sums.items():
                episode_log[f"Episode_Reward/{name}"] = torch.mean(values[env_ids])
                values[env_ids] = 0.0
            self.extras["log"] = episode_log

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        joint_pos[:, self._joint1_dof_idx] += sample_uniform(
            self.cfg.initial_joint1_angle_range[0],
            self.cfg.initial_joint1_angle_range[1],
            joint_pos[:, self._joint1_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._joint2_dof_idx] += sample_uniform(
            self.cfg.initial_joint2_angle_range[0],
            self.cfg.initial_joint2_angle_range[1],
            joint_pos[:, self._joint2_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel[:, self._joint1_dof_idx] += sample_uniform(
            self.cfg.initial_joint_velocity_range[0],
            self.cfg.initial_joint_velocity_range[1],
            joint_vel[:, self._joint1_dof_idx].shape,
            joint_vel.device,
        )
        joint_vel[:, self._joint2_dof_idx] += sample_uniform(
            self.cfg.initial_joint_velocity_range[0],
            self.cfg.initial_joint_velocity_range[1],
            joint_vel[:, self._joint2_dof_idx].shape,
            joint_vel.device,
        )

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.actions[env_ids] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def compute_rewards(
    rew_scale_upright: float,
    rew_scale_angle_error: float,
    rew_scale_angular_vel: float,
    rew_scale_torque: float,
    rew_scale_alive: float,
    rew_scale_success_bonus: float,
    success_angle_threshold: float,
    success_velocity_threshold: float,
    joint1_pos: torch.Tensor,
    joint2_pos: torch.Tensor,
    joint1_vel: torch.Tensor,
    joint2_vel: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward function for swing-up and balance."""
    link1_error = torch.atan2(torch.sin(joint1_pos - math.pi), torch.cos(joint1_pos - math.pi))
    link2_error = torch.atan2(
        torch.sin(joint1_pos + joint2_pos - math.pi),
        torch.cos(joint1_pos + joint2_pos - math.pi),
    )

    rew_upright = rew_scale_upright * (
        -torch.cos(joint1_pos) - torch.cos(joint1_pos + joint2_pos)
    )
    rew_angle_error = rew_scale_angle_error * (torch.square(link1_error) + torch.square(link2_error))
    rew_angular_vel = rew_scale_angular_vel * (torch.square(joint1_vel) + torch.square(joint2_vel))
    rew_torque = rew_scale_torque * torch.sum(torch.square(actions), dim=-1)
    rew_alive = rew_scale_alive * torch.ones_like(joint1_pos)

    success_mask = (
        (torch.abs(link1_error) <= success_angle_threshold)
        & (torch.abs(link2_error) <= success_angle_threshold)
        & (torch.abs(joint1_vel) <= success_velocity_threshold)
        & (torch.abs(joint2_vel) <= success_velocity_threshold)
    )
    rew_success_bonus = rew_scale_success_bonus * success_mask.float()

    total_reward = (
        rew_upright
        + rew_angle_error
        + rew_angular_vel
        + rew_torque
        + rew_alive
        + rew_success_bonus
    )
    return (
        total_reward,
        rew_upright,
        rew_angle_error,
        rew_angular_vel,
        rew_torque,
        rew_alive,
        rew_success_bonus,
    )
