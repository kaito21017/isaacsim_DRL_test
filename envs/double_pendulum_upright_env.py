"""DirectRLEnv for training the local double pendulum to maximize tip height."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.double_pendulum_upright_env_cfg import DoublePendulumUprightEnvCfg


class DoublePendulumUprightEnv(DirectRLEnv):
    """Swing up and balance the two-link tip near its maximum height."""

    cfg: DoublePendulumUprightEnvCfg

    def __init__(self, cfg: DoublePendulumUprightEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint1_dof_idx = self._find_joint(self.cfg.joint1_dof_name)
        self._joint2_dof_idx = self._find_joint(self.cfg.joint2_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._episode_sums = {
            "tip_height": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "torque": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "success_bonus": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        self._success_hold_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._success_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _find_joint(self, name_expr: str) -> list[int]:
        joint_ids, joint_names = self.robot.find_joints(name_expr)
        if not joint_ids:
            raise RuntimeError(f"Joint '{name_expr}' was not found. Available joints: {self.robot.joint_names}")
        print(f"[INFO] matched joint '{name_expr}': {joint_names}")
        return joint_ids

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

    # =========================================================================
    # 2. 行動 (Action) の定義・適用
    # =========================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # PPOから渡される行動 actions は [-1.0, 1.0] の範囲になります。
        # ここで action_scale（最大トルク）を掛けて、実際のトルク値に変換します。
        self.actions = self.cfg.action_scale * torch.clamp(actions.clone(), -1.0, 1.0)

    def _apply_action(self) -> None:
        # 変換したトルクを第1関節と第2関節に適用します。
        self.robot.set_joint_effort_target(self.actions[:, 0].unsqueeze(dim=1), joint_ids=self._joint1_dof_idx)
        self.robot.set_joint_effort_target(self.actions[:, 1].unsqueeze(dim=1), joint_ids=self._joint2_dof_idx)

    # =========================================================================
    # 1. 状態 (Observation / State) の定義
    # =========================================================================
    def _get_observations(self) -> dict:
        self._update_joint_state_cache()

        # 関節の角度 (q1, q2) と 角速度 (dq1, dq2) を取得
        q1 = torch.nan_to_num(self.joint_pos[:, self._joint1_dof_idx[0]], nan=0.0, posinf=0.0, neginf=0.0)
        q2 = torch.nan_to_num(self.joint_pos[:, self._joint2_dof_idx[0]], nan=0.0, posinf=0.0, neginf=0.0)
        dq1 = torch.nan_to_num(self.joint_vel[:, self._joint1_dof_idx[0]], nan=0.0, posinf=0.0, neginf=0.0)
        dq2 = torch.nan_to_num(self.joint_vel[:, self._joint2_dof_idx[0]], nan=0.0, posinf=0.0, neginf=0.0)
        
        # 速度制限でクリッピング
        dq1 = torch.clamp(dq1, -self.cfg.max_joint_velocity, self.cfg.max_joint_velocity)
        dq2 = torch.clamp(dq2, -self.cfg.max_joint_velocity, self.cfg.max_joint_velocity)

        # AIに渡す状態ベクトル（角度は -π〜π に収まるようにラップ）
        # 状態を追加したい場合（例: 先端の座標など）はここに列を追加します。
        obs = torch.stack((wrap_to_pi(q1), wrap_to_pi(q2), dq1, dq2), dim=-1)
        return {"policy": obs}

    # =========================================================================
    # 3. 報酬 (Reward) の計算
    # =========================================================================
    def _get_rewards(self) -> torch.Tensor:
        self._update_joint_state_cache()

        q1 = self.joint_pos[:, self._joint1_dof_idx[0]]
        q2 = self.joint_pos[:, self._joint2_dof_idx[0]]
        
        # 報酬関数の詳細は compute_rewards に分離されています。
        # - 先端が高ければ高いほどプラス報酬
        # - トルク(力)を使いすぎるとマイナス報酬
        # - 5秒間真上をキープできたらボーナス報酬
        rewards = compute_rewards(
            self.cfg.rew_scale_tip_height,
            self.cfg.rew_scale_torque,
            self.cfg.rew_scale_success_bonus,
            self.cfg.link1_length,
            self.cfg.link2_length,
            self.cfg.success_tip_height_ratio,
            self.cfg.success_hold_time_s,
            self.step_dt,
            q1,
            q2,
            self.actions,
            self._success_hold_buf,
            self._success_awarded,
        )
        total_reward, rew_tip_height, rew_torque, rew_success_bonus, success_hold, success_awarded = rewards

        # ログ用の変数に保存
        self._episode_sums["tip_height"] += rew_tip_height
        self._episode_sums["torque"] += rew_torque
        self._episode_sums["success_bonus"] += rew_success_bonus
        self._success_hold_buf = success_hold
        self._success_awarded = success_awarded

        return torch.nan_to_num(total_reward, nan=-100.0, posinf=-100.0, neginf=-100.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_joint_state_cache()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        invalid_state = torch.any(torch.isnan(self.joint_pos) | torch.isnan(self.joint_vel), dim=1)
        invalid_state |= torch.any(torch.isinf(self.joint_pos) | torch.isinf(self.joint_vel), dim=1)
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
        self._success_hold_buf[env_ids] = 0.0
        self._success_awarded[env_ids] = False

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def compute_rewards(
    rew_scale_tip_height: float,
    rew_scale_torque: float,
    rew_scale_success_bonus: float,
    link1_length: float,
    link2_length: float,
    success_tip_height_ratio: float,
    success_hold_time_s: float,
    step_dt: float,
    q1: torch.Tensor,
    q2: torch.Tensor,
    actions: torch.Tensor,
    success_hold: torch.Tensor,
    success_awarded: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward high scores when the estimated pendulum tip is high."""
    tip_height = compute_tip_height(q1, q2, link1_length, link2_length)
    max_tip_height = link1_length + link2_length
    normalized_tip_height = torch.clamp((tip_height + max_tip_height) / (2.0 * max_tip_height), 0.0, 1.0)

    success_mask = tip_height >= max_tip_height * success_tip_height_ratio
    success_hold = torch.where(success_mask, success_hold + step_dt, torch.zeros_like(success_hold))
    newly_successful = (success_hold >= success_hold_time_s) & (~success_awarded)
    success_awarded = success_awarded | newly_successful

    rew_tip_height = rew_scale_tip_height * normalized_tip_height
    rew_torque = rew_scale_torque * torch.sum(torch.abs(actions), dim=-1)
    rew_success_bonus = rew_scale_success_bonus * newly_successful.float()

    total_reward = rew_tip_height + rew_torque + rew_success_bonus
    return total_reward, rew_tip_height, rew_torque, rew_success_bonus, success_hold, success_awarded


def compute_tip_height(
    q1: torch.Tensor,
    q2: torch.Tensor,
    link1_length: float,
    link2_length: float,
) -> torch.Tensor:
    """Estimate tip height relative to the first joint from planar joint angles."""
    _, tip_height = compute_tip_position(q1, q2, link1_length, link2_length)
    return tip_height


def compute_tip_position(
    q1: torch.Tensor,
    q2: torch.Tensor,
    link1_length: float,
    link2_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate planar tip position relative to the first joint from joint angles."""
    tip_horizontal = link1_length * torch.sin(q1) + link2_length * torch.sin(q1 + q2)
    tip_height = -link1_length * torch.cos(q1) - link2_length * torch.cos(q1 + q2)
    return tip_horizontal, tip_height
