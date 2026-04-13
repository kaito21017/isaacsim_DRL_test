"""DirectRLEnv configuration for training a tip-height double-pendulum policy."""

from __future__ import annotations

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .double_pendulum_cfg import DOUBLE_PENDULUM_CFG, JOINT1_NAME_EXPR, JOINT2_NAME_EXPR


@configclass
class DoublePendulumUprightEnvCfg(DirectRLEnvCfg):
    """Config for a torque-controlled tip-height task."""

    decimation = 2
    episode_length_s = 10.0
    action_space = 2
    observation_space = 4
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    viewer: ViewerCfg = ViewerCfg(eye=(12.0, 12.0, 9.0), lookat=(0.0, 0.0, 1.0))

    robot_cfg: ArticulationCfg = DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint1_dof_name = JOINT1_NAME_EXPR
    joint2_dof_name = JOINT2_NAME_EXPR

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # Actions are normalized by rl-games to [-1, 1], then scaled to joint torques [Nm].
    action_scale = 1.0

    initial_joint1_angle_range = [-0.35, 0.35]
    initial_joint2_angle_range = [-0.35, 0.35]
    initial_joint_velocity_range = [-0.1, 0.1]

    link1_length = 0.17
    link2_length = 0.17
    success_tip_height_ratio = math.cos(math.radians(1.0))
    success_velocity_threshold = 0.25
    success_hold_time_s = 5.0
    max_joint_velocity = 50.0
    max_joint_angle = 10.0 * math.pi

    rew_scale_tip_height = 5.0
    rew_scale_settle = 2.0
    rew_scale_angular_velocity = -0.02
    rew_scale_tip_velocity = -0.01
    rew_scale_torque = -0.002
    rew_scale_action_rate = -0.01
    rew_scale_success_bonus = 50.0
