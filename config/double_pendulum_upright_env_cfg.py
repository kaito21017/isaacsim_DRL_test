"""DirectRLEnv configuration for training an upright double-pendulum policy."""

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
    """Config for a torque-controlled upright task."""

    decimation = 2
    episode_length_s = 8.0
    action_space = 2
    observation_space = 8
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

    success_angle_threshold = math.radians(10.0)
    success_velocity_threshold = 0.75
    max_joint_velocity = 50.0
    max_joint_angle = 10.0 * math.pi

    rew_scale_upright = 2.0
    rew_scale_angle_error = -0.2
    rew_scale_angular_vel = -0.01
    rew_scale_torque = -0.002
    rew_scale_alive = 0.05
    rew_scale_success_bonus = 1.0
