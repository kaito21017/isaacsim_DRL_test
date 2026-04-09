"""二重振り子の環境設定 (DirectRLEnvCfg)

倒立二重振り子タスクの設定パラメータを定義する。
観測空間、行動空間、報酬スケール等を含む。
"""

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .double_pendulum_cfg import DOUBLE_PENDULUM_CFG


@configclass
class DoublePendulumEnvCfg(DirectRLEnvCfg):
    """二重振り子のRL環境設定"""

    # ----- 環境基本設定 -----
    decimation = 2                     # 制御頻度 = sim_dt × decimation
    episode_length_s = 10.0            # 1エピソードの長さ [秒]
    action_space = 2                   # 行動: 各関節へのトルク [τ1, τ2]
    observation_space = 6              # 観測: [sin(θ1), cos(θ1), sin(θ2), cos(θ2), ω1, ω2]
    state_space = 0                    # 状態空間 (使わない)

    # ----- シミュレーション設定 -----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,                    # シミュレーション時間刻み
        render_interval=decimation,
    )

    # ----- ロボット (二重振り子) 設定 -----
    robot_cfg: ArticulationCfg = DOUBLE_PENDULUM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    joint1_dof_name = "joint1"            # 関節1の名前
    joint2_dof_name = "joint2"            # 関節2の名前

    # ----- シーン設定 -----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,                 # 並列環境数
        env_spacing=4.0,               # 環境間の間隔
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # ----- アクションスケール -----
    action_scale = 1.0                 # アクション → トルクへのスケーリング [Nm]

    # ----- リセット設定 -----
    # リセット時の初期角度範囲 [rad] (0 = 下向き)
    initial_joint1_angle_range = [-0.5, 0.5]
    initial_joint2_angle_range = [-0.5, 0.5]
    initial_joint_velocity_range = [-0.25, 0.25]

    # ----- 成功判定 -----
    success_angle_threshold = math.radians(12.0)
    success_velocity_threshold = 1.0
    max_joint_velocity = 40.0
    max_joint_angle = 8.0 * math.pi

    # ----- 報酬スケール -----
    rew_scale_upright = 2.0            # 倒立ボーナス (cos項)
    rew_scale_angle_error = -0.25      # 倒立近傍での角度誤差ペナルティ
    rew_scale_angular_vel = -0.01      # 角速度ペナルティ
    rew_scale_torque = -0.001          # トルクペナルティ (省エネ)
    rew_scale_alive = 0.1              # 生存ボーナス
    rew_scale_success_bonus = 1.0      # 倒立維持ボーナス
