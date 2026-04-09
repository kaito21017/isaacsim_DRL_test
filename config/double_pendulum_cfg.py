"""二重振り子のアセット設定 (ArticulationCfg)

URDFファイルから二重振り子モデルを読み込み、
各関節にImplicitActuatorを設定する。
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# URDFファイルのパス (このファイルからの相対パス)
_URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "urdf",
    "double pendulum",
    "double",
    "double.urdf",
)

##
# 二重振り子のアセット設定
##

DOUBLE_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        scale=(2.5, 2.5, 2.5),
        # 台座を世界に固定するため fix_base=True
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None,
                damping=None,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    # 初期状態: 台座を地面から1.5mの高さに配置、関節は0 (下向き)
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        joint_pos={"joint1": 0.0, "joint2": 0.0},
    ),
    actuators={
        # 関節1のアクチュエータ（トルク制御）
        "joint1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            effort_limit_sim=2.0,    # トルク制限 [Nm] (軽量リンク~0.09kgに適合)
            stiffness=0.0,           # 位置剛性なし（純粋なトルク制御）
            damping=0.0,             # ダンピングなし
        ),
        # 関節2のアクチュエータ（トルク制御）
        "joint2_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint2"],
            effort_limit_sim=2.0,    # トルク制限 [Nm] (軽量リンク~0.09kgに適合)
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""二重振り子のArticulationCfg設定。台座固定、2関節トルク制御。"""
