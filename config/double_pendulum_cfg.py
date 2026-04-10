"""Isaac Lab articulation config for the local double-pendulum URDF."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_PATH = PROJECT_ROOT / "urdf" / "double_pendulum" / "double.urdf"
JOINT1_NAME_EXPR = r"base_Revolute[-_]1"
JOINT2_NAME_EXPR = r"link1_Revolute[-_]2"

DOUBLE_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(DEFAULT_URDF_PATH),
        fix_base=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={},
    ),
    actuators={
        "joint1": ImplicitActuatorCfg(
            joint_names_expr=[JOINT1_NAME_EXPR],
            effort_limit_sim=1.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.01,
            dynamic_friction=0.005,
            viscous_friction=0.0,
        ),
        "joint2": ImplicitActuatorCfg(
            joint_names_expr=[JOINT2_NAME_EXPR],
            effort_limit_sim=1.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.01,
            dynamic_friction=0.005,
            viscous_friction=0.0,
        ),
    },
)
