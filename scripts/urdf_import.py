"""Load the local double-pendulum URDF into Isaac Lab and keep it running."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_URDF_PATH = PROJECT_ROOT / "urdf" / "double_pendulum" / "double.urdf"

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Load the local double-pendulum URDF in Isaac Sim.")
parser.add_argument(
    "--urdf_path",
    type=Path,
    default=DEFAULT_URDF_PATH,
    help="Path to the URDF file. Defaults to the URDF stored in this repository.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from config.double_pendulum_cfg import DOUBLE_PENDULUM_CFG


class SceneCfg(InteractiveSceneCfg):
    """Simple scene used to preview the URDF."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = DOUBLE_PENDULUM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=DOUBLE_PENDULUM_CFG.spawn.replace(asset_path=str(args_cli.urdf_path.resolve())),
    )


def main() -> None:
    """Create the scene and keep simulating until Isaac Sim is closed."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view([2.5, 2.0, 1.8], [0.0, 0.0, 1.0])

    scene = InteractiveScene(SceneCfg(args_cli.num_envs, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    print(f"[INFO] Loaded URDF: {args_cli.urdf_path.resolve()}")
    print(f"[INFO] Joint names: {robot.joint_names}")

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        sim.step()
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
