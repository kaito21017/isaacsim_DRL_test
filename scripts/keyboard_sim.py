"""二重振り子 キーボード操作シミュレータ

キーボードで各関節にトルクをかけて、二重振り子の動きを手動で操作・確認できる
インタラクティブシミュレータ。

操作方法:
    Q / A : 関節1 に 正 / 負 トルク
    W / S : 関節2 に 正 / 負 トルク
    R     : リセット (初期姿勢に戻す)
    ESC   : 終了

使用例:
    # GUIモードで起動 (通常はこちら)
    ./isaaclab.sh -p scripts/keyboard_sim.py

    # トルクの大きさを変更
    ./isaaclab.sh -p scripts/keyboard_sim.py --torque_magnitude 5.0
"""

"""Isaac Sim シミュレータを起動"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="二重振り子 キーボード操作シミュレータ")
parser.add_argument("--num_envs", type=int, default=1, help="環境数 (デフォルト: 1)")
parser.add_argument("--torque_magnitude", type=float, default=5.0, help="キー1回あたりのトルク [Nm]")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# GUIモードで起動 (headlessにしない)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下、シミュレータ起動後の処理"""

import torch
import os
import sys
import math

import carb
import carb.input
import omni.appwindow

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# パスを追加してプロジェクトモジュールをインポート
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.double_pendulum_cfg import DOUBLE_PENDULUM_CFG


class KeyboardController:
    """キーボード入力を管理するクラス

    キーの押下状態を追跡し、各関節へのトルク指令値を提供する。
    """

    def __init__(self, torque_magnitude: float = 5.0):
        """
        引数:
            torque_magnitude: キー押下時に適用するトルクの大きさ [Nm]
        """
        self.torque_magnitude = torque_magnitude

        # 各キーの押下状態
        self._key_state = {
            "Q": False,  # 関節1 正トルク
            "A": False,  # 関節1 負トルク
            "W": False,  # 関節2 正トルク
            "S": False,  # 関節2 負トルク
        }
        self._reset_requested = False

        # キーボードイベントの購読
        app_window = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = app_window.get_keyboard()
        self._keyboard_sub = input_interface.subscribe_to_keyboard_events(
            keyboard, self._on_keyboard_event
        )

        print("=" * 60)
        print("  二重振り子 キーボード操作シミュレータ")
        print("=" * 60)
        print(f"  トルク: {self.torque_magnitude} Nm")
        print("")
        print("  操作方法:")
        print("    Q / A : 関節1 に 正 / 負 トルク")
        print("    W / S : 関節2 に 正 / 負 トルク")
        print("    R     : リセット (初期姿勢に戻す)")
        print("    ESC   : 終了")
        print("=" * 60)

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        """キーボードイベントのコールバック"""
        # キー名の取得
        key_name = event.input.name

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # キー押下
            if key_name in self._key_state:
                self._key_state[key_name] = True
            elif key_name == "R":
                self._reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # キー解放
            if key_name in self._key_state:
                self._key_state[key_name] = False

        return True

    def get_torques(self) -> tuple[float, float]:
        """現在のキー押下状態に基づくトルク値を取得

        戻り値:
            (τ1, τ2): 各関節へのトルク [Nm]
        """
        # 関節1: Q(正) / A(負)
        torque1 = 0.0
        if self._key_state["Q"]:
            torque1 += self.torque_magnitude
        if self._key_state["A"]:
            torque1 -= self.torque_magnitude

        # 関節2: W(正) / S(負)
        torque2 = 0.0
        if self._key_state["W"]:
            torque2 += self.torque_magnitude
        if self._key_state["S"]:
            torque2 -= self.torque_magnitude

        return torque1, torque2

    def check_reset(self) -> bool:
        """リセットが要求されたかチェック"""
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False


def main():
    """メイン関数: シミュレーションループ"""

    # シミュレーションコンテキストの設定
    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)

    # カメラの設定 (二重振り子が見えるように)
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 1.5])

    # 地面の追加
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    # 照明の追加
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 二重振り子の作成
    robot_cfg = DOUBLE_PENDULUM_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    # シミュレーションの再生を開始
    sim.reset()

    # 関節インデックスの取得
    joint1_idx, _ = robot.find_joints("joint1")
    joint2_idx, _ = robot.find_joints("joint2")

    # キーボードコントローラーの初期化
    keyboard = KeyboardController(torque_magnitude=args_cli.torque_magnitude)

    # ステップカウンター
    step_count = 0
    print_interval = 60  # 情報表示の間隔 (ステップ数)

    # シミュレーションループ
    while simulation_app.is_running():
        # リセットチェック
        if keyboard.check_reset():
            print("\n[INFO] リセット中...")
            # デフォルト状態にリセット
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            root_state = robot.data.default_root_state.clone()

            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            print("[INFO] リセット完了")

        # キーボードからトルク値を取得
        torque1, torque2 = keyboard.get_torques()

        # トルクをテンソルに変換して適用
        torque1_tensor = torch.tensor([[torque1]], device=sim.device)
        torque2_tensor = torch.tensor([[torque2]], device=sim.device)

        robot.set_joint_effort_target(torque1_tensor, joint_ids=joint1_idx)
        robot.set_joint_effort_target(torque2_tensor, joint_ids=joint2_idx)

        # シミュレーションデータの書き込みとステップ実行
        robot.write_data_to_sim()
        sim.step()

        # シミュレーションデータの更新
        robot.update(sim_cfg.dt)

        # 定期的に状態情報を表示
        step_count += 1
        if step_count % print_interval == 0:
            joint_pos = robot.data.joint_pos[0]
            joint_vel = robot.data.joint_vel[0]

            theta1 = joint_pos[joint1_idx[0]].item()
            theta2 = joint_pos[joint2_idx[0]].item()
            omega1 = joint_vel[joint1_idx[0]].item()
            omega2 = joint_vel[joint2_idx[0]].item()

            # 角度を度に変換して表示
            theta1_deg = math.degrees(theta1)
            theta2_deg = math.degrees(theta2)

            print(
                f"  θ1={theta1_deg:+7.1f}° ω1={omega1:+6.2f} rad/s | "
                f"θ2={theta2_deg:+7.1f}° ω2={omega2:+6.2f} rad/s | "
                f"τ1={torque1:+5.1f} Nm  τ2={torque2:+5.1f} Nm"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
