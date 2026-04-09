"""二重振り子 Gymnasium 環境登録

Isaac Lab の環境としてGymnasiumに登録し、
gym.make("DoublePendulum-Direct-v0") で利用可能にする。
"""

import gymnasium as gym

from . import double_pendulum_env

##
# Gymnasium 環境登録
##

gym.register(
    id="DoublePendulum-Direct-v0",
    entry_point="envs.double_pendulum_env:DoublePendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "config.double_pendulum_env_cfg:DoublePendulumEnvCfg",
        "rl_games_cfg_entry_point": "agents:rl_games_ppo_cfg.yaml",
    },
)
