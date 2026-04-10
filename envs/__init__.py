"""Gymnasium registration for the local double-pendulum RL tasks."""

from __future__ import annotations

import gymnasium as gym


gym.register(
    id="DoublePendulum-Upright-Direct-v0",
    entry_point="envs.double_pendulum_upright_env:DoublePendulumUprightEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "config.double_pendulum_upright_env_cfg:DoublePendulumUprightEnvCfg",
        "rl_games_cfg_entry_point": "agents:rl_games_upright_ppo_cfg.yaml",
    },
)
