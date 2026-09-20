"""Gym registration for the local peg-in-hole project."""

import gymnasium as gym

from . import agents
from .env_cfg import LocalInsertionRLEnvCfg, SpiralBaselineEnvCfg
from .envs import LocalInsertionRLEnv, SpiralBaselineEnv


def _register_once(env_id: str, entry_point: str, kwargs: dict) -> None:
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point=entry_point, disable_env_checker=True, kwargs=kwargs)


_register_once(
    "Isaac-LocalInsertion-Spiral-Direct-v0",
    "local_insertion:SpiralBaselineEnv",
    {"env_cfg_entry_point": SpiralBaselineEnvCfg},
)

_register_once(
    "Isaac-LocalInsertion-RL-Direct-v0",
    "local_insertion:LocalInsertionRLEnv",
    {
        "env_cfg_entry_point": LocalInsertionRLEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

__all__ = [
    "LocalInsertionRLEnv",
    "LocalInsertionRLEnvCfg",
    "SpiralBaselineEnv",
    "SpiralBaselineEnvCfg",
]
