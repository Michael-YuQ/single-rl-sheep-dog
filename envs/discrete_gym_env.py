"""
Gymnasium wrapper around GeneralizedPrimitiveEnv for PPO/A2C (discrete macro-actions).

This gives a fair comparison with DreamerV3: same 22 macro-actions, same 10-dim obs.
Reward is shaped: flock-centroid progress toward goal + success bonus.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .generalized_primitive_env import GeneralizedPrimitiveEnv, N_ACTIONS


class DiscreteHerdingEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n_sheep: int = 10, seed: int = 0):
        super().__init__()
        self._env = GeneralizedPrimitiveEnv(n_sheep=n_sheep, seed=seed)
        self.L    = self._env.L
        self.goal = self._env.goal

        self.observation_space = spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._prev_dist = 0.0

    def reset(self, seed=None, options=None):
        obs = self._env.reset()
        self._prev_dist = self._centroid_dist()
        return obs.astype(np.float32), {}

    def step(self, action: int):
        obs, _, done, info = self._env.step(int(action))
        frac = info.get("fraction_in_goal", 0.0)
        cur_dist = self._centroid_dist()

        r  = (self._prev_dist - cur_dist) / self.L * 3.0
        r += float(frac >= 1.0) * 10.0
        self._prev_dist = cur_dist

        terminated = bool(frac >= 1.0)
        truncated  = done and not terminated
        return obs.astype(np.float32), float(r), terminated, truncated, info

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    def _centroid_dist(self) -> float:
        centroid = self._env._env.sheep_pos.mean(axis=0)
        return float(np.linalg.norm(centroid - self.goal))
