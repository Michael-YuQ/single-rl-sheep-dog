"""
Continuous-action wrapper around SheepEnv for PPO/A2C training.

The agent controls the dog via a direct 2D velocity at every step
(no macro-action abstraction). The observation is the same 10-dim
compressed vector used by the DreamerV3 macro-action environment,
so results are fairly comparable.

Observation (10-dim, normalised to [0,1]):
  [dog_x, dog_y, centroid_x, centroid_y, spread,
   farthest_x, farthest_y, fraction_in_goal, goal_x, goal_y]

Action (2-dim, continuous):
  [vx, vy] in [-1, 1]  — scaled by dog_speed inside SheepEnv.step
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .sheep_env import SheepEnv

MAX_STEPS = 6000


class ContinuousSheepEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n_sheep: int = 10, seed: int = 0, **sheep_kwargs):
        super().__init__()
        self.n = n_sheep
        self._env = SheepEnv(
            n_sheep=n_sheep, seed=seed,
            max_steps=MAX_STEPS,
            dense_reward_scale=0.1,
            **sheep_kwargs,
        )
        self.L    = self._env.L
        self.goal = self._env.goal

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env._rng = np.random.default_rng(seed)
        self._env.reset()
        self._prev_dist = self._centroid_goal_dist()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(1, 2)
        _, _, done, info = self._env.step(action)

        frac = info.get("fraction_in_goal", 0.0)
        cur_dist = self._centroid_goal_dist()

        r  = (self._prev_dist - cur_dist) / self.L * 5.0   # progress reward
        r += float(frac >= 1.0) * 20.0                     # success bonus
        self._prev_dist = cur_dist

        terminated = bool(frac >= 1.0)
        truncated  = done and not terminated
        return self._obs(), float(r), terminated, truncated, info

    def _centroid_goal_dist(self) -> float:
        centroid = self._env.sheep_pos.mean(axis=0)
        return float(np.linalg.norm(centroid - self.goal))

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    def _obs(self) -> np.ndarray:
        sp = self._env.sheep_pos
        dp = self._env.dog_pos_single
        centroid = sp.mean(axis=0)
        spread   = float(np.std(sp, axis=0).mean())
        dists    = np.linalg.norm(sp - self.goal, axis=1)
        farthest = sp[np.argmax(dists)]
        frac     = self._env._fraction_in_goal()
        return np.array([
            dp[0] / self.L, dp[1] / self.L,
            centroid[0] / self.L, centroid[1] / self.L,
            spread / self.L,
            farthest[0] / self.L, farthest[1] / self.L,
            frac,
            self.goal[0] / self.L, self.goal[1] / self.L,
        ], dtype=np.float32)
