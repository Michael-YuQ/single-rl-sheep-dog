"""
Primitive (macro-action) wrapper around SheepEnv.

The upper-level policy selects one of (N+2) discrete macro-actions:
  0        : MUSTER  — gather scattered sheep toward flock centroid
  1        : HOLD    — guard goal entrance, block escapes
  2..N+1   : DRIVE_i — push sheep i toward goal

Each macro-action runs K=15 low-level steps with a geometric controller,
then returns the aggregated reward and the compressed upper-level observation.

Upper-level observation (10-dim):
  [dog_x, dog_y, flock_cx, flock_cy, flock_spread,
   farthest_x, farthest_y, fraction_in_goal, goal_x, goal_y]
  All normalised to [0, 1] by L.
"""

import gym
import numpy as np
from gym import spaces

from .sheep_env import SheepEnv


K = 15          # steps per macro-action
DOG_SPEED = 2.0
BEHIND_OFFSET = 3.0   # metres behind sheep (away from goal) to position dog


class PrimitiveEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n_sheep: int = 10, **sheep_kwargs):
        super().__init__()
        self.n = n_sheep
        self._env = SheepEnv(n_sheep=n_sheep, **sheep_kwargs)
        self.L = self._env.L
        self.goal = self._env.goal

        n_actions = n_sheep + 2   # MUSTER, HOLD, DRIVE_0..DRIVE_{n-1}
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def reset(self):
        self._env.reset()
        return self._upper_obs()

    def step(self, action: int):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(K):
            low_action = self._primitive_action(action)
            _, r, done, info = self._env.step(low_action)
            total_reward += r
            if done:
                break
        return self._upper_obs(), total_reward, done, info

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    # ------------------------------------------------------------------
    def _upper_obs(self) -> np.ndarray:
        sp = self._env.sheep_pos
        dp = self._env.dog_pos_single
        centroid = sp.mean(axis=0)
        spread = np.std(sp, axis=0).mean()
        dists_to_goal = np.linalg.norm(sp - self.goal, axis=1)
        farthest = sp[np.argmax(dists_to_goal)]
        frac = self._env._fraction_in_goal()
        obs = np.array([
            dp[0] / self.L, dp[1] / self.L,
            centroid[0] / self.L, centroid[1] / self.L,
            spread / self.L,
            farthest[0] / self.L, farthest[1] / self.L,
            frac,
            self.goal[0] / self.L, self.goal[1] / self.L,
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    def _primitive_action(self, action_id: int) -> np.ndarray:
        """Return a 2-D velocity command for one low-level step."""
        dp = self._env.dog_pos_single
        sp = self._env.sheep_pos

        if action_id == 0:
            # MUSTER: move toward centroid of sheep farthest from flock
            centroid = sp.mean(axis=0)
            dists = np.linalg.norm(sp - centroid, axis=1)
            outlier = sp[np.argmax(dists)]
            # Position behind outlier (away from centroid)
            away = outlier - centroid
            away_norm = np.linalg.norm(away) + 1e-8
            target = outlier + away / away_norm * BEHIND_OFFSET
            return self._move_toward(dp, target)

        elif action_id == 1:
            # HOLD: position at goal entrance
            goal_entrance = self.goal - np.array([self._env.goal_radius * 1.2, 0.0])
            return self._move_toward(dp, goal_entrance)

        else:
            # DRIVE_i: push sheep (action_id - 2) toward goal
            i = action_id - 2
            i = np.clip(i, 0, self.n - 1)
            sheep = sp[i]
            # Position dog behind sheep relative to goal
            to_goal = self.goal - sheep
            to_goal_norm = np.linalg.norm(to_goal) + 1e-8
            behind = sheep - to_goal / to_goal_norm * BEHIND_OFFSET
            return self._move_toward(dp, behind)

    def _move_toward(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - current
        dist = np.linalg.norm(delta) + 1e-8
        vel = delta / dist  # unit vector, env will scale by dog_speed
        return vel.astype(np.float32)
