"""
Rank-based primitive environment.

Action space is FIXED at 12, independent of N:
  0        : MUSTER  — move behind the sheep farthest from flock centroid
  1        : HOLD    — guard goal entrance
  2..11    : DRIVE_rank{k} — drive the k-th farthest sheep from goal (k=0..9)
             if N < k+1, clamp to the farthest available sheep

Upper-level observation (10-dim, independent of N):
  [dog_x/L, dog_y/L, flock_cx/L, flock_cy/L, flock_spread/L,
   farthest_x/L, farthest_y/L, fraction_in_goal, goal_x/L, goal_y/L]

This env is used for zero-shot generalisation experiments: a policy
trained with N=10 can be applied directly to N=20/50/100 because the
observation and action spaces have the same shape.

max_steps scales linearly with N so every scale gets a fair time budget.
"""

import gym
import numpy as np
from gym import spaces

from .sheep_env import SheepEnv

K_PRIMITIVES = 15          # low-level steps per macro-action
DOG_SPEED    = 2.0
BEHIND_DIST  = 3.0         # metres behind sheep (away from goal)
N_ACTIONS    = 12          # fixed: MUSTER + HOLD + DRIVE_rank{0..9}
BASE_STEPS   = 4000        # max_steps for N=10 (×N/10 for larger herds)


class GeneralizedPrimitiveEnv(gym.Env):
    """Rank-based macro-action env supporting arbitrary n_sheep."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n_sheep: int = 10, seed: int = 0, **sheep_kwargs):
        super().__init__()
        self.n = n_sheep
        # Scale time budget with herd size
        max_steps = int(BASE_STEPS * max(1, n_sheep / 10))
        self._env = SheepEnv(n_sheep=n_sheep, max_steps=max_steps,
                             seed=seed, **sheep_kwargs)
        self.L    = self._env.L
        self.goal = self._env.goal

        self.action_space      = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self):
        self._env.reset()
        return self._upper_obs()

    def step(self, action: int):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(K_PRIMITIVES):
            low_action = self._primitive_action(int(action))
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

    # ------------------------------------------------------------------
    def _primitive_action(self, action_id: int) -> np.ndarray:
        dp = self._env.dog_pos_single
        sp = self._env.sheep_pos

        if action_id == 0:
            # MUSTER: go behind the sheep farthest from flock centroid
            centroid = sp.mean(axis=0)
            dists    = np.linalg.norm(sp - centroid, axis=1)
            outlier  = sp[np.argmax(dists)]
            away     = outlier - centroid
            away_n   = np.linalg.norm(away) + 1e-8
            target   = outlier + away / away_n * BEHIND_DIST
            return self._move_toward(dp, target)

        elif action_id == 1:
            # HOLD: guard goal entrance
            entrance = self.goal - np.array([self._env.goal_radius * 1.2, 0.0])
            return self._move_toward(dp, entrance)

        else:
            # DRIVE rank-k: k = action_id - 2  (0..9)
            k = action_id - 2
            dists_to_goal = np.linalg.norm(sp - self.goal, axis=1)
            # argsort descending; clamp k to valid range
            sorted_idx = np.argsort(dists_to_goal)[::-1]
            i = int(sorted_idx[min(k, len(sorted_idx) - 1)])
            sheep  = sp[i]
            to_g   = self.goal - sheep
            to_g_n = np.linalg.norm(to_g) + 1e-8
            behind = sheep - to_g / to_g_n * BEHIND_DIST
            return self._move_toward(dp, behind)

    def _move_toward(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - current
        dist  = np.linalg.norm(delta) + 1e-8
        return (delta / dist).astype(np.float32)
