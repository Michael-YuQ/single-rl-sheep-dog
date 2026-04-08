"""
Primitive (macro-action) wrapper around SheepEnv.

The upper-level policy selects one of (N+12) discrete macro-actions:
  0        : MUSTER        — gather scattered sheep toward flock centroid (direct)
  1        : HOLD          — guard goal entrance, block escapes
  2..N+1   : DRIVE_i       — push sheep i toward goal (direct)
  N+2      : MUSTER_ARC    — arc around flock to approach outlier from behind
  N+3      : FLANK_LEFT    — arc to left flank of flock, push toward goal
  N+4      : FLANK_RIGHT   — arc to right flank of flock
  N+5      : SWEEP_BEHIND  — arc to position behind entire flock for full push
  N+6..N+11: BYPASS_DRIVE_i — arc around flock then drive sheep i (i=0..5)

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
BEHIND_OFFSET  = 3.0    # metres behind sheep (away from goal) to position dog
BYPASS_FACTOR  = 1.5    # flock spread multiplier for collision check
ARC_OFFSET     = 5.0    # extra metres away from flock when arcing


class PrimitiveEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n_sheep: int = 10, **sheep_kwargs):
        super().__init__()
        self.n = n_sheep
        self._env = SheepEnv(n_sheep=n_sheep, **sheep_kwargs)
        self.L = self._env.L
        self.goal = self._env.goal

        n_actions = n_sheep + 2 + 10   # MUSTER, HOLD, DRIVE*n, + 10 new arc actions
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
        centroid = sp.mean(axis=0)
        spread = float(np.std(sp, axis=0).mean()) + 1.0

        # ---- Original actions ----
        if action_id == 0:
            # MUSTER: direct to behind outlier sheep
            dists = np.linalg.norm(sp - centroid, axis=1)
            outlier = sp[np.argmax(dists)]
            away = outlier - centroid
            away_norm = np.linalg.norm(away) + 1e-8
            target = outlier + away / away_norm * BEHIND_OFFSET
            return self._move_toward(dp, target)

        elif action_id == 1:
            # HOLD: position at goal entrance
            goal_entrance = self.goal - np.array([self._env.goal_radius * 1.2, 0.0])
            return self._move_toward(dp, goal_entrance)

        elif 2 <= action_id <= self.n + 1:
            # DRIVE_i: push sheep i toward goal (direct)
            i = action_id - 2
            i = np.clip(i, 0, self.n - 1)
            sheep = sp[i]
            to_goal = self.goal - sheep
            to_goal_norm = np.linalg.norm(to_goal) + 1e-8
            behind = sheep - to_goal / to_goal_norm * BEHIND_OFFSET
            return self._move_toward(dp, behind)

        # ---- New arc/bypass actions ----
        elif action_id == self.n + 2:
            # MUSTER_ARC: arc around flock to outlier
            dists = np.linalg.norm(sp - centroid, axis=1)
            outlier = sp[np.argmax(dists)]
            away = outlier - centroid
            away_norm = np.linalg.norm(away) + 1e-8
            target = outlier + away / away_norm * BEHIND_OFFSET
            return self._arc_move_toward(dp, target, centroid, spread)

        elif action_id == self.n + 3:
            # FLANK_LEFT: arc to left of flock
            to_goal = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            left = np.array([-to_goal[1], to_goal[0]]) / to_goal_n
            target = centroid - to_goal / to_goal_n * (spread * BYPASS_FACTOR + BEHIND_OFFSET) + left * (spread + ARC_OFFSET)
            return self._arc_move_toward(dp, target, centroid, spread)

        elif action_id == self.n + 4:
            # FLANK_RIGHT: arc to right of flock
            to_goal = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            right = np.array([to_goal[1], -to_goal[0]]) / to_goal_n
            target = centroid - to_goal / to_goal_n * (spread * BYPASS_FACTOR + BEHIND_OFFSET) + right * (spread + ARC_OFFSET)
            return self._arc_move_toward(dp, target, centroid, spread)

        elif action_id == self.n + 5:
            # SWEEP_BEHIND: arc to behind entire flock (away from goal)
            to_goal = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            behind_flock = centroid - to_goal / to_goal_n * (spread * BYPASS_FACTOR + BEHIND_OFFSET)
            return self._arc_move_toward(dp, behind_flock, centroid, spread)

        elif self.n + 6 <= action_id <= self.n + 11:
            # BYPASS_DRIVE_i: arc around flock then drive sheep i
            k = action_id - (self.n + 6)
            k = np.clip(k, 0, self.n - 1)
            sheep = sp[k]
            to_goal = self.goal - sheep
            to_goal_norm = np.linalg.norm(to_goal) + 1e-8
            behind = sheep - to_goal / to_goal_norm * BEHIND_OFFSET
            return self._arc_move_toward(dp, behind, centroid, spread)

        else:
            return np.zeros(2, dtype=np.float32)

    def _move_toward(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - current
        dist = np.linalg.norm(delta) + 1e-8
        vel = delta / dist
        return vel.astype(np.float32)

    def _arc_move_toward(self, current: np.ndarray, target: np.ndarray,
                         centroid: np.ndarray, spread: float) -> np.ndarray:
        """Move toward target, arcing around flock if direct path cuts through."""
        to_target = target - current
        dist_tt   = np.linalg.norm(to_target) + 1e-8
        dir_tt    = to_target / dist_tt

        to_centroid = centroid - current
        proj_len    = float(np.clip(np.dot(to_centroid, dir_tt), 0.0, dist_tt))
        closest     = current + dir_tt * proj_len
        dist_to_path = np.linalg.norm(centroid - closest)

        if dist_to_path < spread * BYPASS_FACTOR:
            perp = np.array([-dir_tt[1], dir_tt[0]], dtype=np.float32)
            waypoint = centroid + perp * (spread * BYPASS_FACTOR + ARC_OFFSET)
            if np.linalg.norm(waypoint - current) > np.linalg.norm(target - current):
                waypoint = centroid - perp * (spread * BYPASS_FACTOR + ARC_OFFSET)
            return self._move_toward(current, waypoint)

        return self._move_toward(current, target)
