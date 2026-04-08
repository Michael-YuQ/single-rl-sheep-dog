"""
Rank-based primitive environment with extended action space.

Action space is FIXED at 22, independent of N:
  0        : MUSTER        — move behind the sheep farthest from flock centroid (direct)
  1        : HOLD          — guard goal entrance
  2–11     : DRIVE_rank{k} — drive the k-th farthest sheep from goal (direct, k=0..9)
  12       : MUSTER_ARC    — arc around flock to approach outlier sheep from behind
  13       : FLANK_LEFT    — arc left around flock centroid to push flock rightward toward goal
  14       : FLANK_RIGHT   — arc right around flock centroid to push flock leftward toward goal
  15       : SWEEP_BEHIND  — arc to position directly behind flock (away from goal) for full push
  16–21    : BYPASS_DRIVE_rank{k} — arc around flock then drive k-th farthest sheep (k=0..5)

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

K_PRIMITIVES   = 15          # low-level steps per macro-action
DOG_SPEED      = 2.0
BEHIND_DIST    = 3.0         # metres behind sheep (away from goal)
N_ACTIONS      = 22          # MUSTER + HOLD + DRIVE*10 + MUSTER_ARC + FLANK_L + FLANK_R + SWEEP + BYPASS*6
BASE_STEPS     = 4000        # max_steps for N=10 (×N/10 for larger herds)
BYPASS_FACTOR  = 1.5         # collision check: flock spread multiplier
ARC_OFFSET     = 5.0         # extra metres away from flock when arcing


class GeneralizedPrimitiveEnv(gym.Env):
    """Rank-based macro-action env with arc/bypass actions supporting arbitrary n_sheep."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n_sheep: int = 10, seed: int = 0, **sheep_kwargs):
        super().__init__()
        self.n = n_sheep
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

        centroid = sp.mean(axis=0)
        spread   = float(np.std(sp, axis=0).mean()) + 1.0  # minimum spread guard

        # ---- Original 12 actions ----
        if action_id == 0:
            # MUSTER: direct line to behind outlier sheep
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

        elif 2 <= action_id <= 11:
            # DRIVE rank-k: direct approach behind k-th farthest sheep
            k = action_id - 2
            dists_to_goal = np.linalg.norm(sp - self.goal, axis=1)
            sorted_idx = np.argsort(dists_to_goal)[::-1]
            i = int(sorted_idx[min(k, len(sorted_idx) - 1)])
            sheep  = sp[i]
            to_g   = self.goal - sheep
            to_g_n = np.linalg.norm(to_g) + 1e-8
            behind = sheep - to_g / to_g_n * BEHIND_DIST
            return self._move_toward(dp, behind)

        # ---- New arc/bypass actions ----
        elif action_id == 12:
            # MUSTER_ARC: arc around flock to reach behind outlier sheep
            dists    = np.linalg.norm(sp - centroid, axis=1)
            outlier  = sp[np.argmax(dists)]
            away     = outlier - centroid
            away_n   = np.linalg.norm(away) + 1e-8
            target   = outlier + away / away_n * BEHIND_DIST
            return self._arc_move_toward(dp, target, sp, centroid, spread)

        elif action_id == 13:
            # FLANK_LEFT: arc to left flank of flock then push toward goal
            to_goal  = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            left     = np.array([-to_goal[1], to_goal[0]]) / to_goal_n
            target   = centroid - to_goal / to_goal_n * (spread * BYPASS_FACTOR + BEHIND_DIST) + left * (spread + ARC_OFFSET)
            return self._arc_move_toward(dp, target, sp, centroid, spread)

        elif action_id == 14:
            # FLANK_RIGHT: arc to right flank of flock then push toward goal
            to_goal  = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            right    = np.array([to_goal[1], -to_goal[0]]) / to_goal_n
            target   = centroid - to_goal / to_goal_n * (spread * BYPASS_FACTOR + BEHIND_DIST) + right * (spread + ARC_OFFSET)
            return self._arc_move_toward(dp, target, sp, centroid, spread)

        elif action_id == 15:
            # SWEEP_BEHIND: arc to position directly behind entire flock (away from goal)
            to_goal  = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            behind_flock = centroid - to_goal / to_goal_n * (spread * BYPASS_FACTOR + BEHIND_DIST)
            return self._arc_move_toward(dp, behind_flock, sp, centroid, spread)

        elif 16 <= action_id <= 21:
            # BYPASS_DRIVE rank-k: arc around flock then approach behind k-th farthest sheep
            k = action_id - 16
            dists_to_goal = np.linalg.norm(sp - self.goal, axis=1)
            sorted_idx = np.argsort(dists_to_goal)[::-1]
            i = int(sorted_idx[min(k, len(sorted_idx) - 1)])
            sheep  = sp[i]
            to_g   = self.goal - sheep
            to_g_n = np.linalg.norm(to_g) + 1e-8
            behind = sheep - to_g / to_g_n * BEHIND_DIST
            return self._arc_move_toward(dp, behind, sp, centroid, spread)

        else:
            return np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    def _move_toward(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - current
        dist  = np.linalg.norm(delta) + 1e-8
        return (delta / dist).astype(np.float32)

    def _arc_move_toward(self, current: np.ndarray, target: np.ndarray,
                         sheep_pos: np.ndarray, centroid: np.ndarray,
                         spread: float) -> np.ndarray:
        """
        Move toward target, arcing around the flock if the direct path would
        cut through the flock.

        Collision check: project flock centroid onto the segment [current, target].
        If the closest point on that segment to centroid is within
        spread * BYPASS_FACTOR, compute a perpendicular waypoint to arc around.
        """
        to_target = target - current
        dist_tt   = np.linalg.norm(to_target) + 1e-8
        dir_tt    = to_target / dist_tt

        to_centroid = centroid - current
        proj_len    = np.dot(to_centroid, dir_tt)
        proj_len    = float(np.clip(proj_len, 0.0, dist_tt))
        closest     = current + dir_tt * proj_len
        dist_to_path = np.linalg.norm(centroid - closest)

        if dist_to_path < spread * BYPASS_FACTOR:
            perp = np.array([-dir_tt[1], dir_tt[0]], dtype=np.float32)
            waypoint = centroid + perp * (spread * BYPASS_FACTOR + ARC_OFFSET)
            if np.linalg.norm(waypoint - current) > np.linalg.norm(target - current):
                waypoint = centroid - perp * (spread * BYPASS_FACTOR + ARC_OFFSET)
            return self._move_toward(current, waypoint)

        return self._move_toward(current, target)
