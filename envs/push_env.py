"""
Push-track environment: tight perimeter arc + continuous-tracking push action
+ multi-milestone reward to eliminate the long-tail problem.

Key differences from PerimeterEnv:
  1. Tighter arc: PUSH_ARC_OFFSET=1.0, PUSH_BYPASS_FACTOR=1.2
  2. action 0 = PUSH_TRACK: continuously stay behind flock centroid
  3. action 22 = COLLECT_STRAGGLER: target the most isolated sheep outside goal
  4. Multi-milestone reward: superlinear bonus at 50/75/90/95/100% to incentivise
     collecting the last few sheep (the long-tail problem)

Action space: 23
  0        : PUSH_TRACK       — track behind flock centroid
  1        : HOLD             — guard goal entrance
  2-11     : DRIVE_rank{k}    — push k-th farthest sheep
  12       : MUSTER_ARC       — arc to outlier sheep
  13       : FLANK_LEFT
  14       : FLANK_RIGHT
  15       : SWEEP_BEHIND
  16-21    : BYPASS_DRIVE{k}
  22       : COLLECT_STRAGGLER — arc to most isolated outside-goal sheep
"""

import numpy as np
from gym import spaces
from .generalized_primitive_env import (
    GeneralizedPrimitiveEnv, BEHIND_DIST, SheepEnv
)

PUSH_ARC_OFFSET    = 1.0
PUSH_BYPASS_FACTOR = 1.2
BASE_STEPS_100     = 40000
N_PUSH_ACTIONS     = 23   # 22 original + COLLECT_STRAGGLER

MILESTONES = {
    0.50: 0.5,
    0.75: 1.0,
    0.90: 2.0,
    0.95: 3.0,
    1.00: 5.0,
}


class PushEnv(GeneralizedPrimitiveEnv):
    """Tight-arc + milestone-reward env, designed for direct N=100 training."""

    def __init__(self, n_sheep: int = 100, seed: int = 0, **sheep_kwargs):
        super().__init__(n_sheep=n_sheep, seed=seed, **sheep_kwargs)
        if n_sheep >= 50:
            max_steps = int(BASE_STEPS_100 * max(1, n_sheep / 100))
            self._env = SheepEnv(
                n_sheep=n_sheep, max_steps=max_steps, seed=seed, **sheep_kwargs)
        self._ms_frac = 0.0
        self.action_space = spaces.Discrete(N_PUSH_ACTIONS)

    def reset(self):
        obs = super().reset()
        self._ms_frac = self._env._fraction_in_goal()
        return obs

    def step(self, action: int):
        obs, base_r, done, info = super().step(action)
        frac = info["fraction_in_goal"]

        milestone_r = 0.0
        for threshold, bonus in MILESTONES.items():
            if self._ms_frac < threshold <= frac:
                milestone_r += bonus
        self._ms_frac = frac

        return obs, base_r + milestone_r, done, info

    # ------------------------------------------------------------------
    def _primitive_action(self, action_id: int) -> np.ndarray:
        dp = self._env.dog_pos_single
        sp = self._env.sheep_pos
        centroid = sp.mean(axis=0)
        spread   = float(np.std(sp, axis=0).mean()) + 1.0

        if action_id == 0:
            to_goal   = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            target    = centroid - to_goal / to_goal_n * BEHIND_DIST

        elif action_id == 1:
            target = self.goal - np.array([self._env.goal_radius * 1.2, 0.0])

        elif 2 <= action_id <= 11:
            k     = action_id - 2
            dists = np.linalg.norm(sp - self.goal, axis=1)
            idx   = np.argsort(dists)[::-1]
            i     = int(idx[min(k, len(idx) - 1)])
            sheep = sp[i]
            to_g  = self.goal - sheep
            target = sheep - to_g / (np.linalg.norm(to_g) + 1e-8) * BEHIND_DIST

        elif action_id == 12:
            dists   = np.linalg.norm(sp - centroid, axis=1)
            outlier = sp[np.argmax(dists)]
            away    = outlier - centroid
            target  = outlier + away / (np.linalg.norm(away) + 1e-8) * BEHIND_DIST

        elif action_id == 13:
            to_goal   = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            left  = np.array([-to_goal[1], to_goal[0]]) / to_goal_n
            target = (centroid
                      - to_goal / to_goal_n * (spread * PUSH_BYPASS_FACTOR + BEHIND_DIST)
                      + left * (spread + PUSH_ARC_OFFSET))

        elif action_id == 14:
            to_goal   = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            right = np.array([to_goal[1], -to_goal[0]]) / to_goal_n
            target = (centroid
                      - to_goal / to_goal_n * (spread * PUSH_BYPASS_FACTOR + BEHIND_DIST)
                      + right * (spread + PUSH_ARC_OFFSET))

        elif action_id == 15:
            to_goal   = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            target = centroid - to_goal / to_goal_n * (spread * PUSH_BYPASS_FACTOR + BEHIND_DIST)

        elif 16 <= action_id <= 21:
            k     = action_id - 16
            dists = np.linalg.norm(sp - self.goal, axis=1)
            idx   = np.argsort(dists)[::-1]
            i     = int(idx[min(k, len(idx) - 1)])
            sheep = sp[i]
            to_g  = self.goal - sheep
            target = sheep - to_g / (np.linalg.norm(to_g) + 1e-8) * BEHIND_DIST

        elif action_id == 22:
            # COLLECT_STRAGGLER: arc to the most isolated sheep still outside goal
            outside_mask = np.linalg.norm(sp - self.goal, axis=1) > self._env.goal_radius
            outside = sp[outside_mask]
            if len(outside) == 0:
                return np.zeros(2, dtype=np.float32)
            centroid_out = outside.mean(axis=0)
            isolation    = np.linalg.norm(outside - centroid_out, axis=1)
            straggler    = outside[np.argmax(isolation)]
            away = straggler - centroid
            target = straggler + away / (np.linalg.norm(away) + 1e-8) * BEHIND_DIST

        else:
            return np.zeros(2, dtype=np.float32)

        return self._tight_arc(dp, target, centroid, spread)

    def _tight_arc(self, current: np.ndarray, target: np.ndarray,
                   centroid: np.ndarray, spread: float) -> np.ndarray:
        to_target    = target - current
        dist_tt      = np.linalg.norm(to_target) + 1e-8
        dir_tt       = to_target / dist_tt
        to_centroid  = centroid - current
        proj_len     = float(np.clip(np.dot(to_centroid, dir_tt), 0.0, dist_tt))
        closest      = current + dir_tt * proj_len
        dist_to_path = np.linalg.norm(centroid - closest)

        if dist_to_path < spread * PUSH_BYPASS_FACTOR:
            perp     = np.array([-dir_tt[1], dir_tt[0]], dtype=np.float32)
            waypoint = centroid + perp * (spread * PUSH_BYPASS_FACTOR + PUSH_ARC_OFFSET)
            if np.linalg.norm(waypoint - current) > np.linalg.norm(target - current):
                waypoint = centroid - perp * (spread * PUSH_BYPASS_FACTOR + PUSH_ARC_OFFSET)
            return self._move_toward(current, waypoint)

        return self._move_toward(current, target)
