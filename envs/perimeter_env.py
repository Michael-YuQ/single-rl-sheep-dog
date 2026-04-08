"""
Perimeter-only environment: ALL actions are forced to arc around the flock.
No geometric controller is ever allowed to cut through the sheep group.

This is a strict superset of the arc-action constraint — even MUSTER, HOLD
and DRIVE_rank{k} (action IDs 0-11) must approach their targets by going
around the flock perimeter instead of straight through.

Inherits GeneralizedPrimitiveEnv unchanged except for _primitive_action,
which routes every target computation through _arc_move_toward.

Action space: 22 (same as GeneralizedPrimitiveEnv) — the semantics of each
action ID are preserved (what to target), only the PATH to that target changes.
"""

import numpy as np
from .generalized_primitive_env import GeneralizedPrimitiveEnv, BEHIND_DIST, BYPASS_FACTOR, ARC_OFFSET


class PerimeterEnv(GeneralizedPrimitiveEnv):
    """All 22 actions arc around the flock — zero straight-through moves."""

    def _primitive_action(self, action_id: int) -> np.ndarray:
        dp = self._env.dog_pos_single
        sp = self._env.sheep_pos
        centroid = sp.mean(axis=0)
        spread   = float(np.std(sp, axis=0).mean()) + 1.0

        # ---- Compute target point (same logic as parent) ----
        if action_id == 0 or action_id == 12:
            # MUSTER / MUSTER_ARC: behind outlier sheep
            dists   = np.linalg.norm(sp - centroid, axis=1)
            outlier = sp[np.argmax(dists)]
            away    = outlier - centroid
            away_n  = np.linalg.norm(away) + 1e-8
            target  = outlier + away / away_n * BEHIND_DIST

        elif action_id == 1:
            # HOLD: goal entrance
            target = self.goal - np.array([self._env.goal_radius * 1.2, 0.0])

        elif action_id == 13:
            # FLANK_LEFT
            to_goal  = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            left  = np.array([-to_goal[1], to_goal[0]]) / to_goal_n
            target = centroid - to_goal / to_goal_n * (spread * BYPASS_FACTOR + BEHIND_DIST) + left * (spread + ARC_OFFSET)

        elif action_id == 14:
            # FLANK_RIGHT
            to_goal  = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            right = np.array([to_goal[1], -to_goal[0]]) / to_goal_n
            target = centroid - to_goal / to_goal_n * (spread * BYPASS_FACTOR + BEHIND_DIST) + right * (spread + ARC_OFFSET)

        elif action_id == 15:
            # SWEEP_BEHIND
            to_goal  = self.goal - centroid
            to_goal_n = np.linalg.norm(to_goal) + 1e-8
            target = centroid - to_goal / to_goal_n * (spread * BYPASS_FACTOR + BEHIND_DIST)

        elif 2 <= action_id <= 11:
            # DRIVE_rank{k}
            k = action_id - 2
            dists_to_goal = np.linalg.norm(sp - self.goal, axis=1)
            sorted_idx = np.argsort(dists_to_goal)[::-1]
            i = int(sorted_idx[min(k, len(sorted_idx) - 1)])
            sheep  = sp[i]
            to_g   = self.goal - sheep
            to_g_n = np.linalg.norm(to_g) + 1e-8
            target = sheep - to_g / to_g_n * BEHIND_DIST

        elif 16 <= action_id <= 21:
            # BYPASS_DRIVE_rank{k}
            k = action_id - 16
            dists_to_goal = np.linalg.norm(sp - self.goal, axis=1)
            sorted_idx = np.argsort(dists_to_goal)[::-1]
            i = int(sorted_idx[min(k, len(sorted_idx) - 1)])
            sheep  = sp[i]
            to_g   = self.goal - sheep
            to_g_n = np.linalg.norm(to_g) + 1e-8
            target = sheep - to_g / to_g_n * BEHIND_DIST

        else:
            return np.zeros(2, dtype=np.float32)

        # ---- ALWAYS arc — no straight-through allowed ----
        return self._arc_move_toward(dp, target, sp, centroid, spread)
