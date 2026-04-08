"""
Three-dog low-level joint action environment.

Design:
  - 3 dogs, each with 5 low-level primitive actions
  - Joint action = (a0, a1, a2), encoded as single int: a0*25 + a1*5 + a2
  - Total joint actions: 5^3 = 125
  - Each joint action runs K=5 low-level steps (shorter than 15-step macro-actions)
  - Milestone reward included for long-tail problem

Per-dog primitives (applied for K=5 steps, recomputed each step):
  0: PUSH_BEHIND    — stay behind flock centroid relative to goal
  1: FLANK_LEFT     — arc to left flank of flock
  2: FLANK_RIGHT    — arc to right flank of flock
  3: HOLD           — guard goal entrance (prevent escapes)
  4: COLLECT_STRAGGLER — chase most isolated outside-goal sheep

Observation (14-dim, normalised to [0,1]):
  [dog0_x, dog0_y, dog1_x, dog1_y, dog2_x, dog2_y,
   centroid_x, centroid_y, spread,
   farthest_x, farthest_y, fraction_in_goal,
   goal_x, goal_y]

Designed for direct N=100 training.
"""

import gym
import numpy as np
from gym import spaces

from .sheep_env import SheepEnv

K_STEPS    = 5      # low-level steps per joint action (shorter than macro K=15)
N_DOGS     = 3
N_PER_DOG  = 5      # per-dog primitives
N_JOINT    = N_PER_DOG ** N_DOGS   # 125
OBS_DIM    = 14
MAX_STEPS  = 40000  # for N=100
BEHIND_D   = 2.5
ARC_R      = 1.2    # bypass factor (tight)
ARC_OFF    = 1.0    # arc offset

MILESTONES = {0.50: 0.5, 0.75: 1.0, 0.90: 2.0, 0.95: 3.0, 1.00: 5.0}


class ThreeDogEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n_sheep: int = 100, seed: int = 0, **kwargs):
        super().__init__()
        self.n = n_sheep
        self._env = SheepEnv(
            n_sheep=n_sheep, n_dogs=N_DOGS,
            max_steps=MAX_STEPS, seed=seed, **kwargs)
        self.L    = self._env.L
        self.goal = self._env.goal
        self._ms_frac = 0.0

        self.action_space      = spaces.Discrete(N_JOINT)
        self.observation_space = spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)

    # ------------------------------------------------------------------
    def reset(self):
        self._env.reset()
        self._ms_frac = self._env._fraction_in_goal()
        return self._obs()

    def step(self, joint_action: int):
        a = self._decode(int(joint_action))   # (3,) per-dog action ids
        total_r = 0.0
        done = False
        info = {}
        for _ in range(K_STEPS):
            velocities = self._compute_velocities(a)  # (3, 2)
            _, r, done, info = self._env.step(velocities)
            total_r += r
            if done:
                break

        frac = info["fraction_in_goal"]
        ms_r = 0.0
        for t, bonus in MILESTONES.items():
            if self._ms_frac < t <= frac:
                ms_r += bonus
        self._ms_frac = frac
        return self._obs(), total_r + ms_r, done, info

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    # ------------------------------------------------------------------
    def _decode(self, joint: int):
        """Decode joint action int → [a0, a1, a2]."""
        a2 = joint % N_PER_DOG
        a1 = (joint // N_PER_DOG) % N_PER_DOG
        a0 = joint // (N_PER_DOG * N_PER_DOG)
        return [a0, a1, a2]

    # ------------------------------------------------------------------
    def _compute_velocities(self, per_dog_actions) -> np.ndarray:
        sp       = self._env.sheep_pos
        dogs     = self._env.dog_pos        # (3, 2)
        centroid = sp.mean(axis=0)
        spread   = float(np.std(sp, axis=0).mean()) + 1.0

        to_goal   = self.goal - centroid
        to_goal_n = np.linalg.norm(to_goal) + 1e-8
        behind_dir = -to_goal / to_goal_n
        left_dir   = np.array([-to_goal[1],  to_goal[0]]) / to_goal_n
        right_dir  = np.array([ to_goal[1], -to_goal[0]]) / to_goal_n

        # Outside-goal straggler
        outside_mask = np.linalg.norm(sp - self.goal, axis=1) > self._env.goal_radius
        outside = sp[outside_mask]
        if len(outside) > 0:
            iso = np.linalg.norm(outside - centroid, axis=1)
            straggler = outside[np.argmax(iso)]
        else:
            straggler = centroid   # fallback

        velocities = np.zeros((N_DOGS, 2), dtype=np.float32)
        for i, action in enumerate(per_dog_actions):
            dp = dogs[i]

            if action == 0:    # PUSH_BEHIND
                target = centroid + behind_dir * (spread * ARC_R + BEHIND_D)
                velocities[i] = self._arc_vel(dp, target, centroid, spread)

            elif action == 1:  # FLANK_LEFT
                target = centroid + behind_dir * (spread * ARC_R + BEHIND_D) \
                       + left_dir * (spread + ARC_OFF * 2)
                velocities[i] = self._arc_vel(dp, target, centroid, spread)

            elif action == 2:  # FLANK_RIGHT
                target = centroid + behind_dir * (spread * ARC_R + BEHIND_D) \
                       + right_dir * (spread + ARC_OFF * 2)
                velocities[i] = self._arc_vel(dp, target, centroid, spread)

            elif action == 3:  # HOLD
                entrance = self.goal - np.array([self._env.goal_radius * 1.2, 0.0])
                velocities[i] = self._move_toward(dp, entrance)

            elif action == 4:  # COLLECT_STRAGGLER
                away = straggler - centroid
                away_n = np.linalg.norm(away) + 1e-8
                target = straggler + away / away_n * BEHIND_D
                velocities[i] = self._arc_vel(dp, target, centroid, spread)

        return velocities

    # ------------------------------------------------------------------
    def _move_toward(self, current, target):
        d = target - current
        n = np.linalg.norm(d) + 1e-8
        return (d / n).astype(np.float32)

    def _arc_vel(self, current, target, centroid, spread):
        to_t = target - current
        dist = np.linalg.norm(to_t) + 1e-8
        dir_ = to_t / dist
        to_c = centroid - current
        proj = float(np.clip(np.dot(to_c, dir_), 0.0, dist))
        closest = current + dir_ * proj
        if np.linalg.norm(centroid - closest) < spread * ARC_R:
            perp = np.array([-dir_[1], dir_[0]], dtype=np.float32)
            wp = centroid + perp * (spread * ARC_R + ARC_OFF)
            if np.linalg.norm(wp - current) > np.linalg.norm(target - current):
                wp = centroid - perp * (spread * ARC_R + ARC_OFF)
            return self._move_toward(current, wp)
        return self._move_toward(current, target)

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        sp = self._env.sheep_pos
        dp = self._env.dog_pos            # (3, 2)
        centroid = sp.mean(axis=0)
        spread   = float(np.std(sp, axis=0).mean())
        dists    = np.linalg.norm(sp - self.goal, axis=1)
        farthest = sp[np.argmax(dists)]
        frac     = self._env._fraction_in_goal()
        return np.array([
            dp[0, 0] / self.L, dp[0, 1] / self.L,
            dp[1, 0] / self.L, dp[1, 1] / self.L,
            dp[2, 0] / self.L, dp[2, 1] / self.L,
            centroid[0] / self.L, centroid[1] / self.L,
            spread / self.L,
            farthest[0] / self.L, farthest[1] / self.L,
            frac,
            self.goal[0] / self.L, self.goal[1] / self.L,
        ], dtype=np.float32)
