"""
Strömbom (2014) sheep-herding environment.

Sheep dynamics:
  - Repulsion from dog within radius r_a (escape)
  - Attraction toward flock centre of mass (cohesion)
  - Random noise
  - Speed capped at sheep_speed

Dog is controlled by a 2-D velocity command (continuous), or can be
driven step-by-step by a geometric primitive controller.
"""

import gym
import numpy as np
from gym import spaces


class SheepEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        n_sheep: int = 10,
        n_dogs: int = 1,
        arena_size: float = 100.0,
        dog_speed: float = 2.0,
        sheep_speed: float = 1.5,
        r_a: float = 12.0,   # dog repulsion radius for sheep
        r_s: float = 2.0,    # sheep-sheep separation radius
        c: float = 1.05,     # relative weight: drive dog away vs flock
        p: float = 0.95,     # inertia (momentum) for sheep direction
        goal_radius: float = 10.0,
        max_steps: int = 2000,
        seed: int = 0,
        dense_reward_scale: float = 0.05,
    ):
        super().__init__()
        self.n = n_sheep
        self.n_dogs = n_dogs
        self.L = arena_size
        self.dog_speed = dog_speed
        self.sheep_speed = sheep_speed
        self.r_a = r_a
        self.r_s = r_s
        self.c = c
        self.p = p
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.dense_reward_scale = dense_reward_scale
        self._rng = np.random.default_rng(seed)

        # Goal position: fixed at (L*0.9, L*0.5)
        self.goal = np.array([self.L * 0.9, self.L * 0.5], dtype=np.float32)

        # Observation: dogs(x,y)*n_dogs + N*sheep(x,y) + N*sheep(vx,vy)
        obs_dim = 2 * n_dogs + self.n * 4
        self.observation_space = spaces.Box(
            low=0.0, high=self.L, shape=(obs_dim,), dtype=np.float32
        )
        # Action: dog velocity 2D per dog, normalised [-1,1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_dogs, 2), dtype=np.float32
        )

        self.dog_pos = None   # (n_dogs, 2)
        self.sheep_pos = None
        self.sheep_vel = None
        self.step_count = 0
        self._prev_frac = 0.0

    @property
    def dog_pos_single(self) -> np.ndarray:
        """Backward-compat: return first dog position as (2,) for single-dog wrappers."""
        return self.dog_pos[0]

    # ------------------------------------------------------------------
    def reset(self):
        self._rng = np.random.default_rng(self._rng.integers(0, 2**32))
        # Dogs start near left side, spread vertically
        ys = np.linspace(20, 80, self.n_dogs)
        self.dog_pos = np.column_stack([
            np.full(self.n_dogs, 10.0),
            ys,
        ]).astype(np.float32)                              # (n_dogs, 2)
        self.sheep_pos = self._rng.uniform(10, 50, size=(self.n, 2)).astype(np.float32)
        self.sheep_vel = np.zeros((self.n, 2), dtype=np.float32)
        self.step_count = 0
        self._prev_frac = self._fraction_in_goal()
        return self._get_obs()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.n_dogs, 2)
        action = np.clip(action, -1.0, 1.0)
        self.dog_pos = np.clip(
            self.dog_pos + action * self.dog_speed,
            0.0, self.L,
        )
        self._update_sheep()
        self.step_count += 1

        frac = self._fraction_in_goal()
        reward = float(self.dense_reward_scale * (frac - self._prev_frac))
        self._prev_frac = frac
        done = (frac >= 1.0) or (self.step_count >= self.max_steps)
        if frac >= 1.0:
            reward += 1.0
        info = {"fraction_in_goal": frac, "step": self.step_count}
        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    def _update_sheep(self):
        sp = self.sheep_pos   # (N, 2)
        N  = self.n

        # --- Escape from nearest dog: take max repulsion across all dogs ---
        # delta: (n_dogs, N, 2),  dist: (n_dogs, N, 1)
        delta_dogs = sp[None, :, :] - self.dog_pos[:, None, :]   # (D, N, 2)
        dist_dogs  = np.linalg.norm(delta_dogs, axis=2, keepdims=True) + 1e-8  # (D,N,1)
        esc_each   = np.where(dist_dogs < self.r_a,
                              delta_dogs / dist_dogs,
                              np.zeros_like(delta_dogs))          # (D, N, 2)
        escape = esc_each.sum(axis=0)                             # (N, 2) sum over dogs

        # --- Pairwise separation (N, N, 2) → (N, 2) ---
        diff = sp[:, None, :] - sp[None, :, :]    # (N, N, 2)
        pnorm = np.linalg.norm(diff, axis=2, keepdims=True) + 1e-8  # (N,N,1)
        diff_unit = diff / pnorm                   # (N, N, 2)
        mask_sep  = (pnorm[..., 0] < self.r_s)    # (N, N) bool
        np.fill_diagonal(mask_sep, False)
        sep = (diff_unit * mask_sep[..., None]).sum(axis=1)  # (N, 2)

        # --- Attraction to flock centroid (excluding self) ---
        total    = sp.sum(axis=0, keepdims=True)                    # (1, 2)
        centroid = (total - sp) / (N - 1)                           # (N, 2)
        attract  = centroid - sp                                     # (N, 2)
        attr_n   = np.linalg.norm(attract, axis=1, keepdims=True) + 1e-8
        attract  = attract / attr_n                                  # (N, 2)

        # --- Combine ---
        desired  = escape * self.c + attract + sep                   # (N, 2)
        des_norm = np.linalg.norm(desired, axis=1, keepdims=True) + 1e-8
        desired  = desired / des_norm

        noise = self._rng.standard_normal((N, 2)).astype(np.float32) * 0.05
        new_vel = self.p * self.sheep_vel + (1 - self.p) * desired + noise
        spd = np.linalg.norm(new_vel, axis=1, keepdims=True) + 1e-8
        new_vel = new_vel / spd * np.minimum(spd, self.sheep_speed)

        self.sheep_vel = new_vel
        self.sheep_pos = np.clip(self.sheep_pos + self.sheep_vel, 0.0, self.L)



    # ------------------------------------------------------------------
    def _fraction_in_goal(self) -> float:
        dists = np.linalg.norm(self.sheep_pos - self.goal, axis=1)
        return float((dists < self.goal_radius).sum()) / self.n

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.dog_pos.ravel(),          # (n_dogs*2,)
            self.sheep_pos.ravel(),
            self.sheep_vel.ravel(),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    def render(self, mode="rgb_array"):
        size = 400
        scale = size / self.L
        img = np.ones((size, size, 3), dtype=np.uint8) * 240
        # goal circle
        cy, cx = int(self.goal[1] * scale), int(self.goal[0] * scale)
        gr = int(self.goal_radius * scale)
        ys, xs = np.ogrid[:size, :size]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= gr ** 2
        img[mask] = [144, 238, 144]
        # sheep
        for s in self.sheep_pos:
            sx, sy = int(s[0] * scale), int(s[1] * scale)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = np.clip(sx + dx, 0, size - 1), np.clip(sy + dy, 0, size - 1)
                    img[ny, nx] = [200, 200, 255]
        # dogs — red shades per dog
        dog_colors = [[255, 80, 80], [255, 160, 0], [80, 200, 80], [80, 80, 255]]
        for d_idx, dp in enumerate(self.dog_pos):
            color = dog_colors[d_idx % len(dog_colors)]
            dx_p, dy_p = int(dp[0] * scale), int(dp[1] * scale)
            for dy in range(-4, 5):
                for ddx in range(-4, 5):
                    nx, ny = np.clip(dx_p + ddx, 0, size - 1), np.clip(dy_p + dy, 0, size - 1)
                    img[ny, nx] = color
        return img
