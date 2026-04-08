"""
embodied.Env adapter for the hierarchical shepherding environment.

Wraps PrimitiveEnv (discrete macro-actions) in the embodied interface
expected by DreamerV3, mirroring embodied/envs/from_gym.py.
"""

import sys
import pathlib
import functools

import elements
import numpy as np

# Make embodied importable from the dreamerv3 repo
_dreamer_root = pathlib.Path(__file__).parent.parent.parent / "minecraft" / "dreamerv3"
if str(_dreamer_root) not in sys.path:
    sys.path.insert(0, str(_dreamer_root))

import embodied

from .primitive_env import PrimitiveEnv
from .generalized_primitive_env import GeneralizedPrimitiveEnv
from .perimeter_env import PerimeterEnv
from .push_env import PushEnv
from .three_dog_env import ThreeDogEnv, N_JOINT, OBS_DIM


class ShepherdEnv(embodied.Env):

    def __init__(self, task: str = "default", n_sheep: int = 10, **kwargs):
        self._env = PrimitiveEnv(n_sheep=n_sheep, **kwargs)
        self._n_sheep = n_sheep
        self._done = True

    @functools.cached_property
    def obs_space(self):
        return {
            "vector": elements.Space(np.float32, (10,)),
            "reward": elements.Space(np.float32, ()),
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),
        }

    @functools.cached_property
    def act_space(self):
        n_actions = self._n_sheep + 2 + 10   # original + 10 arc/bypass actions
        return {
            "action": elements.Space(np.int32, (), 0, n_actions),
            "reset": elements.Space(bool, ()),
        }

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            obs = self._env.reset()
            return self._pack_obs(obs, 0.0, is_first=True)
        obs, reward, self._done, info = self._env.step(int(action["action"]))
        return self._pack_obs(
            obs, reward,
            is_last=bool(self._done),
            is_terminal=bool(self._done),
        )

    def _pack_obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        return {
            "vector": np.asarray(obs, dtype=np.float32),
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def render(self):
        return self._env.render("rgb_array")

    def close(self):
        pass


class PerimeterShepherdEnv(embodied.Env):
    """DreamerV3 adapter for the perimeter-only (all-arc) environment."""

    def __init__(self, task: str = "default", n_sheep: int = 10, **kwargs):
        self._env = PerimeterEnv(n_sheep=n_sheep, **kwargs)
        self._n_sheep = n_sheep
        self._done = True

    @functools.cached_property
    def obs_space(self):
        return {
            "vector": elements.Space(np.float32, (10,)),
            "reward": elements.Space(np.float32, ()),
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),
        }

    @functools.cached_property
    def act_space(self):
        from .generalized_primitive_env import N_ACTIONS
        return {
            "action": elements.Space(np.int32, (), 0, N_ACTIONS),
            "reset": elements.Space(bool, ()),
        }

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            obs = self._env.reset()
            return self._pack_obs(obs, 0.0, is_first=True)
        obs, reward, self._done, info = self._env.step(int(action["action"]))
        return self._pack_obs(
            obs, reward,
            is_last=bool(self._done),
            is_terminal=bool(self._done),
        )

    def _pack_obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        return {
            "vector": np.asarray(obs, dtype=np.float32),
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def render(self):
        return self._env.render("rgb_array")

    def close(self):
        pass


class PushShepherdEnv(embodied.Env):
    """DreamerV3 adapter for the tight-arc push-track environment (N=100)."""

    def __init__(self, task: str = "default", n_sheep: int = 100, **kwargs):
        self._env = PushEnv(n_sheep=n_sheep, **kwargs)
        self._done = True

    @functools.cached_property
    def obs_space(self):
        return {
            "vector": elements.Space(np.float32, (10,)),
            "reward": elements.Space(np.float32, ()),
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),
        }

    @functools.cached_property
    def act_space(self):
        from .push_env import N_PUSH_ACTIONS
        return {
            "action": elements.Space(np.int32, (), 0, N_PUSH_ACTIONS),
            "reset": elements.Space(bool, ()),
        }

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            obs = self._env.reset()
            return self._pack_obs(obs, 0.0, is_first=True)
        obs, reward, self._done, info = self._env.step(int(action["action"]))
        return self._pack_obs(
            obs, reward,
            is_last=bool(self._done),
            is_terminal=bool(self._done),
        )

    def _pack_obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        return {
            "vector": np.asarray(obs, dtype=np.float32),
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def render(self):
        return self._env.render("rgb_array")

    def close(self):
        pass


class ThreeDogShepherdEnv(embodied.Env):
    """DreamerV3 adapter: 3 dogs, 5 per-dog low-level primitives, 5^3=125 joint actions, N=100."""

    def __init__(self, task: str = "default", n_sheep: int = 100, **kwargs):
        self._env = ThreeDogEnv(n_sheep=n_sheep, **kwargs)
        self._done = True

    @functools.cached_property
    def obs_space(self):
        return {
            "vector": elements.Space(np.float32, (OBS_DIM,)),
            "reward": elements.Space(np.float32, ()),
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),
        }

    @functools.cached_property
    def act_space(self):
        return {
            "action": elements.Space(np.int32, (), 0, N_JOINT),
            "reset": elements.Space(bool, ()),
        }

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            obs = self._env.reset()
            return self._pack_obs(obs, 0.0, is_first=True)
        obs, reward, self._done, info = self._env.step(int(action["action"]))
        return self._pack_obs(
            obs, reward,
            is_last=bool(self._done),
            is_terminal=bool(self._done),
        )

    def _pack_obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        return {
            "vector": np.asarray(obs, dtype=np.float32),
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def render(self):
        return self._env.render("rgb_array")

    def close(self):
        pass
