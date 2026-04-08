"""
Train PPO and A2C agents on the continuous-action sheep herding environment.

Both agents observe the same 10-dim compressed state as DreamerV3, but act
directly with a 2D continuous velocity (no macro-action abstraction).

Usage:
  python train_ppo_ac.py [--total_steps 1000000] [--n_envs 8]

Outputs:
  results/ppo_model.zip
  results/ac_model.zip
  results/ppo_train.log
  results/ac_train.log
"""
import sys, pathlib, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from envs.discrete_gym_env import DiscreteHerdingEnv

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

POLICY_KWARGS = dict(net_arch=[256, 256])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ProgressCallback(BaseCallback):
    def __init__(self, log_interval=10000, log_path=None):
        super().__init__()
        self.log_interval = log_interval
        self.log_path = log_path
        self._last_log = 0
        self._ep_rewards = []
        self._ep_lengths = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                self._ep_lengths.append(info["episode"]["l"])

        if self.num_timesteps - self._last_log >= self.log_interval and self._ep_rewards:
            mean_r = np.mean(self._ep_rewards[-50:])
            mean_l = np.mean(self._ep_lengths[-50:])
            msg = (f"step={self.num_timesteps:>8d}  "
                   f"ep_reward={mean_r:.4f}  ep_len={mean_l:.1f}")
            print(msg)
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(msg + "\n")
            self._last_log = self.num_timesteps
        return True


def train_agent(algo_cls, name, total_steps, n_envs):
    print(f"\n{'='*55}")
    print(f"Training {name}  |  steps={total_steps}  |  n_envs={n_envs}  |  device={DEVICE}")
    print(f"{'='*55}")

    def make_env():
        return Monitor(DiscreteHerdingEnv(n_sheep=10))

    vec_env = make_vec_env(make_env, n_envs=n_envs)

    if algo_cls is PPO:
        model = PPO(
            "MlpPolicy", vec_env,
            policy_kwargs=POLICY_KWARGS,
            n_steps=4096,
            batch_size=512,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            learning_rate=1e-4,
            device=DEVICE,
            verbose=0,
        )
    else:
        model = A2C(
            "MlpPolicy", vec_env,
            policy_kwargs=POLICY_KWARGS,
            n_steps=512,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.005,
            learning_rate=3e-4,
            device=DEVICE,
            verbose=0,
        )

    log_path = RESULTS / f"{name.lower()}_train.log"
    log_path.write_text("")
    cb = ProgressCallback(log_interval=10000, log_path=str(log_path))

    ckpt_cb = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path=str(RESULTS / f"{name.lower()}_ckpts"),
        name_prefix=name.lower(),
    )

    model.learn(total_timesteps=total_steps, callback=[cb, ckpt_cb])
    save_path = str(RESULTS / f"{name.lower()}_model")
    model.save(save_path)
    print(f"Saved: {save_path}.zip")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--n_envs",      type=int, default=8)
    parser.add_argument("--algo",        type=str, default="both",
                        choices=["ppo", "ac", "both"])
    args = parser.parse_args()

    if args.algo in ("ppo", "both"):
        train_agent(PPO, "PPO", args.total_steps, args.n_envs)

    if args.algo in ("ac", "both"):
        train_agent(A2C, "AC", args.total_steps, args.n_envs)

    print("\nAll training done.")


if __name__ == "__main__":
    main()
