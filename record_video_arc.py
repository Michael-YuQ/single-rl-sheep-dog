"""
Record comparison video for the extended-action (arc) shepherding agent.
Methods compared side-by-side:
  1. stromberg  - rule-based heuristic
  2. reactive   - DreamerV3 policy (argmax, no lookahead)
  3. cem        - DreamerV3 + CEM planning in latent space

Output: results/shepherd_arc_comparison.mp4

Usage:
  python record_video_arc.py [--n_sheep 10] [--seed 42] [--max_steps 300]
"""
import sys, pathlib, warnings, argparse, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(1, "/root/minecraft/dreamerv3")

import numpy as np
import jax, jax.numpy as jnp
import elements
import ruamel.yaml as yaml
from PIL import Image, ImageDraw

from envs.generalized_primitive_env import GeneralizedPrimitiveEnv, N_ACTIONS
from envs.discrete_gym_env import DiscreteHerdingEnv
from dreamerv3.agent import Agent

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
FRAMES_DIR  = RESULTS_DIR / "tmp_frames_arc"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO   = None  # set dynamically in main()

CONFIG_BASE     = "/root/minecraft/dreamerv3/dreamerv3/configs.yaml"
CONFIG_SHEPHERD = str(pathlib.Path(__file__).parent / "configs/shepherd.yaml")

ACTION_NAMES = [
    "MUSTER", "HOLD",
    "DRIVE0","DRIVE1","DRIVE2","DRIVE3","DRIVE4",
    "DRIVE5","DRIVE6","DRIVE7","DRIVE8","DRIVE9",
    "MUSTER_ARC","FLANK_L","FLANK_R","SWEEP_BHD",
    "BYPASS0","BYPASS1","BYPASS2","BYPASS3","BYPASS4","BYPASS5",
]
ARC_ACTIONS = set(range(12, 22))

# ------------------------------------------------------------------
def find_latest_checkpoint():
    ckpt_dir = RESULTS_DIR / "dreamer_arc" / "ckpt"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoint dir: {ckpt_dir}")
    subdirs = sorted([d for d in ckpt_dir.iterdir() if (d / "agent.pkl").exists()])
    if not subdirs:
        raise FileNotFoundError("No checkpoints with agent.pkl found")
    return str(subdirs[-1] / "agent.pkl")


def build_agent():
    ckpt_path = find_latest_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")

    configs = yaml.YAML(typ="safe").load(elements.Path(CONFIG_BASE).read())
    configs.update(yaml.YAML(typ="safe").load(open(CONFIG_SHEPHERD).read()))
    config = elements.Config(configs["defaults"]).update(configs["shepherd"])

    from envs.shepherd_dreamer import ShepherdEnv
    env = ShepherdEnv(n_sheep=10)
    obs_space = dict(env.obs_space)
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}

    agent_cfg = elements.Config(
        **config.agent,
        logdir="/tmp/arc_agent",
        seed=0,
        jax={**dict(config.jax), "precompile": False, "prealloc": False,
             "transfer_guard": False},
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=0,
        replicas=1,
    )
    pathlib.Path("/tmp/arc_agent").mkdir(exist_ok=True)
    agent = Agent(obs_space, act_space, agent_cfg)

    ckpt_data = pickle.load(open(ckpt_path, "rb"))
    agent.load(ckpt_data)
    print(f"Loaded: {ckpt_data['counters']}")
    return agent, config


def make_cem_fn(agent, horizon=8, n_candidates=256, n_iter=3, top_k=32):
    import ninjax as nj

    model = agent.model
    dyn, rew, con = model.dyn, model.rew, model.con
    params_np = {k: np.array(v) for k, v in agent.params.items()}

    pure_imagine = nj.pure(dyn.imagine)
    pure_rew     = nj.pure(rew.__call__)
    pure_con     = nj.pure(con.__call__)

    def feat2tensor(feat):
        d = feat["deter"]
        s = feat["stoch"].reshape(d.shape[0], -1)
        return jnp.concatenate([d, s], axis=-1)

    def rollout(params, carry0, seq):
        def step(state, act):
            carry, total, disc = state
            _, (new_carry, (feat, _)) = pure_imagine(
                params, carry, {"action": act[None].astype(jnp.int32)},
                1, False, single=True, create=False, modify=False, seed=0)
            ft = feat2tensor(feat)
            _, rd = pure_rew(params, ft, 1, create=False, modify=False, seed=0)
            _, cd = pure_con(params, ft, 1, create=False, modify=False, seed=0)
            r  = rd.pred()[0].astype(jnp.float32)
            sv = jax.nn.sigmoid(cd.logit[0]).astype(jnp.float32)
            return (new_carry, total + disc * r, disc * 0.99 * sv), None
        (_, ret, _), _ = jax.lax.scan(step,
            (carry0, jnp.float32(0.0), jnp.float32(1.0)), seq)
        return ret

    def cem_plan(params, carry, rng):
        carry = jax.tree.map(lambda x: x.astype(jnp.bfloat16), carry)
        logits = jnp.zeros((horizon, N_ACTIONS), dtype=jnp.float32)

        def cem_iter(logits, key):
            probs = jax.nn.softmax(logits, axis=-1)
            keys = jax.random.split(key, horizon)
            seqs = jax.vmap(
                lambda k, p: jax.random.choice(k, N_ACTIONS, shape=(n_candidates,), p=p)
            )(keys, probs).T
            returns = jax.vmap(lambda s: rollout(params, carry, s))(seqs)
            top_idx = jnp.argsort(returns)[-top_k:]
            elite = seqs[top_idx]
            counts = jax.vmap(
                lambda h: jnp.zeros(N_ACTIONS).at[elite[:, h]].add(1.0)
            )(jnp.arange(horizon))
            return jnp.log(counts / top_k + 1e-8), returns[top_idx].mean()

        iter_keys = jax.random.split(rng, n_iter)
        logits, _ = jax.lax.scan(cem_iter, logits, iter_keys)
        return jnp.argmax(logits[0]).astype(jnp.int32)

    return jax.jit(cem_plan), params_np


def make_obs(obs, reward, is_first):
    return {
        "vector":      np.array([obs], dtype=np.float32),
        "reward":      np.array([reward], dtype=np.float32),
        "is_first":    np.array([is_first], dtype=bool),
        "is_last":     np.array([False], dtype=bool),
        "is_terminal": np.array([False], dtype=bool),
    }


# ------------------------------------------------------------------
ACTION_COLORS = {
    "arc":    (255, 200,  50),
    "direct": (100, 180, 255),
    "hold":   ( 50, 220, 120),
    "rule":   (200, 200, 200),
    "ppo":    (255, 140,   0),
    "ac":     (180,  80, 220),
}

PPO_MODEL_PATH = str(RESULTS_DIR / "ppo_model")
AC_MODEL_PATH  = str(RESULTS_DIR / "ac_model")

def load_sb3_model(path, algo="ppo"):
    import torch
    from stable_baselines3 import PPO, A2C
    cls = PPO if algo == "ppo" else A2C
    return cls.load(path, device="cpu")

def render_frame(env, title, step, frac, last_action, size=400):
    img = env.render("rgb_array")
    img = Image.fromarray(img).resize((size, size))
    draw = ImageDraw.Draw(img)

    method_key = title.lower()
    if method_key in ("ppo", "ac"):
        col = ACTION_COLORS[method_key]
        label = ACTION_NAMES[last_action] if last_action is not None and last_action < len(ACTION_NAMES) else ""
    elif last_action is not None and last_action in ARC_ACTIONS:
        col = ACTION_COLORS["arc"]
        label = f"ARC:{ACTION_NAMES[last_action]}"
    elif last_action == 1:
        col = ACTION_COLORS["hold"]
        label = "HOLD"
    elif last_action is not None:
        col = ACTION_COLORS["direct"]
        label = ACTION_NAMES[last_action] if last_action < len(ACTION_NAMES) else f"a{last_action}"
    else:
        col = (60, 60, 60)
        label = ""

    draw.rectangle([0, 0, size, 20], fill=(30, 30, 30))
    draw.text((4, 3), f"{title}  s={step}  {frac:.0%}", fill=(255, 255, 255))
    if label:
        draw.rectangle([0, size - 22, size, size], fill=col)
        draw.text((4, size - 19), label, fill=(20, 20, 20))
    return np.array(img)


def run_episode(method, agent, cem_fn, cem_params, n_sheep, seed, max_steps,
                sb3_model=None):
    env  = GeneralizedPrimitiveEnv(n_sheep=n_sheep, seed=seed)
    obs  = env.reset()
    done = False
    frames = []
    step = 0
    total_r = 0.0
    last_action = None

    carry = agent.init_policy(1) if agent is not None else None
    rng   = jax.random.PRNGKey(seed) if method in ("reactive", "cem") else None

    while not done and step < max_steps:
        frac = float(obs[7])
        frames.append(render_frame(env, method, step, frac, last_action))

        if method in ("ppo", "ac"):
            action, _ = sb3_model.predict(obs, deterministic=True)
            action = int(action)
        else:
            od = make_obs(obs, total_r, step == 0)
            carry, acts, _ = agent.policy(carry, od)

            if method == "reactive":
                action = int(acts["action"][0])
            elif method == "cem":
                dc = {
                    "deter": jnp.array(carry[1]["deter"]),
                    "stoch": jnp.array(carry[1]["stoch"]),
                }
                rng, sk = jax.random.split(rng)
                action = int(cem_fn(cem_params, dc, sk))
            elif method == "stromberg":
                spread = float(obs[4])
                if spread > 0.12:
                    action = 0
                elif frac > 0.8:
                    action = 1
                else:
                    action = 2

        last_action = action
        obs, r, done, info = env.step(action)
        total_r += r
        step += 1

    frames.append(render_frame(env, method, step, float(info["fraction_in_goal"]), last_action))
    print(f"  {method}: {step} steps, frac={info['fraction_in_goal']:.2f}, done={info['fraction_in_goal']>=1.0}")
    return frames, info


def frames_to_video(frames_dict, out_path, fps=10):
    import imageio
    methods  = list(frames_dict.keys())
    max_len  = max(len(f) for f in frames_dict.values())

    all_combined = []
    for i in range(max_len):
        panels = []
        for m in methods:
            fs = frames_dict[m]
            panels.append(fs[min(i, len(fs)-1)])
        combined = np.concatenate(panels, axis=1)
        all_combined.append(combined)

    imageio.mimwrite(str(out_path), all_combined, fps=fps, quality=8)
    print(f"Video saved: {out_path}")


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sheep",   type=int, default=10)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    print("Building agent...")
    agent, _ = build_agent()

    print("Compiling CEM...")
    cem_fn, cem_params = make_cem_fn(agent, horizon=8, n_candidates=256, n_iter=3, top_k=32)
    dummy = {
        "deter": jnp.zeros((1, agent.model.dyn.deter)),
        "stoch": jnp.zeros((1, agent.model.dyn.stoch, agent.model.dyn.classes)),
    }
    cem_fn(cem_params, dummy, jax.random.PRNGKey(0))
    print("CEM ready.")

    print("Loading PPO and A2C models...")
    ppo_model = load_sb3_model(PPO_MODEL_PATH, "ppo")
    ac_model  = load_sb3_model(AC_MODEL_PATH,  "ac")

    all_frames = {}
    sb3_map = {"ppo": ppo_model, "ac": ac_model}
    for method in ["stromberg", "reactive", "cem", "ppo", "ac"]:
        print(f"\nRecording {method} (N={args.n_sheep}, seed={args.seed})...")
        frames, info = run_episode(
            method, agent, cem_fn, cem_params,
            n_sheep=args.n_sheep, seed=args.seed, max_steps=args.max_steps,
            sb3_model=sb3_map.get(method))
        all_frames[method] = frames

    out_video = RESULTS_DIR / f"shepherd_arc_N{args.n_sheep}_seed{args.seed}.mp4"
    print("\nEncoding video...")
    frames_to_video(all_frames, out_video)
    print(f"\nDone! Video: {out_video}")


if __name__ == "__main__":
    main()
