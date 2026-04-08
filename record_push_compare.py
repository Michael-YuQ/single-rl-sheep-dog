"""
Compare three methods on N=100 herding:
  col1: perimeter_zero_shot  — perimeter model (trained N=10) zero-shot to N=100
  col2: push_N100            — push-track model trained directly at N=100
  col3: stromberg            — rule-based baseline

Usage:
  python record_push_compare.py [--seed 7] [--max_steps 600]
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
from envs.perimeter_env import PerimeterEnv
from envs.push_env import PushEnv
from dreamerv3.agent import Agent

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
FRAMES_DIR  = RESULTS_DIR / "tmp_frames_push"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_BASE      = "/root/minecraft/dreamerv3/dreamerv3/configs.yaml"
CONFIG_PERIMETER = str(pathlib.Path(__file__).parent / "configs/shepherd_perimeter.yaml")
CONFIG_PUSH100   = str(pathlib.Path(__file__).parent / "configs/shepherd_push100.yaml")

ACTION_NAMES = [
    "PUSH_TRACK","HOLD",
    "DRIVE0","DRIVE1","DRIVE2","DRIVE3","DRIVE4",
    "DRIVE5","DRIVE6","DRIVE7","DRIVE8","DRIVE9",
    "MUSTER_ARC","FLANK_L","FLANK_R","SWEEP_BHD",
    "BYPASS0","BYPASS1","BYPASS2","BYPASS3","BYPASS4","BYPASS5",
]


def find_ckpt(results_subdir):
    ckpt_dir = RESULTS_DIR / results_subdir / "ckpt"
    subdirs = sorted([d for d in ckpt_dir.iterdir() if (d / "agent.pkl").exists()])
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return str(subdirs[-1] / "agent.pkl")


def build_agent(config_path, ckpt_path, config_name):
    configs = yaml.YAML(typ="safe").load(elements.Path(CONFIG_BASE).read())
    configs.update(yaml.YAML(typ="safe").load(open(config_path).read()))
    config = elements.Config(configs["defaults"]).update(configs[config_name])

    from envs.shepherd_dreamer import ShepherdEnv
    env_tmp = ShepherdEnv(n_sheep=10)
    obs_space = dict(env_tmp.obs_space)
    act_space = {k: v for k, v in env_tmp.act_space.items() if k != "reset"}

    agent_cfg = elements.Config(
        **config.agent, logdir="/tmp/push_agent", seed=0,
        jax={**dict(config.jax), "precompile": False, "prealloc": False,
             "transfer_guard": False},
        batch_size=config.batch_size, batch_length=config.batch_length,
        replay_context=config.replay_context, report_length=config.report_length,
        replica=0, replicas=1,
    )
    pathlib.Path("/tmp/push_agent").mkdir(exist_ok=True)
    agent = Agent(obs_space, act_space, agent_cfg)
    ckpt_data = pickle.load(open(ckpt_path, "rb"))
    agent.load(ckpt_data)
    print(f"  {config_name}: {ckpt_data['counters']}")
    return agent


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
            keys  = jax.random.split(key, horizon)
            seqs  = jax.vmap(
                lambda k, p: jax.random.choice(k, N_ACTIONS, shape=(n_candidates,), p=p)
            )(keys, probs).T
            returns = jax.vmap(lambda s: rollout(params, carry, s))(seqs)
            top_idx = jnp.argsort(returns)[-top_k:]
            elite   = seqs[top_idx]
            counts  = jax.vmap(
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


COLORS = {
    "perimeter_zero_shot": (180,  80, 220),
    "push_N100":           (255, 140,   0),
    "stromberg":           (160, 160, 160),
}

def render_frame(env, label, step, frac, last_action, size=400):
    img = env.render("rgb_array")
    img = Image.fromarray(img).resize((size, size))
    draw = ImageDraw.Draw(img)
    col = COLORS.get(label, (80, 80, 80))
    draw.rectangle([0, 0, size, 20], fill=(30, 30, 30))
    draw.text((4, 3), f"{label}  s={step}  {frac:.0%}", fill=(255, 255, 255))
    if last_action is not None:
        name = ACTION_NAMES[last_action] if last_action < len(ACTION_NAMES) else f"a{last_action}"
        draw.rectangle([0, size-22, size, size], fill=col)
        draw.text((4, size-19), name, fill=(20, 20, 20))
    return np.array(img)


def run_episode(label, env_cls, agent, cem_fn, cem_params, n_sheep, seed, max_steps):
    env  = env_cls(n_sheep=n_sheep, seed=seed)
    obs  = env.reset()
    done = False
    frames = []
    step, total_r = 0, 0.0
    last_action = None
    carry = agent.init_policy(1) if agent else None
    rng = jax.random.PRNGKey(seed)

    while not done and step < max_steps:
        frac = float(obs[7])
        frames.append(render_frame(env, label, step, frac, last_action))

        if label == "stromberg":
            spread = float(obs[4])
            action = 0 if spread > 0.12 else (1 if frac > 0.8 else 2)
        else:
            od = make_obs(obs, total_r, step == 0)
            carry, acts, _ = agent.policy(carry, od)
            dc = {"deter": jnp.array(carry[1]["deter"]),
                  "stoch": jnp.array(carry[1]["stoch"])}
            rng, sk = jax.random.split(rng)
            action = int(cem_fn(cem_params, dc, sk))

        last_action = action
        obs, r, done, info = env.step(action)
        total_r += r
        step += 1

    frames.append(render_frame(env, label, step, float(info["fraction_in_goal"]), last_action))
    print(f"  {label}: {step} steps, frac={info['fraction_in_goal']:.2f}, "
          f"done={info['fraction_in_goal']>=1.0}")
    return frames, info


def frames_to_video(frames_dict, out_path, fps=10):
    import imageio
    methods = list(frames_dict.keys())
    max_len = max(len(f) for f in frames_dict.values())
    out = []
    for i in range(max_len):
        panels = [frames_dict[m][min(i, len(frames_dict[m])-1)] for m in methods]
        out.append(np.concatenate(panels, axis=1))
    imageio.mimwrite(str(out_path), out, fps=fps, quality=8)
    print(f"Video saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sheep",   type=int, default=100)
    parser.add_argument("--seed",      type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=600)
    args = parser.parse_args()

    print("Loading perimeter (N=10 zero-shot) agent...")
    agent_peri = build_agent(CONFIG_PERIMETER, find_ckpt("dreamer_perimeter"), "shepherd_perimeter")
    print("Loading push N=100 agent...")
    agent_push = build_agent(CONFIG_PUSH100, find_ckpt("dreamer_push100"), "shepherd_push100")

    print("Compiling CEM...")
    cem_peri, params_peri = make_cem_fn(agent_peri)
    cem_push, params_push = make_cem_fn(agent_push)
    dummy = {"deter": jnp.zeros((1, agent_peri.model.dyn.deter)),
             "stoch": jnp.zeros((1, agent_peri.model.dyn.stoch, agent_peri.model.dyn.classes))}
    cem_peri(params_peri, dummy, jax.random.PRNGKey(0))
    cem_push(params_push, dummy, jax.random.PRNGKey(0))
    print("CEM ready.")

    configs = [
        ("perimeter_zero_shot", PerimeterEnv, agent_peri, cem_peri, params_peri),
        ("push_N100",           PushEnv,      agent_push, cem_push, params_push),
        ("stromberg",           PushEnv,      None,       None,     None),
    ]

    all_frames = {}
    for label, env_cls, agent, cem_fn, cem_params in configs:
        print(f"\nRecording {label}...")
        frames, _ = run_episode(label, env_cls, agent, cem_fn, cem_params,
                                n_sheep=args.n_sheep, seed=args.seed, max_steps=args.max_steps)
        all_frames[label] = frames

    out = RESULTS_DIR / f"push_compare_N{args.n_sheep}_seed{args.seed}.mp4"
    frames_to_video(all_frames, out)
    print(f"\nDone! {out}")


if __name__ == "__main__":
    main()
