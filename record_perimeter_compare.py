"""
Record side-by-side comparison:
  col1: CEM + original env (mixed arc+direct)
  col2: CEM + perimeter env (all-arc, no straight-through)
  col3: Stromberg heuristic (baseline)

Highlights what changes when the dog is forced to stay on the perimeter:
- Step count differences
- Success rate at N=10 / N=100
- Action pattern differences (should be ALL arc for perimeter)

Usage:
  python record_perimeter_compare.py [--n_sheep 10] [--seed 42] [--max_steps 400]
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
from dreamerv3.agent import Agent

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
FRAMES_DIR  = RESULTS_DIR / "tmp_frames_perim"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_BASE      = "/root/minecraft/dreamerv3/dreamerv3/configs.yaml"
CONFIG_ORIG      = str(pathlib.Path(__file__).parent / "configs/shepherd.yaml")
CONFIG_PERIMETER = str(pathlib.Path(__file__).parent / "configs/shepherd_perimeter.yaml")

ACTION_NAMES = [
    "MUSTER", "HOLD",
    "DRIVE0","DRIVE1","DRIVE2","DRIVE3","DRIVE4",
    "DRIVE5","DRIVE6","DRIVE7","DRIVE8","DRIVE9",
    "MUSTER_ARC","FLANK_L","FLANK_R","SWEEP_BHD",
    "BYPASS0","BYPASS1","BYPASS2","BYPASS3","BYPASS4","BYPASS5",
]
ARC_ACTIONS = set(range(12, 22))


def find_checkpoint(results_subdir):
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
        **config.agent,
        logdir="/tmp/perim_agent",
        seed=0,
        jax={**dict(config.jax), "precompile": False, "prealloc": False,
             "transfer_guard": False},
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=0, replicas=1,
    )
    pathlib.Path("/tmp/perim_agent").mkdir(exist_ok=True)
    agent = Agent(obs_space, act_space, agent_cfg)
    ckpt_data = pickle.load(open(ckpt_path, "rb"))
    agent.load(ckpt_data)
    print(f"  Loaded {config_name}: {ckpt_data['counters']}")
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


METHOD_COLORS = {
    "cem_orig":      (100, 180, 255),
    "cem_perimeter": (255, 120,  60),
    "stromberg":     (180, 180, 180),
}

def render_frame(env, title, step, frac, last_action, arc_count, total_steps, size=400):
    img = env.render("rgb_array")
    img = Image.fromarray(img).resize((size, size))
    draw = ImageDraw.Draw(img)

    col = METHOD_COLORS.get(title.lower(), (80, 80, 80))
    draw.rectangle([0, 0, size, 20], fill=(30, 30, 30))
    draw.text((4, 3), f"{title}  s={step}  {frac:.0%}", fill=(255,255,255))

    action_label = ""
    bar_col = col
    if last_action is not None:
        is_arc = last_action in ARC_ACTIONS
        action_label = ACTION_NAMES[last_action] if last_action < len(ACTION_NAMES) else f"a{last_action}"
        if is_arc:
            bar_col = (255, 200, 50)
            action_label = f"ARC:{action_label}"

    draw.rectangle([0, size-22, size, size], fill=bar_col)
    if total_steps > 0:
        arc_pct = arc_count / total_steps
        draw.text((4, size-19), f"{action_label}  arc={arc_pct:.0%}", fill=(20,20,20))
    return np.array(img)


def run_episode(label, env_cls, agent, cem_fn, cem_params, n_sheep, seed, max_steps):
    env  = env_cls(n_sheep=n_sheep, seed=seed)
    obs  = env.reset()
    done = False
    frames = []
    step = 0
    total_r = 0.0
    arc_count = 0
    last_action = None
    carry = agent.init_policy(1) if agent else None
    rng   = jax.random.PRNGKey(seed)

    while not done and step < max_steps:
        frac = float(obs[7])
        frames.append(render_frame(env, label, step, frac, last_action, arc_count, step))

        if label == "stromberg":
            spread = float(obs[4])
            if spread > 0.12:
                action = 0
            elif frac > 0.8:
                action = 1
            else:
                action = 2
        else:
            od = make_obs(obs, total_r, step == 0)
            carry, acts, _ = agent.policy(carry, od)
            dc = {
                "deter": jnp.array(carry[1]["deter"]),
                "stoch": jnp.array(carry[1]["stoch"]),
            }
            rng, sk = jax.random.split(rng)
            action = int(cem_fn(cem_params, dc, sk))

        last_action = action
        if action in ARC_ACTIONS:
            arc_count += 1
        obs, r, done, info = env.step(action)
        total_r += r
        step += 1

    frames.append(render_frame(env, label, step, float(info["fraction_in_goal"]),
                               last_action, arc_count, step))
    arc_pct = arc_count / max(step, 1)
    print(f"  {label}: {step} steps, frac={info['fraction_in_goal']:.2f}, "
          f"done={info['fraction_in_goal']>=1.0}, arc={arc_pct:.0%}")
    return frames, info


def frames_to_video(frames_dict, out_path, fps=10):
    import imageio
    methods  = list(frames_dict.keys())
    max_len  = max(len(f) for f in frames_dict.values())
    combined_frames = []
    for i in range(max_len):
        panels = [frames_dict[m][min(i, len(frames_dict[m])-1)] for m in methods]
        combined_frames.append(np.concatenate(panels, axis=1))
    imageio.mimwrite(str(out_path), combined_frames, fps=fps, quality=8)
    print(f"Video saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sheep",   type=int, default=10)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=400)
    args = parser.parse_args()

    print("Loading original CEM agent...")
    ckpt_orig = find_checkpoint("dreamer_arc")
    agent_orig, _ = build_agent(CONFIG_ORIG, ckpt_orig, "shepherd")

    print("Loading perimeter CEM agent...")
    ckpt_peri = find_checkpoint("dreamer_perimeter")
    agent_peri, _ = build_agent(CONFIG_PERIMETER, ckpt_peri, "shepherd_perimeter")

    print("Compiling CEM (orig)...")
    cem_orig, params_orig = make_cem_fn(agent_orig)
    cem_peri, params_peri = make_cem_fn(agent_peri)
    dummy = {
        "deter": jnp.zeros((1, agent_orig.model.dyn.deter)),
        "stoch": jnp.zeros((1, agent_orig.model.dyn.stoch, agent_orig.model.dyn.classes)),
    }
    cem_orig(params_orig, dummy, jax.random.PRNGKey(0))
    cem_peri(params_peri, dummy, jax.random.PRNGKey(0))
    print("CEM ready.")

    configs = [
        ("cem_orig",      GeneralizedPrimitiveEnv, agent_orig, cem_orig, params_orig),
        ("cem_perimeter", PerimeterEnv,            agent_peri, cem_peri, params_peri),
        ("stromberg",     GeneralizedPrimitiveEnv, None,       None,     None),
    ]

    all_frames = {}
    for label, env_cls, agent, cem_fn, cem_params in configs:
        print(f"\nRecording {label} (N={args.n_sheep}, seed={args.seed})...")
        frames, info = run_episode(
            label, env_cls, agent, cem_fn, cem_params,
            n_sheep=args.n_sheep, seed=args.seed, max_steps=args.max_steps)
        all_frames[label] = frames

    out_video = RESULTS_DIR / f"perimeter_compare_N{args.n_sheep}_seed{args.seed}.mp4"
    print("\nEncoding video...")
    frames_to_video(all_frames, out_video)
    print(f"\nDone! {out_video}")


if __name__ == "__main__":
    main()
