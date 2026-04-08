"""
Record 3-dog joint action RSSM reactive policy video.
Decodes joint action back to per-dog primitives and shows them.
"""
import sys, pathlib, warnings, pickle, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(1, "/root/minecraft/dreamerv3")

import numpy as np
import elements, ruamel.yaml as yaml
from PIL import Image, ImageDraw
from dreamerv3.agent import Agent
from envs.three_dog_env import ThreeDogEnv, N_JOINT, OBS_DIM, N_PER_DOG

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
CONFIG_BASE  = "/root/minecraft/dreamerv3/dreamerv3/configs.yaml"
CONFIG_3DOG  = str(pathlib.Path(__file__).parent / "configs/shepherd_threedog.yaml")

DOG_COLORS   = [(255, 80, 80), (80, 200, 80), (80, 80, 255)]
ACTION_NAMES = ["PUSH", "FLANK_L", "FLANK_R", "HOLD", "COLLECT"]
ACTION_COLS  = {
    "PUSH":    (255, 200, 50),
    "FLANK_L": (100, 220, 255),
    "FLANK_R": (100, 180, 255),
    "HOLD":    (50,  220, 120),
    "COLLECT": (255,  60,  60),
}


def build_agent():
    configs = yaml.YAML(typ="safe").load(elements.Path(CONFIG_BASE).read())
    configs.update(yaml.YAML(typ="safe").load(open(CONFIG_3DOG).read()))
    config = elements.Config(configs["defaults"]).update(configs["shepherd_threedog"])

    import elements as el
    obs_space = {"vector": el.Space(np.float32, (OBS_DIM,)),
                 "reward": el.Space(np.float32, ()),
                 "is_first": el.Space(bool, ()), "is_last": el.Space(bool, ()),
                 "is_terminal": el.Space(bool, ())}
    act_space = {"action": el.Space(np.int32, (), 0, N_JOINT)}

    pathlib.Path("/tmp/td_rec").mkdir(exist_ok=True)
    agent_cfg = elements.Config(**config.agent, logdir="/tmp/td_rec", seed=0,
        jax={**dict(config.jax), "precompile": False, "prealloc": False,
             "transfer_guard": False},
        batch_size=config.batch_size, batch_length=config.batch_length,
        replay_context=config.replay_context, report_length=config.report_length,
        replica=0, replicas=1)
    agent = Agent(obs_space, act_space, agent_cfg)

    ckpt_dir = RESULTS_DIR / "dreamer_threedog" / "ckpt"
    subdirs  = sorted([d for d in ckpt_dir.iterdir() if (d / "agent.pkl").exists()])
    data = pickle.load(open(str(subdirs[-1] / "agent.pkl"), "rb"))
    agent.load(data)
    print(f"Loaded: {data['counters']}")
    return agent


def decode(joint):
    return joint // (N_PER_DOG**2), (joint // N_PER_DOG) % N_PER_DOG, joint % N_PER_DOG


def render_frame(env, step, frac, joint_action, size=500):
    img = env.render("rgb_array")
    img = Image.fromarray(img).resize((size, size))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, size, 22], fill=(20, 20, 20))
    draw.text((4, 4), f"3-Dog RSSM  step={step}  {frac:.1%} in goal",
              fill=(255, 255, 255))

    if joint_action is not None:
        a0, a1, a2 = decode(joint_action)
        names = [ACTION_NAMES[a0], ACTION_NAMES[a1], ACTION_NAMES[a2]]
        cols  = [DOG_COLORS[i] for i in range(3)]
        panel_w = size // 3
        for i, (name, dc, ac) in enumerate(zip(names, cols, [ACTION_COLS[n] for n in names])):
            x0, x1 = i * panel_w, (i + 1) * panel_w
            draw.rectangle([x0, size - 22, x1, size], fill=ac)
            draw.text((x0 + 4, size - 19), f"D{i}:{name}", fill=(20, 20, 20))

    return np.array(img)


def run_episode(agent, n_sheep=100, seed=7, max_steps=2000):
    env    = ThreeDogEnv(n_sheep=n_sheep, seed=seed)
    obs    = env.reset()
    carry  = agent.init_policy(1)
    done   = False
    step   = 0
    frames = []
    last_action = None

    while not done and step < max_steps:
        frac = env._env._fraction_in_goal()
        frames.append(render_frame(env, step, frac, last_action))

        od = {"vector": np.array([obs], dtype=np.float32),
              "reward": np.array([0.], dtype=np.float32),
              "is_first": np.array([step == 0], dtype=bool),
              "is_last": np.array([False], dtype=bool),
              "is_terminal": np.array([False], dtype=bool)}
        carry, acts, _ = agent.policy(carry, od)
        action = int(acts["action"][0])
        last_action = action
        obs, r, done, info = env.step(action)
        step += 1

    frames.append(render_frame(env, step, info["fraction_in_goal"], last_action))
    return frames, info, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sheep",   type=int, default=100)
    parser.add_argument("--seed",      type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--out_dir",   type=str,
        default="/inspire/hdd/global_user/yuqi-253114050256/sheep/threedog")
    args = parser.parse_args()

    import imageio
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Building agent...")
    agent = build_agent()

    for seed in [args.seed] if args.seed != -1 else range(10):
        print(f"\nRecording seed={seed}...")
        frames, info, steps = run_episode(agent, args.n_sheep, seed, args.max_steps)
        frac = info["fraction_in_goal"]
        print(f"  steps={steps}  frac={frac:.3f}  done={frac>=1.0}")
        out = pathlib.Path(args.out_dir) / f"threedog_N{args.n_sheep}_seed{seed}.mp4"
        imageio.mimwrite(str(out), frames, fps=15, quality=8)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
