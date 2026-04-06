"""
Record side-by-side comparison video: CEM vs Reactive vs Stromberg for N=50.
Output: results/shepherd_N50_comparison.mp4
"""
import sys, pathlib, warnings, subprocess, tempfile, os
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(1, "/root/minecraft/dreamerv3")

import numpy as np
import jax, jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont

from envs.generalized_primitive_env import GeneralizedPrimitiveEnv
from cem_planner import build_agent, make_cem_fn, _make_obs

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
OUT_VIDEO   = RESULTS_DIR / "shepherd_N50_comparison.mp4"
FRAMES_DIR  = RESULTS_DIR / "tmp_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

FPS     = 10   # frames per second in output video
N_SHEEP = 50
SEED    = 7
MAX_MACRO = 200   # record at most this many macro-steps per method


# ------------------------------------------------------------------
def render_frame(env, title, step, frac, size=400):
    """Render one frame with title + stats overlay."""
    img = env.render("rgb_array")
    img = Image.fromarray(img).resize((size, size))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size, 18], fill=(30, 30, 30))
    draw.text((4, 2), f"{title}  step={step}  in_goal={frac:.0%}", fill=(255, 255, 255))
    return np.array(img)


def run_and_record(method_name, agent, cem_fn, cem_params, seed):
    env  = GeneralizedPrimitiveEnv(n_sheep=N_SHEEP, seed=seed)
    obs  = env.reset()
    done = False
    frames = []
    macro_steps = 0
    total_reward = 0.0
    carry = agent.init_policy(1)
    rng   = jax.random.PRNGKey(seed)

    while not done and macro_steps < MAX_MACRO:
        frac = float(obs[7])
        frames.append(render_frame(env, method_name, macro_steps, frac))

        if method_name == "reactive":
            od = _make_obs(obs, total_reward, macro_steps == 0)
            carry, acts, _ = agent.policy(carry, od)
            action = int(acts["action"][0])

        elif method_name == "cem":
            od = _make_obs(obs, total_reward, macro_steps == 0)
            carry, acts, _ = agent.policy(carry, od)
            dc = {"deter": jnp.array(carry[1]["deter"]),
                  "stoch": jnp.array(carry[1]["stoch"])}
            rng, sk = jax.random.split(rng)
            action = int(cem_fn(cem_params, dc, sk))

        elif method_name == "stromberg":
            sp   = env._env.sheep_pos
            spread = float(obs[4])
            if spread > 0.12:
                action = 0
            elif frac > 0.8:
                action = 1
            else:
                action = 2

        obs, r, done, info = env.step(action)
        total_reward += r
        macro_steps += 1

    # Final frame
    frames.append(render_frame(env, method_name, macro_steps,
                               float(info["fraction_in_goal"])))
    return frames, info


# ------------------------------------------------------------------
def frames_to_video(all_frames_dict, out_path):
    """Stitch methods side by side and write mp4."""
    methods  = list(all_frames_dict.keys())
    max_len  = max(len(f) for f in all_frames_dict.values())
    h, w, _  = list(all_frames_dict.values())[0][0].shape

    frame_paths = []
    for i in range(max_len):
        panels = []
        for m in methods:
            frames = all_frames_dict[m]
            idx    = min(i, len(frames) - 1)
            panels.append(frames[idx])
        combined = np.concatenate(panels, axis=1)   # side by side
        img_path = FRAMES_DIR / f"frame_{i:05d}.png"
        Image.fromarray(combined).save(img_path)
        frame_paths.append(str(img_path))

    # ffmpeg
    cmd = [
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(FRAMES_DIR / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", str(out_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Video saved: {out_path}")

    # Cleanup frames
    for p in FRAMES_DIR.glob("frame_*.png"):
        p.unlink()


# ------------------------------------------------------------------
def main():
    print("Building agent...")
    agent, _ = build_agent()

    print("Compiling CEM H=8 N=512...")
    cem_fn, cem_params = make_cem_fn(
        agent, horizon=8, n_candidates=512, n_iter=3, top_k=64, n_actions=12)
    _dummy = {"deter": jnp.zeros((1, 512)), "stoch": jnp.zeros((1, 32, 4))}
    cem_fn(cem_params, _dummy, jax.random.PRNGKey(0))
    print("CEM ready.")

    all_frames = {}
    for method in ["stromberg", "reactive", "cem"]:
        print(f"Recording {method} N={N_SHEEP}...")
        frames, info = run_and_record(method, agent, cem_fn, cem_params, seed=SEED)
        all_frames[method] = frames
        print(f"  {method}: {len(frames)} frames, "
              f"success={info['fraction_in_goal']>=1.0}, "
              f"frac={info['fraction_in_goal']:.2f}")

    print("Encoding video...")
    frames_to_video(all_frames, OUT_VIDEO)


if __name__ == "__main__":
    main()
