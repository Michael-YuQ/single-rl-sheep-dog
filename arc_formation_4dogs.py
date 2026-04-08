"""
4-dog arc formation controller for herding 100 sheep.

Formation strategy (inspired by Li et al. 2023 + Van Havermaet et al. 2024):
  - 4 dogs are evenly spaced on a semicircular arc behind the flock
  - Arc center = flock centroid, arc radius = flock_spread * R_FACTOR
  - Arc opens toward the goal (dogs are on the opposite side)
  - Each dog is assigned one of 4 equally-spaced arc positions
  - Dogs move to maintain their arc position while the flock moves

Additionally, straggler collection: when a sheep is far from the flock (>2*spread),
the nearest dog temporarily breaks formation to collect it.

Usage:
  python arc_formation_4dogs.py [--n_sheep 100] [--seed 7] [--max_steps 800]
  python arc_formation_4dogs.py --eval --n_seeds 20
"""
import sys, pathlib, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from PIL import Image, ImageDraw

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
FRAMES_DIR  = RESULTS_DIR / "tmp_frames_arc4"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

R_FACTOR      = 1.5    # arc radius = spread * R_FACTOR
ARC_OFFSET    = 4.0    # extra metres beyond spread
STRAGGLER_THR = 2.0    # sheep is straggler if >spread*STRAGGLER_THR from centroid
DOG_SPEED     = 2.0


def arc_positions(centroid, goal, spread, n_dogs=4):
    """
    Compute n_dogs equally-spaced positions on a semicircular arc
    behind the flock (away from goal).
    Arc spans from -70° to +70° around the "away from goal" axis.
    """
    to_goal   = goal - centroid
    to_goal_n = np.linalg.norm(to_goal) + 1e-8
    behind    = -to_goal / to_goal_n          # unit vector away from goal

    r = spread * R_FACTOR + ARC_OFFSET
    angles = np.linspace(-np.pi * 0.6, np.pi * 0.6, n_dogs)  # -108° to +108°

    perp = np.array([-behind[1], behind[0]])  # perpendicular
    positions = []
    for a in angles:
        pos = centroid + r * (np.cos(a) * behind + np.sin(a) * perp)
        positions.append(pos)
    return positions


def assign_dogs_to_positions(dog_positions, arc_pos):
    """Greedy assignment: minimize total distance dog→arc_position."""
    from itertools import permutations
    best_cost = float('inf')
    best_perm = list(range(len(arc_pos)))
    for perm in permutations(range(len(arc_pos))):
        cost = sum(np.linalg.norm(dog_positions[i] - arc_pos[perm[i]])
                   for i in range(len(arc_pos)))
        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)
    # perm[i] = which arc slot dog i goes to
    # return targets[i] = arc_pos for dog i
    targets = [arc_pos[best_perm[i]] for i in range(len(arc_pos))]
    return targets


def move_toward(current, target, speed=DOG_SPEED, arena=100.0):
    delta = target - current
    dist  = np.linalg.norm(delta) + 1e-8
    vel   = delta / dist * min(1.0, dist / speed)  # unit vector
    return vel.astype(np.float32)


def control_step(dog_pos, sheep_pos, goal, goal_radius, n_dogs=4):
    """
    Compute velocity commands for all dogs.
    Returns actions array of shape (n_dogs, 2).
    """
    sp       = sheep_pos
    centroid = sp.mean(axis=0)
    spread   = float(np.std(sp, axis=0).mean()) + 1.0

    # Identify stragglers (outside goal, far from flock)
    outside_mask   = np.linalg.norm(sp - goal, axis=1) > goal_radius
    outside        = sp[outside_mask]
    straggler_idx  = None
    if len(outside) > 0:
        d_from_centroid = np.linalg.norm(outside - centroid, axis=1)
        if d_from_centroid.max() > spread * STRAGGLER_THR:
            straggler_idx = np.argmax(d_from_centroid)
            straggler_pos = outside[straggler_idx]

    # Compute arc positions
    arc_pos = arc_positions(centroid, goal, spread, n_dogs)

    # Assign dogs to arc positions
    targets = assign_dogs_to_positions(dog_pos, arc_pos)

    # If there's a straggler, nearest dog goes to collect it
    if straggler_idx is not None:
        dists_to_straggler = np.linalg.norm(dog_pos - straggler_pos, axis=1)
        collector = int(np.argmin(dists_to_straggler))
        # Position behind straggler (away from centroid)
        away = straggler_pos - centroid
        away_n = np.linalg.norm(away) + 1e-8
        targets[collector] = straggler_pos + away / away_n * 3.0

    actions = np.array([move_toward(dog_pos[i], targets[i])
                        for i in range(n_dogs)], dtype=np.float32)
    return actions


# -----------------------------------------------------------------------
def render_frame(env, step, frac, size=500):
    img = env.render("rgb_array")
    img = Image.fromarray(img).resize((size, size))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size, 22], fill=(20, 20, 20))
    draw.text((4, 4), f"4-Dog Arc Formation  step={step}  {frac:.1%} in goal",
              fill=(255, 255, 255))
    return np.array(img)


def run_episode(n_sheep=100, seed=7, max_steps=800, record=True):
    from envs.sheep_env import SheepEnv
    env = SheepEnv(n_sheep=n_sheep, n_dogs=4, max_steps=max_steps,
                   arena_size=100.0, seed=seed)
    obs = env.reset()
    done = False
    step = 0
    frames = []
    frac_history = []

    while not done and step < max_steps:
        frac = env._fraction_in_goal()
        frac_history.append(frac)
        if record:
            frames.append(render_frame(env, step, frac))

        actions = control_step(
            env.dog_pos, env.sheep_pos, env.goal,
            env.goal_radius, n_dogs=4)
        obs, r, done, info = env.step(actions)
        step += 1

    frac = info["fraction_in_goal"]
    if record:
        frames.append(render_frame(env, step, frac))
    return frames, info, step


def frames_to_video(frames, out_path, fps=15):
    import imageio
    imageio.mimwrite(str(out_path), frames, fps=fps, quality=8)
    print(f"Video saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sheep",   type=int, default=100)
    parser.add_argument("--seed",      type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--eval",      action="store_true")
    parser.add_argument("--n_seeds",   type=int, default=20)
    args = parser.parse_args()

    if args.eval:
        print(f"Evaluating 4-dog arc formation on N={args.n_sheep}, {args.n_seeds} seeds...")
        successes, steps_list, fracs = [], [], []
        for seed in range(args.n_seeds):
            _, info, steps = run_episode(args.n_sheep, seed, args.max_steps, record=False)
            f = info["fraction_in_goal"]
            successes.append(f >= 1.0)
            steps_list.append(steps)
            fracs.append(f)
            print(f"  seed={seed:3d}: steps={steps:4d}  frac={f:.3f}  {'OK' if f>=1.0 else '--'}")
        print(f"\nSuccess: {sum(successes)}/{args.n_seeds} ({np.mean(successes):.0%})")
        print(f"Mean steps: {np.mean(steps_list):.0f}  Mean frac: {np.mean(fracs):.3f}")
    else:
        print(f"Recording 4-dog arc formation N={args.n_sheep} seed={args.seed}...")
        frames, info, steps = run_episode(args.n_sheep, args.seed, args.max_steps)
        frac = info["fraction_in_goal"]
        print(f"  steps={steps}  frac={frac:.3f}  done={frac>=1.0}")
        out = RESULTS_DIR / f"arc4_N{args.n_sheep}_seed{args.seed}.mp4"
        frames_to_video(frames, out)


if __name__ == "__main__":
    main()
