"""
Scale generalisation experiment — parallel env version.

Runs B envs in parallel (default B=64) to saturate the 4090.
GPU handles B obs in one batched policy call; env steps run in
a ThreadPoolExecutor (Python GIL released during numpy).

Each method × N: 100 episodes, result appended to
  results/scale/{method}_N{n}.jsonl  (one JSON object per line)

Resume-safe: re-running skips already-completed episodes.

Usage:
  python scale_eval.py [--n_episodes 100] [--n_list 10,20,30,50,100] [--batch 64]
"""

import sys, json, time, argparse, pathlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(1, "/root/minecraft/dreamerv3")

import jax
import jax.numpy as jnp
import warnings
warnings.filterwarnings("ignore")

from envs.generalized_primitive_env import GeneralizedPrimitiveEnv
from cem_planner import build_agent, make_cem_fn, _make_obs

BATCH_SIZE = 64   # parallel envs; override via --batch

RESULTS_DIR = pathlib.Path(__file__).parent / "results" / "scale"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K_PRIMITIVES = 15  # kept same as env


# ------------------------------------------------------------------
# Single-episode runner (used by both serial and parallel paths)
# ------------------------------------------------------------------
def run_one(method, n_sheep, seed, agent, cem_fn, cem_params):
    env = GeneralizedPrimitiveEnv(n_sheep=n_sheep, seed=seed)
    obs_np = env.reset()
    done = False
    total_reward = 0.0
    macro_steps = 0
    decision_times = []

    carry = agent.init_policy(1) if method in ("reactive", "cem") else None
    rng   = jax.random.PRNGKey(seed) if method == "cem" else None

    while not done:
        t0 = time.perf_counter()

        if method == "reactive":
            obs_dict = _make_obs(obs_np, total_reward, macro_steps == 0)
            carry, acts, _ = agent.policy(carry, obs_dict)
            action = int(acts["action"][0])

        elif method == "cem":
            obs_dict = _make_obs(obs_np, total_reward, macro_steps == 0)
            carry, acts, _ = agent.policy(carry, obs_dict)
            dc = {"deter": jnp.array(carry[1]["deter"]),
                  "stoch": jnp.array(carry[1]["stoch"])}
            rng, sk = jax.random.split(rng)
            action = int(cem_fn(cem_params, dc, sk))

        elif method == "stromberg":
            spread = float(obs_np[4])
            frac   = float(obs_np[7])
            if spread > 0.12:   action = 0
            elif frac > 0.8:    action = 1
            else:               action = 2

        elif method == "random":
            action = int(np.random.randint(0, 12))

        dt = time.perf_counter() - t0
        decision_times.append(dt * 1000)
        obs_np, reward, done, info = env.step(action)
        total_reward += reward
        macro_steps += 1

    return {
        "success":          bool(info["fraction_in_goal"] >= 1.0),
        "fraction_in_goal": float(info["fraction_in_goal"]),
        "macro_steps":      macro_steps,
        "low_level_steps":  macro_steps * K_PRIMITIVES,
        "total_reward":     float(total_reward),
        "mean_decision_ms": float(np.mean(decision_times)),
        "n_sheep":          n_sheep,
        "method":           method,
        "seed":             seed,
    }


# ------------------------------------------------------------------
# Batched GPU runner for reactive/cem: B envs stepped together,
# env steps parallelised via ThreadPoolExecutor.
# ------------------------------------------------------------------
def _step_worker(args):
    env, action = args
    return env.step(int(action))


def run_batch_gpu(method, n_sheep, seeds, agent, cem_fn, cem_params, B):
    """
    Vectorised runner: run exactly B episodes simultaneously until ALL done,
    then return results.  No slot recycling — avoids carry reset complexity.
    B should equal len(seeds) for simplicity; call in chunks if needed.
    Env steps are parallelised across B threads (numpy releases the GIL).
    """
    B = len(seeds)   # always run all seeds together
    executor = ThreadPoolExecutor(max_workers=min(B, 64))

    envs       = [GeneralizedPrimitiveEnv(n_sheep=n_sheep, seed=int(s)) for s in seeds]
    obs_arr    = np.stack([e.reset() for e in envs])          # (B, 10)
    done_flags = np.zeros(B, bool)
    total_rews = np.zeros(B, np.float32)
    macro_steps= np.zeros(B, np.int32)
    dec_times  = [[] for _ in range(B)]
    last_infos = [{"fraction_in_goal": 0.0}] * B

    carry = agent.init_policy(B)
    rng   = jax.random.PRNGKey(int(seeds[0]))

    while not done_flags.all():
        obs_b = {
            "vector":      obs_arr.copy(),
            "reward":      total_rews.copy(),
            "is_first":    (macro_steps == 0),
            "is_last":     np.zeros(B, bool),
            "is_terminal": np.zeros(B, bool),
        }

        t0 = time.perf_counter()
        carry, acts, _ = agent.policy(carry, obs_b)
        gpu_dt = (time.perf_counter() - t0) * 1000 / max(1, (~done_flags).sum())

        if method == "cem":
            rng, *subkeys = jax.random.split(rng, B + 1)
            deter   = np.array(carry[1]["deter"])
            stoch   = np.array(carry[1]["stoch"])
            actions = np.array(acts["action"])
            for i in range(B):
                if not done_flags[i]:
                    dc = {"deter": jnp.array(deter[i])[None],
                          "stoch": jnp.array(stoch[i])[None]}
                    actions[i] = int(cem_fn(cem_params, dc, subkeys[i]))
        else:
            actions = np.array(acts["action"])

        for i in range(B):
            if not done_flags[i]:
                dec_times[i].append(gpu_dt)

        # Step all active envs in parallel threads
        step_args = [(envs[i], int(actions[i])) for i in range(B) if not done_flags[i]]
        active_idx = [i for i in range(B) if not done_flags[i]]
        step_outs  = list(executor.map(_step_worker, step_args))

        for j, i in enumerate(active_idx):
            obs_i, rew_i, done_i, info_i = step_outs[j]
            obs_arr[i]      = obs_i
            total_rews[i]  += rew_i
            macro_steps[i] += 1
            last_infos[i]   = info_i
            if done_i:
                done_flags[i] = True

    executor.shutdown(wait=False)

    results = []
    for i in range(B):
        info_i = last_infos[i]
        results.append({
            "success":          bool(info_i["fraction_in_goal"] >= 1.0),
            "fraction_in_goal": float(info_i["fraction_in_goal"]),
            "macro_steps":      int(macro_steps[i]),
            "low_level_steps":  int(macro_steps[i]) * K_PRIMITIVES,
            "total_reward":     float(total_rews[i]),
            "mean_decision_ms": float(np.mean(dec_times[i])) if dec_times[i] else 0.0,
            "n_sheep":          n_sheep,
            "method":           method,
            "seed":             int(seeds[i]),
        })
    return results


# ------------------------------------------------------------------
# Per (method, N) evaluation with resume
# ------------------------------------------------------------------
def evaluate(method, n_sheep, n_episodes, agent, cem_fn, cem_params,
             base_seed=42, batch_size=BATCH_SIZE):
    out_file = RESULTS_DIR / f"{method}_N{n_sheep}.jsonl"

    # Load already-completed
    done_seeds = set()
    completed  = []
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_seeds.add(r["seed"])
                completed.append(r)

    rng_np   = np.random.default_rng(base_seed + n_sheep)
    all_seeds = np.array([int(rng_np.integers(0, 2**31))
                          for _ in range(n_episodes)], dtype=np.int64)
    todo_seeds = np.array([s for s in all_seeds if s not in done_seeds])

    skipped = n_episodes - len(todo_seeds)
    if skipped > 0:
        print(f"  [{method} N={n_sheep}] resuming: {skipped}/{n_episodes} done")
    if len(todo_seeds) == 0:
        return completed

    with open(out_file, "a") as f:
        if method in ("stromberg", "random"):
            # Pure-CPU: multiprocess for full parallelism
            from concurrent.futures import ProcessPoolExecutor
            from parallel_worker import run_heuristic
            args = [(method, n_sheep, int(s)) for s in todo_seeds]
            nw   = min(batch_size, len(todo_seeds), 32)
            with ProcessPoolExecutor(max_workers=nw) as pool:
                batch_results = list(pool.map(run_heuristic, args))
        else:
            # GPU methods (reactive/cem): serial, flush after every episode
            batch_results = []
            for s in todo_seeds:
                r = run_one(method, n_sheep, int(s), agent, cem_fn, cem_params)
                batch_results.append(r)
                completed.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
            return completed  # already appended above

        for r in batch_results:
            completed.append(r)
            f.write(json.dumps(r) + "\n")
        f.flush()

    return completed


# ------------------------------------------------------------------
# Summary helpers
# ------------------------------------------------------------------
def summarize(results):
    succ  = [r["success"]          for r in results]
    steps = [r["macro_steps"]      for r in results]
    ll    = [r["low_level_steps"]  for r in results]
    dt    = [r["mean_decision_ms"] for r in results]
    succ_results = [r for r in results if r["success"]]
    succ_steps = [r["macro_steps"] for r in succ_results] if succ_results else [0]
    return {
        "success_rate":           float(np.mean(succ)),
        "mean_macro_steps":       float(np.mean(steps)),
        "mean_macro_steps_succ":  float(np.mean(succ_steps)),
        "mean_low_level_steps":   float(np.mean(ll)),
        "mean_decision_ms":       float(np.mean(dt)),
        "n_episodes":             len(results),
        "n_success":              int(sum(succ)),
    }


def print_progress(method, n_sheep, completed, n_episodes):
    s = summarize(completed)
    done = len(completed)
    print(f"  [{method:12s} N={n_sheep:3d}] {done:3d}/{n_episodes} "
          f"succ={s['success_rate']:.3f} "
          f"steps={s['mean_macro_steps']:.1f} "
          f"ll={s['mean_low_level_steps']:.0f} "
          f"dt={s['mean_decision_ms']:.1f}ms")


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
def plot_all(all_data, n_list, methods):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    colors = {"reactive": "steelblue", "cem": "darkorange",
              "stromberg": "forestgreen", "random": "tomato"}
    markers = {"reactive": "o", "cem": "s", "stromberg": "^", "random": "x"}
    labels  = {"reactive": "Reactive DreamerV3", "cem": "CEM-H8-N512 (zero-shot)",
               "stromberg": "Strömbom Heuristic", "random": "Random Primitives"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("success_rate",          "Success Rate",           "Success Rate"),
        ("mean_macro_steps_succ", "Mean Macro-Steps (succ.)", "Settling Time (macro-steps)"),
        ("mean_low_level_steps",  "Mean Low-Level Steps",   "Settling Time (low-level steps, ×15)"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for m in methods:
            ys = []
            for n in n_list:
                summary = all_data.get((m, n))
                if summary:
                    ys.append(summary.get(key, float("nan")))
                else:
                    ys.append(float("nan"))
            ax.plot(n_list, ys, marker=markers[m], color=colors[m],
                    label=labels[m], linewidth=2, markersize=7)
        ax.set_xlabel("Number of Sheep (N)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(n_list)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "scale_results.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved: {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--n_list",     type=str, default="10,20,30,50,100")
    parser.add_argument("--methods",    type=str, default="reactive,cem,stromberg,random")
    parser.add_argument("--batch",      type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    n_list  = [int(x) for x in args.n_list.split(",")]
    methods = args.methods.split(",")

    print("Building agent and loading checkpoint...")
    agent, _ = build_agent()

    cem_fn = cem_params = None
    if "cem" in methods:
        print("Compiling CEM H=8 N=512...")
        cem_fn, cem_params = make_cem_fn(
            agent, horizon=8, n_candidates=512, n_iter=3, top_k=64, n_actions=12)
        _dummy = {"deter": jnp.zeros((1, 512)), "stoch": jnp.zeros((1, 32, 4))}
        cem_fn(cem_params, _dummy, jax.random.PRNGKey(0))  # warm-up JIT
        print("CEM compiled.")

    all_data = {}
    summary_rows = []

    for n in n_list:
        print(f"\n{'='*55}")
        print(f"N = {n} sheep")
        print(f"{'='*55}")
        for method in methods:
            results = evaluate(
                method, n, args.n_episodes, agent, cem_fn, cem_params,
                batch_size=args.batch)
            print_progress(method, n, results, args.n_episodes)
            s = summarize(results)
            all_data[(method, n)] = s
            summary_rows.append({"method": method, "n_sheep": n, **s})

    # Save JSON summary
    out_json = RESULTS_DIR / "scale_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary saved: {out_json}")

    # Print table
    print(f"\n{'Method':12s} {'N':>5} {'Succ':>6} {'MacroSteps':>11} "
          f"{'LLSteps':>9} {'dt_ms':>7}")
    print("-" * 58)
    for row in summary_rows:
        print(f"{row['method']:12s} {row['n_sheep']:5d} "
              f"{row['success_rate']:6.3f} "
              f"{row['mean_macro_steps_succ']:11.1f} "
              f"{row['mean_low_level_steps']:9.0f} "
              f"{row['mean_decision_ms']:7.1f}")

    plot_all(all_data, n_list, methods)


if __name__ == "__main__":
    main()
