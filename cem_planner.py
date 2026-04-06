"""
CEM-in-Latent-Space planner for the shepherding task.

Loads frozen DreamerV3 world model (enc, dyn, rew, con) from checkpoint,
then runs Cross-Entropy Method (CEM) over macro-action sequences inside
the RSSM latent space.

Comparison matrix:
  - reactive      : DreamerV3 policy (argmax), no lookahead
  - cem_H5_N256   : CEM, horizon=5,  candidates=256
  - cem_H8_N512   : CEM, horizon=8,  candidates=512
  - cem_H12_N512  : CEM, horizon=12, candidates=512
  - stromberg     : rule-based heuristic (loaded from prior results)

Resume safety: each method appends completed episodes to its own JSONL
file under results/cem/. Re-running skips already-completed episodes.

Usage:
  python cem_planner.py [--n_episodes 200] [--methods reactive,cem_H5_N256,...]
"""

import sys, os, argparse, json, pickle, time, pathlib, math

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(1, "/root/minecraft/dreamerv3")

import numpy as np
import jax
import jax.numpy as jnp
import ninjax as nj
import elements
import ruamel.yaml as yaml

from dreamerv3.agent import Agent
from envs.shepherd_dreamer import ShepherdEnv
from envs.primitive_env import PrimitiveEnv

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
CKPT = "/root/shepherd/results/hierarchical_dreamer/ckpt/20260402T073624F267996/agent.pkl"
CONFIG_BASE = "/root/minecraft/dreamerv3/dreamerv3/configs.yaml"
CONFIG_SHEPHERD = str(pathlib.Path(__file__).parent / "configs/shepherd.yaml")
RESULTS_DIR = pathlib.Path(__file__).parent / "results" / "cem"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Build Agent + load checkpoint
# ------------------------------------------------------------------
def build_agent():
    configs = yaml.YAML(typ="safe").load(elements.Path(CONFIG_BASE).read())
    configs.update(yaml.YAML(typ="safe").load(open(CONFIG_SHEPHERD).read()))
    config = elements.Config(configs["defaults"]).update(configs["shepherd"])

    env = ShepherdEnv()
    obs_space = dict(env.obs_space)
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}

    agent_cfg = elements.Config(
        **config.agent,
        logdir="/tmp/cem_agent",
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
    pathlib.Path("/tmp/cem_agent").mkdir(exist_ok=True)

    agent = Agent(obs_space, act_space, agent_cfg)

    # Load checkpoint parameters
    ckpt_data = pickle.load(open(CKPT, "rb"))
    agent.load(ckpt_data)
    print(f"Loaded checkpoint: {ckpt_data['counters']}")
    return agent, config


# ------------------------------------------------------------------
# CEM planner: pure JAX, JIT-compiled
# ------------------------------------------------------------------
def make_cem_fn(agent, horizon: int, n_candidates: int, n_iter: int, top_k: int,
                n_actions: int = 12):
    """
    Returns (jit_fn, params_dict).

    jit_fn(params, dyn_carry, rng_key) -> best_action (int scalar)

    dyn_carry: {'deter': (1, deter), 'stoch': (1, stoch, classes)}

    Strategy: use nj.pure with the *full* ninjax state (params injected)
    to call dyn.imagine, rew, con in a single scan loop.
    """
    model = agent.model
    dyn   = model.dyn
    rew   = model.rew
    con   = model.con

    # Snapshot params as plain numpy (preserving original dtype, typically bfloat16)
    params_np = {k: np.array(v, dtype=np.array(v).dtype) for k, v in agent.params.items()}

    # --- nj.pure wrappers (state = params dict) ---
    pure_imagine = nj.pure(dyn.imagine)
    pure_rew     = nj.pure(rew.__call__)
    pure_con     = nj.pure(con.__call__)

    def feat2tensor(feat):
        deter = feat["deter"]                          # (B, deter)
        stoch = feat["stoch"].reshape(deter.shape[0], -1)  # (B, stoch*classes)
        return jnp.concatenate([deter, stoch], axis=-1)

    def _rollout_return(params, carry0, action_seq):
        """Rollout H steps, return discounted sum of rew*(1-done)."""

        def step_fn(state, act_idx):
            carry, total, disc = state
            # DictConcat handles int -> one_hot internally; pass (B,) int32
            act_dict = {"action": act_idx[None].astype(jnp.int32)}  # (1,)

            # nj.pure returns (new_state, fn_return)
            # dyn.imagine single=True returns (new_carry, (feat, action))
            _, (new_carry, (feat, _)) = pure_imagine(
                params, carry, act_dict, 1, False, single=True,
                create=False, modify=False, seed=0)

            ft = feat2tensor(feat)   # (1, feat_dim)

            # rew returns TwoHot dist; .pred() → (1,)
            _, rew_dist = pure_rew(params, ft, 1,
                                   create=False, modify=False, seed=0)
            # con returns Binary dist; .logit is an array attribute
            _, con_dist = pure_con(params, ft, 1,
                                   create=False, modify=False, seed=0)

            r  = rew_dist.pred()[0].astype(jnp.float32)       # scalar
            sv = jax.nn.sigmoid(con_dist.logit[0]).astype(jnp.float32)  # survival

            total = total + disc * r
            disc  = disc  * 0.99 * sv
            return (new_carry, total, disc), None

        (_, total_return, _), _ = jax.lax.scan(
            step_fn,
            (carry0, jnp.float32(0.0), jnp.float32(1.0)),
            action_seq)
        return total_return

    def cem_plan(params, dyn_carry, rng_key):
        # Cast carry to bfloat16 (model compute dtype)
        dyn_carry = jax.tree.map(lambda x: x.astype(jnp.bfloat16), dyn_carry)

        H = horizon
        N = n_candidates
        K = top_k
        A = n_actions

        logits = jnp.zeros((H, A), dtype=jnp.float32)  # uniform init

        def cem_iter(logits, iter_key):
            probs = jax.nn.softmax(logits, axis=-1)          # (H, A)
            step_keys = jax.random.split(iter_key, H)
            # Sample (N, H) sequences
            seqs = jax.vmap(
                lambda k, p: jax.random.choice(k, A, shape=(N,), p=p)
            )(step_keys, probs)                               # (H, N)
            seqs = seqs.T                                     # (N, H)

            # Evaluate in parallel over N
            returns = jax.vmap(
                lambda seq: _rollout_return(params, dyn_carry, seq)
            )(seqs)                                           # (N,)

            # Elite update
            elite_idx  = jnp.argsort(returns)[-K:]
            elite_seqs = seqs[elite_idx]                      # (K, H)
            counts = jax.vmap(
                lambda h: jnp.zeros(A).at[elite_seqs[:, h]].add(1.0)
            )(jnp.arange(H))                                  # (H, A)
            new_logits = jnp.log(counts / K + 1e-8)
            return new_logits, returns[elite_idx].mean()

        iter_keys = jax.random.split(rng_key, n_iter)
        logits, _ = jax.lax.scan(cem_iter, logits, iter_keys)

        return jnp.argmax(logits[0]).astype(jnp.int32)

    return jax.jit(cem_plan), params_np


# ------------------------------------------------------------------
# Reactive policy helper
# ------------------------------------------------------------------
def reactive_policy(agent, carry, obs_dict):
    """Run agent.policy and return (new_carry, action_int)."""
    carry, acts, _ = agent.policy(carry, obs_dict)
    return carry, int(acts["action"][0])


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------
def _make_obs(obs_np, reward, is_first):
    """Pack a single env observation into the batched format agent.policy expects."""
    return {
        "vector":      np.array([obs_np], dtype=np.float32),      # (1, 10)
        "reward":      np.array([reward],  dtype=np.float32),      # (1,)
        "is_first":    np.array([is_first], dtype=bool),           # (1,)
        "is_last":     np.array([False],    dtype=bool),           # (1,)
        "is_terminal": np.array([False],    dtype=bool),           # (1,)
    }


def run_episode(method: str, agent, cem_fn, cem_params, n_actions: int,
                seed: int, horizon: int = 0, n_candidates: int = 0):
    env = PrimitiveEnv(n_sheep=10, seed=seed)
    obs_np = env.reset()
    done = False
    total_reward = 0.0
    macro_steps = 0
    decision_times = []

    carry = agent.init_policy(1)
    if method.startswith("cem"):
        rng = jax.random.PRNGKey(seed)

    while not done:
        t0 = time.perf_counter()
        obs_dict = _make_obs(obs_np, total_reward, macro_steps == 0)

        if method == "reactive":
            carry, acts, _ = agent.policy(carry, obs_dict)
            action = int(acts["action"][0])

        elif method.startswith("cem"):
            # Step the world model forward to update dyn carry
            carry, acts, _ = agent.policy(carry, obs_dict)
            # carry[1] is the dyn_carry dict: {'deter': (deter,), 'stoch': (stoch, cls)}
            dyn_carry_raw = carry[1]
            dyn_carry_jax = {
                "deter": jnp.array(dyn_carry_raw["deter"]),  # (1, deter)
                "stoch": jnp.array(dyn_carry_raw["stoch"]),  # (1, stoch, cls)
            }
            rng, subkey = jax.random.split(rng)
            action = int(cem_fn(cem_params, dyn_carry_jax, subkey))

        elif method == "stromberg":
            sp = env._env.sheep_pos
            goal = env.goal
            spread = float(obs_np[4])
            frac = float(obs_np[7])
            if spread > 0.12:
                action = 0
            else:
                dists = np.linalg.norm(sp - goal, axis=1)
                if frac > 0.8:
                    action = 1
                else:
                    action = 2 + int(np.argmax(dists))

        dt = time.perf_counter() - t0
        decision_times.append(dt * 1000)  # ms

        obs_np, reward, done, info = env.step(action)
        total_reward += reward
        macro_steps += 1

    return {
        "success": info["fraction_in_goal"] >= 1.0,
        "fraction_in_goal": float(info["fraction_in_goal"]),
        "macro_steps": macro_steps,
        "total_reward": float(total_reward),
        "mean_decision_ms": float(np.mean(decision_times)),
        "seed": seed,
    }


# ------------------------------------------------------------------
# Main evaluation loop with resume
# ------------------------------------------------------------------
def evaluate_method(method: str, agent, cem_fn, cem_params, n_episodes: int,
                    n_actions: int, horizon: int = 0, n_candidates: int = 0):
    out_file = RESULTS_DIR / f"{method}.jsonl"

    # Load already-completed episodes
    done_seeds = set()
    completed = []
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_seeds.add(r["seed"])
                completed.append(r)
    print(f"[{method}] resuming: {len(completed)}/{n_episodes} already done")

    rng = np.random.default_rng(777)
    all_seeds = [int(rng.integers(0, 2**31)) for _ in range(n_episodes)]

    with open(out_file, "a") as f:
        for ep, seed in enumerate(all_seeds):
            if seed in done_seeds:
                continue
            result = run_episode(
                method, agent, cem_fn, cem_params, n_actions,
                seed=seed, horizon=horizon, n_candidates=n_candidates)
            completed.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

            if len(completed) % 10 == 0 or len(completed) == n_episodes:
                succ = np.mean([r["success"] for r in completed])
                steps = np.mean([r["macro_steps"] for r in completed])
                dt = np.mean([r["mean_decision_ms"] for r in completed])
                print(f"  [{method}] ep {len(completed)}/{n_episodes} "
                      f"success={succ:.3f} steps={steps:.1f} dt={dt:.1f}ms")

    return completed


# ------------------------------------------------------------------
# Summary + plot
# ------------------------------------------------------------------
def summarize(results):
    success = [r["success"] for r in results]
    steps = [r["macro_steps"] for r in results]
    rewards = [r["total_reward"] for r in results]
    dt = [r["mean_decision_ms"] for r in results]
    return {
        "success_rate": float(np.mean(success)),
        "mean_macro_steps": float(np.mean(steps)),
        "std_macro_steps": float(np.std(steps)),
        "mean_reward": float(np.mean(rewards)),
        "mean_decision_ms": float(np.mean(dt)),
        "n_episodes": len(results),
    }


def plot_results(all_results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    methods = list(all_results.keys())
    success_rates = [all_results[m]["success_rate"] for m in methods]
    mean_steps = [all_results[m]["mean_macro_steps"] for m in methods]
    std_steps = [all_results[m].get("std_macro_steps", 0) for m in methods]
    dec_times = [all_results[m]["mean_decision_ms"] for m in methods]

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    bars = ax.bar(x, success_rates, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylim(0, 1.1); ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate")
    for bar, v in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
                ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, mean_steps, yerr=std_steps, color=colors, alpha=0.85,
                  capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel("Mean Macro-Steps"); ax.set_title("Planning Efficiency (lower=better)")
    for bar, v in zip(bars, mean_steps):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}",
                ha='center', va='bottom', fontsize=8)

    ax = axes[2]
    bars = ax.bar(x, dec_times, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel("Mean Decision Time (ms)"); ax.set_title("Computation Cost")
    for bar, v in zip(bars, dec_times):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}ms",
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "cem_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved: {out}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--methods", type=str,
                        default="reactive,cem_H5_N256,cem_H8_N512,cem_H12_N512,stromberg")
    args = parser.parse_args()

    methods = args.methods.split(",")
    n_episodes = args.n_episodes

    # Method configs
    cem_configs = {
        "cem_H5_N256":  dict(horizon=5,  n_candidates=256, n_iter=3, top_k=32),
        "cem_H8_N512":  dict(horizon=8,  n_candidates=512, n_iter=3, top_k=64),
        "cem_H12_N512": dict(horizon=12, n_candidates=512, n_iter=3, top_k=64),
    }

    print("Building agent and loading checkpoint...")
    agent, config = build_agent()
    n_actions = 12  # N_sheep + 2

    # Pre-build CEM functions for all needed configs
    cem_fns = {}
    cem_params_map = {}
    for m in methods:
        if m.startswith("cem"):
            cfg = cem_configs[m]
            print(f"Compiling CEM for {m} (H={cfg['horizon']}, N={cfg['n_candidates']})...")
            fn, params = make_cem_fn(
                agent,
                horizon=cfg["horizon"],
                n_candidates=cfg["n_candidates"],
                n_iter=cfg["n_iter"],
                top_k=cfg["top_k"],
                n_actions=n_actions,
            )
            # Warm-up JIT
            dummy_carry = {
                "deter": jnp.zeros((1, agent.model.dyn.deter)),
                "stoch": jnp.zeros((1, agent.model.dyn.stoch, agent.model.dyn.classes)),
            }
            _ = fn(params, dummy_carry, jax.random.PRNGKey(0))
            print(f"  {m} compiled.")
            cem_fns[m] = fn
            cem_params_map[m] = params

    # Run each method
    all_summaries = {}
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running: {method}  ({n_episodes} episodes)")
        print(f"{'='*50}")
        cfg = cem_configs.get(method, {})
        results = evaluate_method(
            method=method,
            agent=agent,
            cem_fn=cem_fns.get(method),
            cem_params=cem_params_map.get(method),
            n_episodes=n_episodes,
            n_actions=n_actions,
            horizon=cfg.get("horizon", 0),
            n_candidates=cfg.get("n_candidates", 0),
        )
        s = summarize(results)
        all_summaries[method] = s
        print(f"  SUMMARY: {json.dumps(s, indent=2)}")

    # Save final JSON
    summary_file = RESULTS_DIR / "cem_results.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved: {summary_file}")

    # Plot
    plot_results(all_summaries)


if __name__ == "__main__":
    main()
