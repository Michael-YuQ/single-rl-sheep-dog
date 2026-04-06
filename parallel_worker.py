"""
Standalone worker for multiprocessing pool.
Handles heuristic methods (no GPU/JAX) AND reactive/cem via a
subprocess-local agent loaded from checkpoint.
Each worker is a separate process → full CPU parallelism, no GIL.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(1, "/root/minecraft/dreamerv3")

from envs.generalized_primitive_env import GeneralizedPrimitiveEnv
import numpy as np

K = 15  # primitives per macro-step

# Module-level agent cache so each worker process loads it only once
_agent_cache = {}
_cem_cache   = {}


def _get_agent():
    if "agent" not in _agent_cache:
        import warnings; warnings.filterwarnings("ignore")
        from cem_planner import build_agent
        agent, _ = build_agent()
        _agent_cache["agent"] = agent
    return _agent_cache["agent"]


def _get_cem(agent):
    if "cem" not in _cem_cache:
        import jax.numpy as jnp
        from cem_planner import make_cem_fn
        fn, params = make_cem_fn(
            agent, horizon=8, n_candidates=512, n_iter=3, top_k=64, n_actions=12)
        import jax
        fn(params, {"deter": jnp.zeros((1,512)), "stoch": jnp.zeros((1,32,4))},
           jax.random.PRNGKey(0))
        _cem_cache["cem"] = (fn, params)
    return _cem_cache["cem"]


def run_episode_worker(args):
    """Universal worker: handles all 4 methods."""
    method, n_sheep, seed = args
    import time

    env = GeneralizedPrimitiveEnv(n_sheep=n_sheep, seed=int(seed))
    obs = env.reset()
    done = False
    total_r = 0.0
    steps = 0
    dec_times = []

    if method in ("reactive", "cem"):
        import jax, jax.numpy as jnp
        from cem_planner import _make_obs
        agent = _get_agent()
        carry = agent.init_policy(1)
        rng   = jax.random.PRNGKey(int(seed)) if method == "cem" else None
        if method == "cem":
            cem_fn, cem_params = _get_cem(agent)

    while not done:
        t0 = time.perf_counter()

        if method == "reactive":
            from cem_planner import _make_obs
            od = _make_obs(obs, total_r, steps == 0)
            carry, acts, _ = agent.policy(carry, od)
            action = int(acts["action"][0])

        elif method == "cem":
            from cem_planner import _make_obs
            import jax.numpy as jnp
            od = _make_obs(obs, total_r, steps == 0)
            carry, acts, _ = agent.policy(carry, od)
            dc = {"deter": jnp.array(carry[1]["deter"]),
                  "stoch": jnp.array(carry[1]["stoch"])}
            rng, sk = jax.random.split(rng)
            action = int(cem_fn(cem_params, dc, sk))

        elif method == "stromberg":
            spread = float(obs[4]); frac = float(obs[7])
            if spread > 0.12:  action = 0
            elif frac > 0.8:   action = 1
            else:              action = 2

        else:  # random
            action = int(np.random.randint(0, 12))

        dec_times.append((time.perf_counter() - t0) * 1000)
        obs, r, done, info = env.step(action)
        total_r += r
        steps += 1

    return {
        "success":          bool(info["fraction_in_goal"] >= 1.0),
        "fraction_in_goal": float(info["fraction_in_goal"]),
        "macro_steps":      steps,
        "low_level_steps":  steps * K,
        "total_reward":     float(total_r),
        "mean_decision_ms": float(np.mean(dec_times)) if dec_times else 0.0,
        "n_sheep":          n_sheep,
        "method":           method,
        "seed":             int(seed),
    }


def run_heuristic(args):
    """Kept for backward compat — delegates to run_episode_worker."""
    return run_episode_worker(args)
