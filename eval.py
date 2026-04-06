"""
Evaluate trained DreamerV3 agent + plot learning curves vs baselines.

Usage:
  python eval.py --logdir results/hierarchical_dreamer --n_episodes 100
"""

import sys
import pathlib
import json
import argparse
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(1, str(pathlib.Path(__file__).parent.parent / "minecraft" / "dreamerv3"))

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


# ------------------------------------------------------------------
# Parse metrics.jsonl from dreamerv3 log
# ------------------------------------------------------------------
def load_metrics(logdir: pathlib.Path):
    metrics_file = logdir / "metrics.jsonl"
    if not metrics_file.exists():
        raise FileNotFoundError(metrics_file)
    records = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_learning_curve(records, key="episode/score"):
    steps, values = [], []
    for r in records:
        if key in r:
            steps.append(r.get("step", len(steps)))
            values.append(r[key])
    return np.array(steps), np.array(values)


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
def plot_results(logdir: pathlib.Path, out_dir: pathlib.Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    records = load_metrics(logdir)
    steps, scores = extract_learning_curve(records, "episode/score")

    # Load baseline jsons
    stromberg_data = {}
    random_data = {}
    sb_path = out_dir / "baseline_stromberg.json"
    rb_path = out_dir / "baseline_random.json"
    if sb_path.exists():
        stromberg_data = json.loads(sb_path.read_text())["summary"]
    if rb_path.exists():
        random_data = json.loads(rb_path.read_text())["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Learning curve
    ax = axes[0]
    if len(steps) > 0:
        window = max(1, len(scores) // 20)
        smooth = np.convolve(scores, np.ones(window) / window, mode="valid")
        ax.plot(steps[:len(smooth)], smooth, label="Hierarchical DreamerV3", color="steelblue")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Score")
    ax.set_title("Learning Curve")
    ax.legend()

    # Comparison bar chart
    ax = axes[1]
    labels, vals = [], []
    if stromberg_data:
        labels.append("Strömbom\nHeuristic")
        vals.append(stromberg_data["success_rate"])
    if random_data:
        labels.append("Random\nPrimitives")
        vals.append(random_data["success_rate"])

    # DreamerV3 final success rate (score >= 1.0 as proxy)
    if len(scores) > 0:
        window = min(50, len(scores))
        final_rate = float(np.mean(scores[-window:] >= 1.0))
        labels.append("Hierarchical\nDreamerV3")
        vals.append(final_rate)

    if labels:
        colors = ["forestgreen", "tomato", "steelblue"][:len(labels)]
        ax.bar(labels, vals, color=colors, alpha=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Success Rate")
        ax.set_title("Baseline Comparison")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    out_file = out_dir / "results_plot.png"
    plt.savefig(out_file, dpi=150)
    print(f"Plot saved to {out_file}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="results/hierarchical_dreamer")
    parser.add_argument("--n_episodes", type=int, default=100)
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    if not logdir.is_absolute():
        logdir = pathlib.Path(__file__).parent / logdir

    print(f"Loading metrics from {logdir}")
    try:
        records = load_metrics(logdir)
        print(f"Loaded {len(records)} metric records")
        steps, scores = extract_learning_curve(records, "episode/score")
        if len(scores) > 0:
            print(f"Steps range: {steps[0]} – {steps[-1]}")
            print(f"Score: mean={scores.mean():.3f} last50={scores[-50:].mean():.3f}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    plot_results(logdir, RESULTS_DIR)


if __name__ == "__main__":
    main()
