# Shepherd-Dreamer: Single-Dog DreamerV3 + CEM Herding

World-model-based herding using **DreamerV3 RSSM** trained on a 10-dim aggregated observation, then planning at test time with **CEM (Cross-Entropy Method)** in latent space. Trained on N=10 sheep, zero-shot generalises to N=100.

## Architecture

```
Observation (10-dim, N-invariant):
  [dog_x, dog_y, centroid_x, centroid_y, spread,
   farthest_x, farthest_y, fraction_in_goal, goal_x, goal_y]

Macro-actions (12 discrete):
  0: MUSTER  — go behind outlier sheep
  1: HOLD    — guard goal entrance
  2-11: DRIVE rank-k — push k-th farthest sheep toward goal

Each macro-action = 15 low-level geometric control steps
```

## Results

| N | CEM Success | Reactive Success |
|---|------------|-----------------|
| 10 | 100% | 100% |
| 30 | 99% | 97% |
| 50 | 93% | 87% |
| 100 | **77%** | 58% |

## Installation

```bash
conda create -n dreamer python=3.10 -y
conda activate dreamer

# Download DreamerV3 source
curl -L https://github.com/danijar/dreamerv3/archive/refs/heads/main.zip -o dreamerv3.zip
unzip dreamerv3.zip -d /path/to/dreamerv3_root

pip install "jax[cuda12]==0.4.33" ninjax elements portal "ruamel.yaml" pillow numpy gymnasium gym==0.26.2 optax
```

## Training

```bash
# Train single-dog world model (N=10, ~2.5h on RTX 4090)
python train.py --configs shepherd
# Checkpoint: results/hierarchical_dreamer/ckpt/
```

`configs/shepherd.yaml`:
```yaml
shepherd:
  task: shepherd_default
  run:
    steps: 500000
  rssm: {deter: 512, hidden: 64, classes: 4}
```

## Evaluation

```bash
# CEM vs Reactive vs Random (N=10, 200 episodes)
python cem_planner.py --n_episodes 200

# Zero-shot scale generalisation (N=10 to 100)
python scale_eval.py --n_list 10,20,30,50,100

# Record comparison video (N=100)
python record_video_N100.py
```

## CEM Planning

CEM operates entirely in the RSSM latent space — no environment interaction at planning time:

```
logits: (H=8, A=12) initialized uniform
for iter in range(3):
    sample N=512 action sequences of length H
    roll out each in latent space → compute discounted return
    elite = top-64 by return
    update logits from elite counts
execute argmax(logits[0])
```

## Dependencies

- [DreamerV3](https://github.com/danijar/dreamerv3) (GitHub main branch)
- JAX 0.4.33 + CUDA 12
- `elements`, `portal`, `ninjax`, `embodied`

## Reference

- Hafner et al. (2023). *Mastering Diverse Domains through World Models (DreamerV3)*. arXiv:2301.04104
- Strömbom et al. (2014). *Solving the shepherding problem*. J. R. Soc. Interface.
