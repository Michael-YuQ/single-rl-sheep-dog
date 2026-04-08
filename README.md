# Shepherd-Dreamer: Multi-Dog DreamerV3 + CEM Herding

World-model-based sheep herding with **DreamerV3 RSSM + CEM planning**. Supports single-dog and multi-dog scenarios with extended action spaces, perimeter-only arc movement, multi-milestone rewards, and 3-dog joint-action coordination.

---

## What's New (Extended Version)

### 1. Extended Arc Action Space (22 actions → was 12)
Original 12 macro-actions + 10 new arc/bypass actions:

| ID | Name | Description |
|----|------|-------------|
| 0 | MUSTER | Direct approach behind outlier sheep |
| 1 | HOLD | Guard goal entrance |
| 2–11 | DRIVE_rank{k} | Push k-th farthest sheep (direct) |
| 12 | MUSTER_ARC | Arc around flock to outlier sheep |
| 13 | FLANK_LEFT | Arc to left flank of flock |
| 14 | FLANK_RIGHT | Arc to right flank of flock |
| 15 | SWEEP_BEHIND | Arc to directly behind flock |
| 16–21 | BYPASS_DRIVE{k} | Arc + drive k-th farthest sheep |

### 2. Perimeter-Only Constraint (`envs/perimeter_env.py`)
All 22 actions forced to arc around the flock — dog never cuts through sheep group. Physically realistic (real sheepdogs stay on the outside).

### 3. Push-Track Environment for N=100 (`envs/push_env.py`)
- Tight arc parameters (ARC_OFFSET=1.0 vs 5.0) — hugs flock perimeter
- `PUSH_TRACK` (action 0): continuously tracks flock centroid behind position every step
- `COLLECT_STRAGGLER` (action 22): targets most isolated outside-goal sheep
- **Multi-milestone reward**: superlinear bonuses at 50/75/90/95/100% to eliminate long-tail problem
- Designed for direct N=100 training (not zero-shot)

### 4. 3-Dog Joint Action RSSM (`envs/three_dog_env.py`)
- 3 dogs, each with 5 low-level primitives: PUSH_BEHIND / FLANK_LEFT / FLANK_RIGHT / HOLD / COLLECT_STRAGGLER
- Joint action space: 5³ = **125 combinations** (cross-product coordination)
- K=5 steps per joint action (more responsive than K=15 macro-actions)
- 14-dim observation includes all 3 dog positions
- RSSM learns WHEN to coordinate (e.g., dog0=PUSH + dog1=FLANK_L + dog2=HOLD)

### 5. PPO and A2C Baselines (`train_ppo_ac.py`)
- Discrete macro-action PPO and A2C using stable-baselines3
- Same 22-action space, same 10-dim obs — fair comparison with DreamerV3
- Both converge to near-100% success on N=10 in 500k steps

### 6. 4-Dog Arc Formation Rule-Based Controller (`arc_formation_4dogs.py`)
- Geometric arc formation: 4 dogs evenly spaced on a semicircle behind flock
- Straggler collection: nearest dog breaks formation to collect isolated sheep
- 60% success rate on N=100 (20 seeds), 97.5% mean completion

---

## Architecture

```
envs/
  sheep_env.py              — Strömbom dynamics, n_dogs support
  generalized_primitive_env.py  — 22 discrete macro-actions (arc + direct)
  primitive_env.py          — N+12 actions (original interface)
  perimeter_env.py          — All-arc perimeter-only constraint
  push_env.py               — N=100 tight-arc + PUSH_TRACK + milestone reward (23 actions)
  three_dog_env.py          — 3-dog joint low-level actions (125 joint actions)
  continuous_env.py         — Continuous 2D velocity (for PPO/A2C)
  discrete_gym_env.py       — Gymnasium wrapper for PPO/A2C on macro-actions
  shepherd_dreamer.py       — DreamerV3 adapters for all envs

configs/
  shepherd.yaml             — Single-dog N=10 (22 actions)
  shepherd_perimeter.yaml   — Perimeter-only N=10
  shepherd_push100.yaml     — Push-track N=100 (22 actions)
  shepherd_milestone.yaml   — Push-track N=100 + milestone reward (23 actions)
  shepherd_threedog.yaml    — 3-dog joint actions N=100 (125 actions)
  shepherd_multidog.yaml    — Multi-dog config

train.py                    — DreamerV3 training entry point (all suites)
train_ppo_ac.py             — PPO + A2C training with stable-baselines3
cem_planner.py              — CEM planning in RSSM latent space
arc_formation_4dogs.py      — Rule-based 4-dog arc formation

record_video_arc.py         — 5-method comparison (stromberg/reactive/cem/ppo/ac)
record_perimeter_compare.py — Perimeter vs original CEM comparison
record_push_compare.py      — Push-baseline vs push-milestone comparison
record_milestone_compare.py — Milestone reward long-tail fix comparison
record_threedog.py          — 3-dog joint action RSSM recording
```

---

## Observation Spaces

| Environment | Obs Dim | Contents |
|-------------|---------|----------|
| Single-dog (all variants) | 10 | dog_xy, centroid_xy, spread, farthest_xy, frac, goal_xy |
| Three-dog | 14 | dog0_xy, dog1_xy, dog2_xy, centroid_xy, spread, farthest_xy, frac, goal_xy |

---

## Results Summary

### N=100 Zero-Shot / Direct Training (CEM)

| Method | N_train | Steps | Success (N=100) | Avg Steps |
|--------|---------|-------|-----------------|-----------|
| Strömbom heuristic | — | — | ~76% | 600 (timeout) |
| DreamerV3 reactive (N=10 zero-shot) | 10 | 500k | 100% | 582 |
| DreamerV3 CEM (N=10 zero-shot) | 10 | 500k | **100%** | **148** |
| Perimeter CEM (N=10 zero-shot) | 10 | 500k | 92% | 600 (timeout) |
| Push-track CEM (N=100 direct) | 100 | 200k | **100%** | **192** |
| Push-milestone CEM (N=100) | 100 | 500k | 50% (CEM, improving) | 428 |
| 4-Dog Arc Formation (rule) | — | — | 60% | 787 |
| 3-Dog Joint RSSM (N=100) | 100 | 100k | ~92% reactive | training... |

### Milestone Reward (Long-Tail Fix)
| Method | Success (20 seeds, CEM) | Mean Frac |
|--------|------------------------|-----------|
| Push baseline | 10% | 83.4% |
| Push milestone (500k) | **50%** | **93.0%** |

---

## Training

```bash
# Single-dog DreamerV3 (22 actions, N=10)
python train.py --configs shepherd --logdir results/dreamer_arc

# Perimeter-only (all-arc, N=10)
python train.py --configs shepherd_perimeter --logdir results/dreamer_perimeter

# Push-track direct N=100
python train.py --configs shepherd_push100 --logdir results/dreamer_push100

# Milestone reward N=100
python train.py --configs shepherd_milestone --logdir results/dreamer_milestone

# 3-dog joint actions N=100
python train.py --configs shepherd_threedog --logdir results/dreamer_threedog

# PPO + A2C (discrete macro-actions)
python train_ppo_ac.py --total_steps 500000 --n_envs 8 --algo both
```

---

## Recording Videos

```bash
# 5-method comparison (stromberg/reactive/cem/ppo/ac), N=10/50/100
python record_video_arc.py --n_sheep 100 --seed 7 --max_steps 600

# Perimeter vs original CEM
python record_perimeter_compare.py --n_sheep 100 --seed 7

# Push-baseline vs push-milestone (long-tail fix)
python record_milestone_compare.py --n_sheep 100 --seed 7

# 3-dog joint RSSM
python record_threedog.py --n_sheep 100 --seed -1   # all 10 seeds

# 4-dog arc formation (rule-based)
python arc_formation_4dogs.py --n_sheep 100 --seed 0
python arc_formation_4dogs.py --eval --n_seeds 20
```

---

## SOTA Context

| Method | Success | Year | Notes |
|--------|---------|------|-------|
| Hierarchical Learning (RL) | 100% | 2024 | [arXiv:2508.02632](https://arxiv.org/abs/2508.02632) |
| Hierarchical PPO (decentralized) | 99.7% | 2025 | [arXiv:2504.02479](https://arxiv.org/abs/2504.02479) |
| This project CEM (N=100 direct) | 100% | 2026 | Push-track, 192 steps |
| Strömbom heuristic | ~76% | 2014 | Baseline |
