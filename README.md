# Sepsis ICU Treatment Optimisation — RL Group Project

**Group:** Fausto Gomez, Miguel Marques, Tomás Gonçalves, Tomás Vitorino  
**Course:** Reinforcement Learning — Master in Data Science & Advanced Analytics

---

## Overview

This project trains Reinforcement Learning agents to optimise treatment decisions for ICU sepsis patients. Agents control two continuous clinical actions — vasopressor dosage and IV fluid levels — across 25 discrete action combinations, with the goal of maximising patient survival while penalising unnecessary treatment intensity.

The environment is built on real MIMIC-III patient data via the [`icu-sepsis`](https://github.com/CLAIRlab/icu-sepsis) package.

---

## Project Structure

```
sepsis_project - Students Version/
├── group_n.ipynb               # Main notebook — all experiments and results
├── requirements.txt            # Python dependencies
├── envs/
│   ├── env_setup.py            # Constants + make_sepsis_env() factory (Config A)
│   ├── continuous_sepsis_env.py# 47-dim continuous observation wrapper (Config B)
│   ├── wrappers.py             # Clinical reality wrappers + make_clinical_env()
│   └── __init__.py
├── plots/                      # All generated figures (saved automatically)
└── models/                     # Saved model weights (created after training)
```

---

## Setup

> **Python 3.10 or higher is required.** The `icu-sepsis` package does not support Python 3.9 or below.

### 1. Create a conda environment

```bash
conda create -n rl_project python=3.11
conda activate rl_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook group_n.ipynb
```

---

## Environment Details

### Config A — Tabular MDP

| Parameter | Value |
|-----------|-------|
| Environment | `Sepsis/ICU-Sepsis-v2` |
| States | 716 (714 clinical + 2 terminal) |
| Actions | 25 (5 vasopressor × 5 IV fluid levels) |
| Reward | +1 survival, 0 death, −lam×intensity per step |
| Discount γ | 1.0 (no time discounting) |
| `sofa_bias` | 5.0 (shifts initial state toward sicker patients) |
| `lam` | 0.02 (treatment parsimony penalty) |

### Config B — Continuous Observations + Clinical Wrappers

Built on top of Config A, `make_clinical_env()` adds:

| Wrapper | Effect |
|---------|--------|
| `ContinuousICUSepsisEnv` | Maps discrete states to 47-dim continuous observations |
| `EpisodicNoisyObsEnv` | 15% chance per episode of Gaussian noise (std=0.10) on all observations — models monitor malfunction |
| `EpisodicMissingObsEnv` | 15% chance per episode of 4 features zeroed — models missing lab values |
| `AcuteEventEnv` | 1% per-step chance of sudden patient death regardless of treatment — models cardiac arrest |

---

## Algorithms Implemented

### Config A (Tabular RL)
- **Policy Iteration** — model-based, exact convergence
- **Q-Learning** — off-policy TD, ε-greedy with exponential decay

### Config B (Deep RL)
- **DQN with Double DQN updates** — experience replay, target network, cosine LR schedule
- **Dueling DQN** — value + advantage decomposition streams

---

## Expected Results

| Agent | Survival Rate |
|-------|--------------|
| Random baseline (Config A) | ~78% |
| Random baseline (Config B) | ~74–77% |
| Policy Iteration | ~87–90% |
| Q-Learning | ~83–87% |
| DQN | ~83–87% |
| Dueling DQN | ~83–87% |

---

## Quick Usage — Environments

```python
from envs.env_setup import make_sepsis_env

# Config A: tabular environment
env = make_sepsis_env()
obs, info = env.reset()
```

```python
from envs.wrappers import make_clinical_env

# Config B: continuous + clinical wrappers
env = make_clinical_env()
obs, info = env.reset()  # obs.shape == (47,)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data handling |
| `matplotlib`, `seaborn` | Plotting |
| `scipy` | Scientific utilities |
| `torch` | Deep RL (DQN, Dueling DQN) |
| `gymnasium` | RL environment interface |
| `icu-sepsis>=2.0.0` | ICU-Sepsis-v2 environment (requires Python ≥ 3.10) |
| `stable-baselines3` | RL utilities / baselines |
