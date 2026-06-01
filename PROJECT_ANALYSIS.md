# RL Group Project — Analysis & Completion Guide
**Clinical Treatment Optimisation: Sepsis ICU Management**
Group: Fausto Gomez, Miguel Marques, Tomás Gonçalves, Tomás Vitorino

---

## 1. Project Overview

This is a Reinforcement Learning course project for the Master in Data Science & Advanced Analytics. The goal is to train RL agents to optimise treatment decisions (vasopressor and IV fluid dosage) for ICU sepsis patients, using the `ICU-Sepsis-v2` environment built from real MIMIC-III patient data.

The project is divided into three parts:

| Part | Description | Difficulty |
|------|-------------|------------|
| **Config A** | Tabular RL on discrete MDP (716 states, 25 actions) | Foundation |
| **Config B** | Deep RL on continuous 47-dim observations + 3 clinical wrappers | Main challenge |
| **Creative Extension** | Original analysis demonstrating clinical insight | Differentiator |

**Environment details:**
- States: 716 (714 clinical + 2 terminal: survived=714, died=713)
- Actions: 25 (5 vasopressor levels × 5 IV fluid dose levels)
- Reward: +1 at survival, 0 at death, –lam×intensity at each step (lam=0.02)
- Discount: γ=1.0 (no time discounting — paper convention)
- Hard config: `sofa_bias=5.0` shifts initial state toward sicker patients

**Config B adds three clinical reality wrappers:**
- `EpisodicNoisyObsEnv`: 15% chance entire episode has Gaussian noise (std=0.10) on observations (monitor malfunction)
- `EpisodicMissingObsEnv`: 15% chance 4 features are zeroed for entire episode (missing labs)
- `AcuteEventEnv`: 1% per-step chance of sudden patient death regardless of treatment (cardiac arrest)

---

## 2. What Has Already Been Implemented

### `envs/` package (complete, production-quality)

| File | Status | Contents |
|------|--------|----------|
| `env_setup.py` | Complete | Constants (ENV_ID, N_STATES, N_ACTIONS, GAMMA, INTENSITY, SOFA_BIAS, LAM), `make_sepsis_env()` factory |
| `continuous_sepsis_env.py` | Complete | `ContinuousICUSepsisEnv` — wraps discrete ICU-Sepsis-v2 with 47-dim continuous observations |
| `wrappers.py` | Complete | `EpisodicNoisyObsEnv`, `EpisodicMissingObsEnv`, `AcuteEventEnv`, `make_clinical_env()` |

### `new_notebook.ipynb` — current group submission

#### Config A (Tabular RL)
- [x] Environment exploration with 3-panel visualisation (state visitation, reward sparsity, transition entropy)
- [x] Random baseline (1000 episodes)
- [x] **Policy Iteration** — model-based, exact convergence, evaluates every iteration
- [x] **Q-Learning** — off-policy TD, 30k episodes, ε-greedy with exponential decay (1.0 → 0.01)
- [x] Full results visualisation: convergence curves, survival rate curves, ε-decay, Q-table heatmaps, policy agreement bar chart
- [x] Comparative results table (Random vs PI vs Q-Learning)

#### Config B (Deep RL)
- [x] Clinical environment setup + wrapper verification
- [x] Random baseline for Config B (1000 episodes)
- [x] `evaluate_deep()` evaluation function (500 episodes, greedy)
- [x] `ReplayBuffer` (deque-based, capacity 50k)
- [x] **DQN with Double DQN updates** — 3000 episodes, experience replay, target network (sync every 200 steps), cosine LR scheduler, gradient clipping (norm 10)
- [x] **Dueling DQN** — identical training setup, value+advantage streams, logs stream magnitudes
- [x] Full results visualisation: training curves, eval convergence, TD loss, dueling stream magnitudes, results table

#### Creative Extension
- [x] `collect_action_distribution()` — action frequency + per-action survival rate over 1000 episodes
- [x] Action frequency heatmaps (5×5 vasopressor×fluid grid) for random, DQN, Dueling DQN
- [x] Survival rate per action heatmaps (DQN and Dueling DQN)

#### Generated plots (all present in `plots/`)
All plots from both Config A and Config B are present, plus the creative extension heatmaps and a robustness-by-wrapper analysis from an earlier notebook version.

---

## 3. Issues Detected

### BLOCKING — Project will not run
| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **Python 3.9 incompatible with `icu-sepsis`** — the package requires Python ≥ 3.10 | Environment | Nothing runs |
| 2 | **`stable-baselines3` not installed** | Environment | Config B fails |

### Code bugs / incomplete cells
| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 3 | **Cell 25 is truncated** — `train_dueling_dqn()` training call ends at `print('Training Duel` mid-string; the call `train_dueling_dqn()` and result unpacking are missing | `new_notebook.ipynb` Cell 25 | Dueling DQN never trains, all downstream cells fail with NameError on `duel_train_ret`, `duel_eval`, `duel_losses`, `duel_mean`, `duel_surv`, `duel_len`, `duel_policy` |
| 4 | **Cell 16 truncated** — policy agreement plot ends at `plt.s` without `plt.savefig()`/`plt.show()` | `new_notebook.ipynb` Cell 16 | Plot not saved, cell raises SyntaxError |
| 5 | **`build_P(env)` ignores its argument** — uses outer-scope `P` and `R_sasp` matrices directly instead of extracting from the passed env | `new_notebook.ipynb` Cell 9 | Works by coincidence (same env), but creates fragile coupling between cells |

### Design / correctness concerns
| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 6 | **Creative extension claims "all four trained agents"** but only evaluates DQN and Dueling DQN — tabular Q-tables cannot be applied to the 47-dim continuous obs from `make_clinical_env()` without a discrete state lookup | `new_notebook.ipynb` Cell 31–32 markdown vs code | The markdown overstates coverage; PI and Q-Learning distributions are never collected |
| 7 | **No model persistence** — no `torch.save()` for DQN or Dueling DQN weights | Config B | Full retraining required on every kernel restart |
| 8 | **MSE loss for DQN** — standard practice uses Huber loss (SmoothL1), which is more robust to reward outliers in sparse environments | Cell 23 & 25 | Minor performance risk |
| 9 | **3000 training episodes is very low for Config B** — with 15% wrapper noise and missing obs, convergence may be unstable at this scale | Cells 23 & 25 | Agents may not reach true optimal policy; results are sensitive to seed |
| 10 | **`evaluate_deep()` calls `make_clinical_env()` each time** — creates fresh env with default params; fine but if you ever want to evaluate with custom wrapper params you need a different eval function | Cell 21 | Low impact but inflexible |

---

## 4. Improvement Strategies

### Immediate fixes (must do to get the project running)
1. **Upgrade Python** — create a conda env with Python 3.11:
   ```
   conda create -n rl_project python=3.11
   conda activate rl_project
   pip install -r requirements.txt
   ```
2. **Fix Cell 25** — complete the Dueling DQN training call (see Section 5).
3. **Fix Cell 16** — add `plt.savefig(...)` and `plt.show()` to the policy agreement plot.

### Performance improvements
4. **Increase training episodes** — 3000 is marked as "5 MINUTE RUN"; for a final submission increase DQN/Dueling DQN to 6000–10000 episodes for more stable convergence.
5. **Switch to Huber loss** — replace `F.mse_loss` with `F.smooth_l1_loss` in both DQN and Dueling DQN. More robust on sparse rewards.
6. **Save trained models** — add `torch.save(q_net.state_dict(), 'models/dqn.pt')` after training so results are reproducible without retraining.

### Grade-maximising additions
7. **Fix the creative extension to include tabular agents** — run PI and Q-Learning on Config A env (`make_sepsis_env()`) and collect action distributions there. Then compare Config A vs Config B action patterns side-by-side. This makes the "all four agents" claim true and adds genuine cross-config insight.
8. **Add per-wrapper robustness analysis** — the old notebook has this and the plot (`ext_robustness_by_wrapper.png`) already exists in `plots/`. Stratify DQN/Dueling results by which wrapper was active (clean/noisy/missing/acute). This is clinically meaningful and directly addresses what the teacher called "three clinical reality wrappers".
9. **Add treatment intensity analysis** — plot mean treatment intensity (vasopressor + fluid level) vs. survival rate. Shows whether the agent learned parsimonious treatment (the `lam=0.02` penalty was designed to incentivise this).

---

## 5. What You Need to Do to Finish with Maximum Grade

The grade breakdown is not explicitly stated in the project files. Based on the project structure (2 required configs + extension) and the emphasis the template places on each section, the inferred weighting is:

| Component | Inferred Weight | Status |
|-----------|----------------|--------|
| Config A — 2 algorithms correctly implemented | ~30% | Implemented but blocked by Python version |
| Config B — 2 deep RL algorithms on clinical env | ~40% | Implemented but Cell 25 truncated |
| Creative Extension — originality + clinical insight | ~20% | Partial (only Config B, no tabular comparison) |
| Notebook quality (explanations, plots, results tables) | ~10% | Good — well-documented with markdown |

### Step-by-step checklist to maximum grade

#### Environment (do first)
- [ ] Create Python 3.11 conda environment and install all requirements
- [ ] Verify `icu-sepsis` installs and `python envs/continuous_sepsis_env.py` runs clean

#### Fix broken notebook cells
- [ ] Complete Cell 25: add the `train_dueling_dqn()` call, unpack results, define `duel_policy`
- [ ] Fix Cell 16: add `plt.savefig(f'{PLOTS_DIR}/A_policy_agreement.png', bbox_inches='tight')` and `plt.show()`
- [ ] Run the full notebook top-to-bottom and confirm zero errors

#### Strengthen Config A (30% of grade)
- [ ] Verify Policy Iteration reaches ≥85% survival rate
- [ ] Verify Q-Learning clearly beats the random baseline (~78% survival)
- [ ] Make sure the hyperparameter justification in the markdown cells is complete and specific

#### Strengthen Config B (40% of grade)
- [ ] Increase training episodes to at least 6000 for both DQN and Dueling DQN
- [ ] Switch to Huber loss (`F.smooth_l1_loss`) for more stable training
- [ ] Save model weights after training
- [ ] Ensure Dueling DQN outperforms standard DQN (or explain why it doesn't)
- [ ] Verify both algorithms beat the Config B random baseline

#### Strengthen Creative Extension (20% of grade)
- [ ] Add robustness breakdown: evaluate each agent under clean / noisy-only / missing-only / both / acute conditions (the old notebook's Extension 1 code and plot exist as reference)
- [ ] Add treatment intensity analysis: plot intensity distribution per agent, compare to random baseline — this directly tests whether `lam=0.02` achieved parsimonious treatment
- [ ] Fix the claim "all four agents" in the action distribution analysis: either run tabular agents on Config A env and compare, or correct the markdown to say "both deep RL agents"
- [ ] Add a 2–3 sentence clinical interpretation per plot explaining what the result means for a real ICU deployment

#### Notebook presentation (10% of grade)
- [ ] Confirm all plots are saved to `plots/` with consistent naming
- [ ] Confirm the final results summary cell runs and prints clean tables
- [ ] Remove any dead/broken code or placeholder cells before submission
- [ ] Fill in the group name header (currently says "Group N")

---

## 6. Quick Reference: Key Numbers

| Metric | Expected value | Notes |
|--------|---------------|-------|
| Random baseline Config A | ~78% survival | Benchmark to beat |
| Random baseline Config B | ~74–77% survival | Lower due to wrappers |
| Policy Iteration | ~87–90% survival | Model-based optimal |
| Q-Learning | ~83–87% survival | Model-free, near-optimal |
| DQN (3000 eps) | ~78–83% survival | May underfit at 3k |
| Dueling DQN (3000 eps) | ~79–84% survival | Should match or beat DQN |
| DQN (6000+ eps) | ~83–87% survival | Target with more training |
