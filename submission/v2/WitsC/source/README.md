# Source code

Project code for the WitsC DQN and PPO agents. Everything here was written for this project on top of the official ARC-AGI-3 framework.

## Layout

| Path | Contents |
|---|---|
| `environment/arc_gym_env.py` | Gymnasium adapter for ARC-AGI-3: frame stacking, action discretisation, action masks, resets and training reward |
| `rl_agents/dqn/` | DQN agent, Q-network and bit-packed replay buffer |
| `rl_agents/ppo/` | PPO agent (`ppo_agent.py`), novelty bonus (`novelty.py`) and action-history memory (`action_memory.py`) |
| `rl_agents/policy_agent.py` | Wraps a trained agent in the official Agent interface |
| `evaluation/rhae.py` | Implementation of the ARC-AGI-3 action-efficiency (RHAE) scoring formula |
| `kaggle/` | Inference code embedded in the competition notebooks (`my_agent.py`, `runtime.py`) |
| `scripts/` | Training, evaluation and diagnosis scripts (below) |
| `tests/` | Unit tests for the adapter, both agents, the improvements and the scoring |
| `SPLIT.md` | How the development and held-out games were chosen |

## Setup

Python 3.12 or later.

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   or   source .venv/bin/activate
pip install -r requirements.txt               (CPU)
pip install -r requirements-cuda.txt          (NVIDIA GPU, CUDA 13)
python -m pytest
```

All 49 tests should pass. Running games also requires access to ARC-AGI-3, either through an API key or a local game cache (`--offline`).

## Reproducing the experiments

Train one agent on one game:

```
python scripts/run_baseline.py --agent dqn --games ft09 --seed 0 --env-steps 50000 --max-actions 1000 --out-dir runs/dqn_ft09
python scripts/run_baseline.py --agent ppo --games ft09 --seed 0 --env-steps 50000 --max-actions 1000 --out-dir runs/ppo_ft09
```

Improvement flags:

| Flag | Agent | Effect |
|---|---|---|
| `--novelty-bonus 0.01` | DQN | First-visit screen bonus |
| `--noop-masking` | DQN | No-change action memory |
| `--ppo-novelty-bonus 0.01` | PPO | Count-based screen bonus |
| `--ppo-action-memory` | PPO | Action-history input planes |
| `--click-grid 16` | both | 16-by-16 click lattice |

Evaluate a frozen checkpoint:

```
python scripts/eval_checkpoint.py --agent ppo --checkpoint runs/ppo_ft09/ppo_seed0/ppo.pt --games ft09 --episodes 20 --seed 100 --stochastic --tag seed0_sampled
python scripts/eval_checkpoint.py --agent dqn --checkpoint runs/dqn_ft09/dqn_seed0/dqn.pt --games ft09 --episodes 20 --seed 100 --eval-epsilon 0.3 --tag seed0_eps0.3
```

Full experiment sets:

| Script | Runs |
|---|---|
| `run_ppo_experiments.py` | PPO baseline on all development games and seeds |
| `run_ppo_novelty.py` | PPO novelty improvement |
| `run_ppo_action_memory.py` | PPO 16-by-16 control, memory + novelty, memory only |
| `run_final_supplement.py` | Random reference and combined DQN ablation |
| `run_final_holdout.py` | Held-out transfer evaluation |
| `score_rhae.py` | ARC-AGI-3 RHAE scores for all recorded evaluation episodes (no replay needed) |
| `scan_games.py` | Random-policy screen of the public games |

Diagnosis:
- `diagnose_ppo_visual.py`, `build_ppo_replay.py` and `check_ft09_click_coverage.py` produce the replay and click-coverage analysis in report Section 6.
- `build_final_evidence.py` builds the report figures.
- `build_kaggle_package.py` builds the competition notebooks and model dataset from trained checkpoints.

## Key conventions

- An episode ends on WIN or after 1,000 actions. GAME_OVER resets the game inside the same episode.
- Reward (training only): +1 per level, +5 on WIN, −0.05 on death, −0.002 per action.
- `--seed` on evaluation sets the action-sampling seed, not the game layout.
- Training logs record cumulative wall time in `wall_s`.
