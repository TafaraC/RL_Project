# WitsC: DQN and PPO agents for ARC-AGI-3

COMS4061A/COMS7071A Reinforcement Learning Assignment, 2026.

Nathan Le Roux (2717495), Tafara Chibanda (2683459), Samuel Snyman (2677730), Joshua Snyman (2677722).

## Contents

| Item | Description |
|---|---|
| `WitsC_Report.pdf` | The project report. Start here. |
| `kaggle/` | Competition notebooks and the model weights they load. `kaggle/README.md` explains how to run them. |
| `source/` | Full project code: environment adapter, DQN and PPO agents, improvements, experiment scripts and tests. |
| `evidence/` | Every evaluation and training episode behind the report, as five tables. `evidence/README.md` explains them. |
| `paper_track_writeup.md` | Condensed write-up (under 1,500 words) for the ARC Prize 2026 Paper Track. |
| `CONTRIBUTIONS.md` | Each member's contribution, AI assistance and reused resources. |

## Competition entry

- **Nominated notebook:** `kaggle/dqn_baseline.ipynb`. DQN baseline, training seed 1, epsilon 0.3, 8-by-8 click grid, 1,000 actions per game.
- **PPO alternative:** `kaggle/ppo_novelty.ipynb`. PPO with novelty bonus, training seed 2, sampled actions.
- **Code:** https://github.com/TafaraC/RL_Project
- **Paper Track write-up:** `paper_track_writeup.md`

The nominated checkpoint was chosen on development results before any held-out game was evaluated.

## Results in brief

- Best development result: on ft09, DQN with epsilon 0.3 completes a level in 42/60 episodes. Novelty-trained PPO completes one in 28/60.
- ARC-AGI-3 score (RHAE, computed locally): at most 0.36% on ft09 and below 0.01% on the held-out games. Agents need hundreds of actions where a human needs 43.
- No agent wins a full game.
- On the three held-out games, one DQN baseline episode completes one level. Every other episode completes none.
- The original 8-by-8 click grid cannot reach any tile on ft09 level 2. This limits every 8-by-8 agent, whatever its policy.

See the report for the full analysis.
