# Evidence

These tables hold every evaluation and training episode behind the report. All numbers are local measurements on public games, not Kaggle leaderboard scores. Each file opens in Excel or any spreadsheet tool.

| File | One row per | Use it to |
|---|---|---|
| `results_summary.csv` | condition × game × training seed × evaluation policy | Check the report's tables quickly |
| `evaluation_episodes_development.csv` | evaluation episode on a development game | Inspect individual frozen-policy episodes |
| `evaluation_episodes_holdout.csv` | evaluation episode on a held-out game | Inspect the transfer test |
| `training_runs.csv` | training run | See training cost, settings and hardware |
| `training_episodes.csv` | training episode | Rebuild the learning curves (report, Figure 1) |
| `rhae_scores.csv` | condition × game × training seed × evaluation policy | Check the ARC-AGI-3 scores in report Table 2 |
| `human_baselines.csv` | game level | See the human action counts the scores are measured against |

To reproduce Table 1, filter `results_summary.csv` to `split = development`, `game = ft09` and `eval_policy` = `epsilon 0.3`, `sampled` or `uniform legal`.

## ARC-AGI-3 scores

`rhae_scores.csv` applies the official relative human action efficiency (RHAE) methodology to every recorded evaluation episode:
- A level solved in *a* actions scores min(1.15, (*h*/*a*)²), where *h* is the human baseline for that level.
- A game score weights level *l* by *l*, over all of the game's levels; unsolved levels count as zero.
- Scores are given in percent, so 100 means every level solved at human efficiency.

`rhae_mean_percent` is the mean over that row's episodes. Rebuild the file with `source/scripts/score_rhae.py`.

## Games and protocol

- **Development games:** dc22, ft09, g50t, m0r0. Used for training, diagnosis and choosing improvements.
- **Held-out games:** cd82, tr87, vc33. Used once, for the final transfer test, after all design choices were fixed.
- The split comes from a seeded shuffle of the 25 public games (`source/SPLIT.md`).
- **Training:** each development game is trained separately with training seeds 0, 1 and 2, for about 50,000 decisions per run.
- **Development evaluation:** 20 frozen-policy episodes per checkpoint, sampling seeds 100–119, with 1,000 actions each.
- **Held-out evaluation:**
  - All three ft09-trained seeds of six stages: DQN baseline, novelty and no-change memory; PPO baseline, novelty and memory + novelty.
  - Five episodes per checkpoint per game, with sampling seeds 100–104 and 1,000 actions each: 270 episodes in total.
  - No parameters change during evaluation. Memories reset between games.

## Column guide

- `condition`: agent version, using the names in the report.
- `training_seed`: the independently trained model (0, 1 or 2). Episodes from one seed are repeated rollouts of one model, not independent models.
- `eval_policy`:
  - `epsilon 0.3`: DQN with 30% random legal actions.
  - `greedy`: highest-valued or most probable action.
  - `sampled`: PPO sampling from its policy.
  - `uniform legal`: random reference.
  - `not recorded`: the setting was not saved for that run.
- `levels_completed`, `won`: success measures. The report counts an episode as successful if `levels_completed` ≥ 1.
- `actions_at_level_completions`: action count at which each level was cleared. Empty if none was cleared.
- `no_change_fraction`: share of actions that left the screen unchanged.
- `most_common_action`, `most_common_action_share`: the agent's most frequent action index and how often it was used.
- `wall_s_cumulative`: elapsed seconds since the start of that run, so a run's total time is its last value.
- `decisions_so_far` (training): environment decisions taken up to the end of that episode.
- `original_run`: the log folder the episode was merged from, kept so any row can be traced to its source run.

## Notes

- Earlier DQN evaluations did not save their settings. Their condition and policy are taken from the run folder names, which `original_run` keeps.
- DQN baseline training runs predate configuration logging, so their settings columns in `training_runs.csv` are empty. They use the defaults in report Section 3.1.
- Some early greedy DQN evaluations have only one episode per seed. Check `episodes` in the summary before comparing rates.
- The rows `Random (earlier run)`, `Random + no-change mask (earlier run)` and `DQN, no-change mask (earlier diagnostic)` are earlier diagnostics. They are not part of the report's main comparison.
- DQN was trained on the CPU and PPO on an RTX 3050 Laptop GPU, so training times are not a like-for-like algorithm comparison.
