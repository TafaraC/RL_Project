# Evidence guide

All numbers here are local measurements, not Kaggle leaderboard scores.

- `ppo/verified_results.csv` and `ppo/evaluation_logs/`: 108 frozen PPO evaluation files (2,160 episodes), with three training seeds and 20 rollouts per checkpoint/mode. The historical baseline greedy layout is excluded from this export.
- `dqn_results.csv` and `dqn/`: 75 historical evaluation files recovered from Nathan's complete archive. Historical eval configurations were not saved; directory names identify intended policy settings. Some greedy files have one episode, so inspect each denominator.
- `training_cost.csv` and `training/`: 99 completed training logs, including configurations/runtime records where available. These are training behaviour, not frozen policy performance. `wall_s` is cumulative: use each file's final value for its runtime.
- `holdout_results.csv`, `holdout/`, `holdout_protocol.json`: 270 frozen episodes on cd82, tr87 and vc33, six stages, three independent training seeds and five rollouts per checkpoint/game. One DQN baseline rollout completes one tr87 level; no full-game wins.
- `supplement_results.csv` and `supplement/`: final random reference (240 episodes: three seeds, 20 per development game) and combined DQN novelty-plus-memory ablation (60 ft09 episodes). Random ft09 counts are 2, 5, 8; combined DQN counts are 19, 8, 18, each out of 20.
- `random/`: older ft09-only random and memory-masked random runs. Their different random-stream schedule produces different counts; they are historical supporting evidence, not the final reference table.

Episode successes mean at least one level cleared, not a full game won. Training seeds are independent models; repeated evaluation episodes of one checkpoint are not independent training runs. Sampled PPO and epsilon-0.3 DQN differ in decision policy, and CPU DQN/GPU PPO training costs are not a clean hardware-matched comparison.
