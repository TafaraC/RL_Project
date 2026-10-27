# When exploration cannot reach the puzzle: a small PPO study on ARC-AGI-3

**WitsC** — Nathan Le Roux (2717495), Tafara Chibanda (2683459), Samuel Snyman (2677730), Joshua Snyman (2677722)

## Abstract

We investigate a student-scale reinforcement-learning baseline for interactive ARC-AGI-3 puzzles. Starting from convolutional proximal policy optimization (PPO), we test visible-screen novelty, finer click discretization, and explicit action history. Novelty increases first-level completion on one development game, but performance varies substantially across three training seeds. Visual replay reveals a structural failure: the original click lattice cannot reach any puzzle tile in that game's second level. Repairing reachability does not improve measured completion under our fixed training budget, and action history underperforms its matched control. Our contribution is a reproducible development study distinguishing exploration, action coverage, and observable activity from actual puzzle solving. We do not demonstrate generalisation or a complete-game solution.

## Motivation and prior work

Interactive puzzles require discovering controls, inferring rules, and executing useful sequences from visual feedback. Sparse success signals make it difficult to distinguish promising actions from accidental screen changes. Our project implements both DQN, following the value-learning approach of Mnih et al. [1], and PPO [2]. This write-up focuses on the verified PPO experiments; available DQN evidence does not support a matched final comparison.

PPO alternates environment interaction with several updates to a clipped policy objective. Our additions are deliberately simple: an exact-screen visitation bonus and a short action-history representation. We claim an empirical diagnosis, not a new reinforcement-learning algorithm.

## Approach

The agent observes four 64-by-64 frames, each encoded into 16 binary colour planes. A three-layer convolutional network produces masked categorical action probabilities and a value estimate. The initial action set contains seven simple actions and 64 click coordinates on an 8-by-8 lattice. Masks follow the public available-action interface.

For each 128-transition rollout, we retain observations, actions, masks, old log probabilities, rewards and values. Generalized advantage estimation uses discount 0.99 and lambda 0.95. PPO then performs four epochs of minibatch updates (batch size 32), with clipping 0.2, value coefficient 0.5, entropy coefficient 0.01, Adam learning rate 0.0003 and gradient clipping 0.5. Terminal states have zero bootstrap value; action-limit truncations bootstrap without carrying advantages across episode boundaries.

Training reward combines level completion (+1), winning (+5), death (-0.05), and an action cost (-0.002). These are learning signals, not competition scores. The first improvement adds 0.01/sqrt(N) for the current visible screen's visit count N, capped at 0.5 total intrinsic reward per episode. Counts survive automatic death resets within that episode. No intrinsic reward or weight updates are used during evaluation.

Following visual diagnosis, we expand the click lattice to 16-by-16, producing 263 actions. The second substantive intervention adds 16 binary planes describing the previous simple action or click location, consecutive repeated actions, and consecutive unchanged screens. Both streaks saturate at four. Memory clears on reset and persists across ordinary level transitions. A finer-grid novelty-only control separates memory's effect from the action-space change; a memory-without-novelty condition provides an additional ablation.

## Experimental protocol and local results

Development games are ft09, dc22, g50t and m0r0, selected by a recorded shuffle before baseline results. Each game has a separately trained agent for seeds 0, 1 and 2, requesting 50,000 environment decisions per run. The runner finishes its final episode, so actual counts exceed this threshold. Episodes allow 1,000 actions, including internal reset calls. Neural-network training uses an NVIDIA RTX 3050 Laptop GPU; game simulation runs on the CPU.

For each checkpoint we evaluate 20 frozen-policy episodes with action-sampling seeds 100–119. These seeds do not seed underlying game layouts. The following are **development diagnostics on games used for training**, not held-out public evaluation or leaderboard results. Success denotes an episode completing at least one level.

| PPO condition | ft09 successes by training seed, each /20 | Pooled ft09 rate |
|---|---|---|
| Baseline, 8-by-8 | 2, 0, 3 | 8.3% |
| Novelty, 8-by-8 | 4, 4, 20 | 46.7% |
| Novelty, 16-by-16 control | 0, 9, 1 | 16.7% |
| Memory + novelty, 16-by-16 | 1, 1, 1 | 5.0% |
| Memory only, 16-by-16 | 2, 0, 4 | 10.0% |

All conditions record zero level completions on dc22 and m0r0. On g50t, memory-only completes one level in one of 60 episodes; other conditions complete none. No condition wins a full game. Greedy evaluations also fail to complete levels. Thus sampling behaviour materially affects observed performance. Three trained seeds, rather than 60 independent models, underpin each pooled rate; the 20/20 novelty checkpoint should not be mistaken for reliable performance across initialisations.

## What visual inspection explains

A sampled ft09 novelty checkpoint clears level one, then makes 690 ineffective decisions. A read-only coverage audit finds that the 8-by-8 lattice reaches all eight first-level tiles but none of 13 second-level tiles. The 16-by-16 lattice reaches all 13. Learning cannot select an unavailable coordinate, making action coverage a necessary prerequisite for this interface.

However, coverage is insufficient: expanding the lattice reduces pooled first-level completion in our experiment. A larger exploration space is a plausible explanation, not an established cause. Similarly, memory fails to improve the matched finer-grid control. These results do not establish that history is generally unhelpful; the representation and budget may be inadequate.

A greedy policy repeatedly toggles a first-level tile. Its screen changes without meaningful progress, illustrating why movement, pixel novelty and shaped return cannot substitute for task completion.

## Limitations, reproducibility and submission status

Held-out games cd82, tr87 and vc33 have not been evaluated. We provide our implementation through the GitHub repository linked below. No official Kaggle leaderboard score is available. Reported results are local development diagnostics and do not establish held-out generalisation or official competition performance.

Development-game source and private state were inspected for diagnosis. Policies consume public frames and their own actions; no private-state features, game-specific scripts or target masks are supplied to training or evaluation. Source-informed development must nevertheless be disclosed and checked against applicable rules. Coverage was audited for two ft09 levels only. One host-memory training failure was restarted; incomplete outputs are excluded from final results.

Code is available at [TafaraC/RL_Project, revision 6079ab1](https://github.com/TafaraC/RL_Project/tree/6079ab1). Experiment configurations, per-seed logs and a script-generated evidence table accompany this write-up locally. End-to-end Kaggle execution remains unvalidated. The project used AI assistance, including Claude for scaffolding and Codex for implementation, debugging, analysis and writing. Nathan worked on DQN, Tafara coordinated PPO development, and Samuel and Joshua worked on setup and documentation.

## Conclusion

Our strongest result is a diagnosis rather than a solver: an unreachable action representation can prevent progress regardless of policy optimisation. Novelty helps one development level, while finer clicks and action memory do not yield broader solving within our budget. Future work should test reachable action representations and deliberate sequence learning under matched budgets, then evaluate frozen designs on untouched games.

## References

1. Mnih et al. (2015). [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236).
2. Schulman et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
3. ARC Prize (2026). [Paper Prize requirements](https://arcprize.org/competitions/2026/paper).
