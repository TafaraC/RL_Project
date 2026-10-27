# When exploration cannot reach the puzzle: DQN and PPO on ARC-AGI-3

**WitsC**: Nathan Le Roux, Tafara Chibanda, Samuel Snyman, Joshua Snyman (University of the Witwatersrand)

Code: https://github.com/TafaraC/RL_Project

## Abstract

We study student-scale DQN and PPO agents on the interactive puzzles of ARC-AGI-3. We test visible-screen novelty, action memory and a finer click grid. Novelty helps PPO on one development game but hurts DQN, and results vary strongly across seeds. Visual replay shows that the original click grid cannot reach any second-level puzzle tile. Restoring reachability does not improve completion within our budget. Neither algorithm wins a complete game or transfers reliably to held-out games. Our contribution is a reproducible investigation that separates exploration, action coverage and screen activity from puzzle solving.

## Motivation and prior work

Interactive puzzles require an agent to discover controls, infer rules and carry out useful action sequences from visual feedback alone. With sparse success signals, a promising action is hard to tell apart from an accidental screen change. We implement DQN [1] and PPO [2] and compare frozen policies on the same games and action budgets. Our novelty bonus follows count-based exploration [3]. We report an empirical diagnosis, not a new algorithm.

## Approach

**Observation and actions.** The agent sees the last four 64-by-64 frames, each one-hot encoded into 16 colour planes. A three-layer CNN maps this to action outputs. The action set has seven simple actions plus 64 click positions on an 8-by-8 grid, masked to the actions the game allows.

**Reward.** Training reward is +1 per level, +5 per win, −0.05 per death and −0.002 per action. This reward is for learning only and is not the competition score.

**DQN.** DQN learns masked action values from replay with a target network and Huber loss. Settings: discount 0.99, learning rate 0.00025, batch size 64, an update every four decisions after 1,000 transitions, and a target copy every 1,000 updates. There are two improvements:
1. A first-visit screen bonus during training.
2. A per-screen memory that masks actions which previously left the screen unchanged. The full mask is restored if every action would be excluded. This targets what the novelty agent's logs showed: its greedy ft09 policy repeats one ineffective action for 99% of its moves.

DQN is evaluated with epsilon 0.3, chosen on development games; this introduces selection bias.

**PPO.** PPO collects 128-transition rollouts. It computes generalised advantage estimates (discount 0.99, lambda 0.95) and then runs four epochs of minibatch updates with these settings: batch 32, clip 0.2, value coefficient 0.5, entropy coefficient 0.01, learning rate 0.0003 and gradient clipping 0.5. There are two improvements:
1. A training bonus of 0.01/√N for the current screen's visit count N, capped at 0.5 per episode.
2. After visual diagnosis, a 16-by-16 click grid (263 actions) plus 16 binary planes recording the previous action, repeated actions and unchanged screens.

A 16-by-16 novelty-only control separates the effect of memory from the grid change. A memory-only condition removes novelty.

## Experimental protocol and results

Development games (ft09, dc22, g50t, m0r0) and held-out games (cd82, tr87, vc33) come from a seeded shuffle recorded before any results. Each development game is trained separately for seeds 0, 1 and 2, with about 50,000 decisions per run. Each checkpoint is evaluated for 20 frozen episodes of up to 1,000 actions. Success means completing at least one level.

| Condition | ft09 successes by seed (each /20) | Pooled |
|---|---|---|
| Random reference | 2, 5, 8 | 25.0% |
| DQN baseline, epsilon 0.3 | 17, 20, 5 | 70.0% |
| DQN novelty | 7, 1, 5 | 21.7% |
| DQN no-change memory | 3, 17, 5 | 41.7% |
| DQN novelty + memory | 19, 8, 18 | 75.0% |
| PPO baseline, 8×8 | 2, 0, 3 | 8.3% |
| PPO novelty, 8×8 | 4, 4, 20 | 46.7% |
| PPO novelty, 16×16 control | 0, 9, 1 | 16.7% |
| PPO memory + novelty, 16×16 | 1, 1, 1 | 5.0% |
| PPO memory only, 16×16 | 2, 0, 4 | 10.0% |

- No condition completes a level on dc22 or m0r0.
- On g50t, PPO memory-only completes one level in 60 episodes.
- No condition wins a full game, and every greedy PPO evaluation completes zero levels.
- Combined DQN beats each single DQN improvement but only slightly beats the baseline.
- Three trained seeds underpin each rate, so a single 20/20 checkpoint does not show reliability.
- DQN trained on the CPU and PPO on a GPU, so their training times are not directly comparable.
- Under the official RHAE metric, computed locally from the published human baselines, the best pooled ft09 score is 0.36% (DQN baseline and combined DQN). Successful episodes need a median of 372 actions for a level a human clears in 43.

## What visual inspection explains

A sampled ft09 novelty checkpoint clears level one, then makes 690 ineffective decisions. A coverage check finds that the 8-by-8 grid reaches all eight first-level tiles but none of the 13 second-level tiles; the 16-by-16 grid reaches all 13. Learning cannot select an action that does not exist.

Coverage is not sufficient, though. The finer grid lowers first-level completion in our experiment, possibly because the larger action space is harder to explore. Action memory also fails to improve on the matched finer-grid control. This does not show that history is unhelpful in general; our representation or budget may be inadequate. A greedy policy that repeatedly toggles one tile shows why screen changes, pixel novelty and shaped return are no substitute for task progress.

## Transfer and limitations

Held-out games test transfer. All three ft09-trained seeds of each algorithm's baseline, novelty and memory stages play five frozen episodes on cd82, tr87 and vc33: 270 episodes in all. One DQN baseline episode clears a tr87 level; every other episode clears none, and no game is won.

Development-game source and state were inspected to diagnose failures. Agents use only public frames and their own actions. Coverage was checked for two ft09 levels only. Evaluation reuses training games except in the transfer test, and all results are local.

The nominated entry is DQN baseline seed 1 with epsilon 0.3, chosen on development results; PPO novelty seed 2 is the PPO alternative. Claude assisted with the scaffold; Codex assisted with implementation, debugging, analysis and writing. Nathan Le Roux worked on DQN and Tafara Chibanda on PPO; Samuel Snyman and Joshua Snyman worked on setup and documentation.

## Conclusion

Our strongest result is a diagnosis rather than a solver: an action representation that cannot reach the puzzle blocks progress however well the policy is optimised. Novelty helps on one development game, while finer clicks and action memory do not lead to broader solving within our budget. Future work should test reachable action representations and sequence learning under matched budgets, then evaluate frozen designs on untouched games.

## References

1. Mnih et al. (2015). [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236). Nature 518.
2. Schulman et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
3. Bellemare et al. (2016). [Unifying count-based exploration and intrinsic motivation](https://arxiv.org/abs/1606.01868). NeurIPS.
4. ARC Prize (2026). [ARC-AGI-3 documentation](https://docs.arcprize.org/arc-prize-2026).
