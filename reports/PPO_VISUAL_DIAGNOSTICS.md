# PPO visual diagnostics — 23 October 2026

The strongest finding is an action-interface limitation: on ft09's second level,
the current 8×8 click lattice cannot reach any puzzle tile. This is separate
from the greedy policy's repeated-toggle behaviour on the first level.

## Protocol and provenance

Six frozen-policy diagnostic episodes, each with a 1,000-action budget, on the
RTX 3050 Laptop GPU (`cuda:0`). Novelty-trained checkpoints from
`runs/ppo_novelty_20261023/novelty`; policy sampling seed 100 set after loading.
No training, novelty reward, no-op masking, or checkpoint changes. This is a
selected trajectory study, not a replacement for the full three-seed evaluation.
Sampling seeds do not seed the underlying ARC engine.

At the user's explicit request, this analysis read the development game source
and logged private game-state fields. Those fields were never supplied to the
policy or used to choose actions. Held-out game sources were not inspected.
Disclose this source-informed diagnosis in subsequent reporting and distinguish
it from the original observation-only experiments.

The recorder retains every policy decision, policy probabilities for the top
three actions and chosen action, value, entropy, exact observation grids,
level counters, and raw responses including deaths before automatic resets.
Only the final frame of each action is shown in the default before/after viewer;
intermediate frames are retained in the compressed records.

## Observed episodes

| Game / training seed / evaluation mode | Levels completed | Deaths | Unchanged observations |
|---|---:|---:|---:|
| ft09 / 2 / sampled | 1 | 1 | 94.8% |
| ft09 / 2 / greedy | 0 | 23 | 22.0% |
| ft09 / 0 / sampled | 0 | 2 | 91.7% |
| dc22 / 0 / sampled | 0 | 7 | 47.6% |
| g50t / 0 / sampled | 0 | 7 | 21.7% |
| m0r0 / 0 / sampled | 0 | 6 | 55.6% |

No full-game wins. A no-change percentage compares consecutive returned
observations, not every intermediate engine frame. Deaths consume an additional
automatic reset action, so the number of policy decisions can be below 1,000.

## ft09: unreachable next level

The sampled seed-2 agent completes the first level at action 310 (policy decision
309). Its remaining 690 decisions all leave the returned screen unchanged.
The private tile states and move allowance also stay unchanged. This is not
merely a low-probability action: none of its 64 available click positions hits
an interactive puzzle tile on that level.

The read-only coverage audit replays the recorded actions to reach the second
level, then maps all lattice positions through the engine camera and checks
the same tile tags used by the game's action handler. It does not take
counterfactual actions or modify the game state.

| Level (human numbering) | Puzzle tiles | Tiles reachable with 8×8 | Tiles reachable with 16×16 |
|---|---:|---:|---:|
| 1 | 8 | 8 | 8 |
| 2 | 13 | 0 | 13 |

The 8×8 centres are screen coordinates 4, 12, 20, …, 60. On level 2 these
positions pass between the interactive rows. A 16×16 lattice has 26 click
positions hitting the 13 tiles. Coverage is proven for these two recorded
levels only; it is not evidence that 16×16 covers every later game layout.

This interface is shared with DQN. Any DQN using the same click-grid setting
faces the same level-2 reachability limit. Increasing the grid changes the
policy's output size, so existing checkpoints cannot simply be evaluated with
the larger action space. A fair new experiment needs compatible newly trained
policies and explicit accounting for the changed action abstraction.

## ft09: greedy toggling, not useful progress

Greedy seed 2 chooses action 61, click (52,52), 762 times out of 977 decisions.
These clicks target a puzzle tile, repeatedly change its colour, consume the
move allowance, and ultimately lose/reset. The episode has 23 deaths and no
level completion. A low no-change rate therefore does not imply learning the
puzzle: repeatedly reversing a change still counts as visual activity.

The sampled policy sometimes escapes this first-level cycle, while seed 0
spends 906/998 decisions on background/feedback targets. Source inspection
confirms that ft09 can issue feedback animation for ineffective clicks and
that only specific tile clicks change the puzzle. Screen novelty alone is not
a reliable measure of task progress. These trajectories do not prove that the
novelty reward caused the cycling; that requires an ablation.

## Other games: provisional observations

- dc22: 7 deaths and no level completion despite many visual changes. A
  moving screen or timer cannot be treated as progress.
- g50t: ACTION2 (down) accounts for 469/993 decisions; 7 deaths, no level
  completion. The source has an action-driven timer, so screen changes may
  include budget consumption even when movement is ineffective.
- m0r0: click (4,4) accounts for 554/994 decisions (55.7%); only 29 decisions
  use directional actions 1–4. Its source separates selection/mode-changing
  clicks from directional movement. The policy heavily favours clicking over
  executing movement sequences. This episode is insufficient to isolate every
  rule the agent has failed to learn.

## Next experiment priority

1. Correct and audit the click abstraction first, starting with a simple 16×16
   lattice. Check coverage on development layouts before spending another
   full training budget. Preserve existing runs as the 8×8 baseline.
2. Retrain the corresponding PPO with the same training budget and seeds;
   separate this change from reward or network changes.
3. Then revisit repeated-action/memory improvements. Penalising or masking
   only unchanged observations would miss the greedy colour-toggle cycle.

No algorithm or action-space changes were made in this diagnostic task.

## Files and reproduction

- `scripts/diagnose_ppo_visual.py`: six frozen CUDA evaluation episodes.
- `scripts/check_ft09_click_coverage.py`: replay and read-only coverage audit.
- `scripts/build_ppo_replay.py` and `scripts/ppo_replay_template.html`: interactive
  before/after replay with click overlays and event jumps.
- `runs/ppo_visual_diagnostics_20261023/summary.json`: metrics and checkpoint SHA256s.
- Same directory: `*.json.gz` full traces, `ft09_click_coverage.json`,
  `ft09_diagnostic.png`, and standalone `replay.html`.

Observed frame colours come from the installed ARC renderer's `COLOR_MAP`.
The viewer was checked against recorded observations and the coverage audit;
replay controls display stored data and never interact with the game engine.
