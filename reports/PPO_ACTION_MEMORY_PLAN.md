# PPO improvement 2: action history with a repaired click abstraction

Protocol frozen 23 October 2026, before the new full training runs.
Branch: `codex/ppo-action-memory`. Built on our existing custom PPO and novelty
wrapper from the Nathan2-based experiment; Nathan's DQN model is unchanged.

## Limitation recorded

The visual diagnostic confirmed that the 8×8 click lattice cannot hit any of
ft09 level 2's 13 puzzle tiles. A 16×16 lattice reaches all 13. The sampled
seed-2 policy reaches this level, then makes 690 ineffective decisions. Its
greedy counterpart repeatedly toggles a first-level tile and loses 23 times.
The latter changes pixels, so a no-change penalty/mask alone would not address it.
These diagnoses do not establish that novelty reward caused the behaviour.

Coverage has been verified for ft09 levels 1 and 2, not every later layout or
other game. The larger action space may make exploration harder. Existing 8×8
checkpoints and results stay intact; they are not compatible with 16×16 outputs.
The correction applies only to the explicitly configured new PPO experiments.

At the user's request, development-game source and private state were inspected
for diagnosis. Disclose this source-informed development. The new policy uses
only public observation frames and the agent's own chosen actions. No private
state, game-specific action rules, solver, target masks, or held-out source is
used by training or evaluation.

## Substantive intervention

Hypothesis: four frames alone cannot identify which actions led to identical
screens, and do not explicitly identify long same-action runs. A small action
history may let PPO learn when to switch actions without banning legitimate
repetition or waiting. This is a representation/memory improvement; the finer
grid is separately controlled and is not presented alone as sufficient for the
assignment's second substantive improvement requirement.

Prepend 16 binary planes to the existing 64 observation channels:

- Seven planes identify the previous simple action (one all-one plane).
- One spatial plane marks the previous click's lattice cell (4×4 pixels with
  the new 16×16 grid); empty for simple actions.
- Four thermometer-coded planes represent the same-action streak, capped at 4.
- Four thermometer-coded planes represent the unchanged-screen streak, capped
  at 4; only the newest public frame is compared.

The same CNN architecture consumes 80 channels instead of 64, and outputs 263
action logits plus a value (7 simple actions + 256 clicks). PPO's clipped loss,
GAE, rollout length, optimizer, entropy bonus and learning rate are unchanged.
Binary encoding keeps the existing packed rollout buffer valid. The exact
augmented observations are stored and reused for old/new policy comparisons
and next-value bootstrapping. Novelty remains computed on the newest 16 colour
channels, ignoring memory.

Memory starts empty at each episode and clears on explicit and automatic
resets. It persists across ordinary level transitions. Clearing uses an
explicit adapter reset event, not an inference from pixel changes. Repeat
streaks count consecutive decisions regardless of pixel changes; unchanged
streaks count consecutive identical latest frames regardless of chosen action.

Training, evaluation and the submission adapter share `ActionMemory`. Each
checkpoint saves an observation-schema identifier; old pixel-only checkpoints
remain loadable, and mismatched memory settings fail clearly.

## Controlled comparisons

| Variant | Click grid | Count novelty | Action memory | Purpose |
|---|---|---|---|---|
| Existing improvement 1 | 8×8 | On | Off | Preserved historical reference |
| grid_control | 16×16 | On | Off | Isolate action-coverage change |
| memory_novelty | 16×16 | On | On | Candidate improvement 2 |
| memory_only | 16×16 | Off | On | Remove novelty from candidate |

All three new variants train from scratch on development games ft09, dc22,
g50t and m0r0, training seeds 0/1/2, 50,000 requested decisions per checkpoint,
1,000 actions per episode. The runner finishes the final episode and records
actual decisions. CUDA is required with no CPU fallback. Count novelty is
unchanged: beta 0.01, episode cap 0.5. CPU thread counts are fixed at two.

Primary evaluation samples PPO; greedy is secondary. Each mode evaluates 20
episodes with policy sampling seeds 100–119, no novelty reward or learning.
The ARC engine layout is not seeded by this wrapper; this remains disclosed.
Three variants × four games × three training seeds = 36 training runs and
72 evaluation jobs (108 sequential jobs, one GPU job at a time).
The order interleaves variants for each game/seed, starting with ft09.

Assess completion rates, level-completion action counts, failed-run budgets,
per-training-seed variation, learning curves and runtime. Do not declare an
improvement based on more movement, shaped reward, or one good checkpoint.
Held-out games stay untouched until design decisions are frozen. Local
completion metrics are not official competition scores.

## Validation and execution

- Unit tests cover binary memory, click locations, unchanged observations,
  repeat streaks despite changing frames, caps and reset boundaries.
- The submission adapter is compared with the training wrapper across a death
  and automatic reset using mocked official frames.
- CUDA tests cover PPO updates, strict checkpoint schemas, and a controlled
  history-dependent task with identical screen pixels.
- A short real-game CUDA training/save/load/evaluation smoke run precedes the
  full experiment and is kept separate from benchmark results.

Run `python scripts/run_ppo_action_memory.py` from the repository virtual
environment. Outputs: `runs/ppo_action_memory_20261023`; status and exact
commands are saved in JSON, with per-job logs and completion markers. Completed
jobs can be skipped. Partial jobs are not silently overwritten or resumed with
missing optimizer state; preserve/archive them before an explicit restart.

Submission factory options for the new model must include `click_grid=16`,
`history_len=4`, `ppo_action_memory=True`, `stochastic=True`, and the intended
action budget. Official competition integration/scoring still needs separate
validation. This experiment does not by itself finish the assignment deliverables.

Validation outcome before launch: the existing suite plus memory tests passed
(44 tests), followed by the additional controlled history-learning CUDA test
(1 test). A real ft09 smoke run completed 999 training decisions / 1,000 actions
on the RTX 3050 with finite logged update metrics, then its checkpoint loaded
and ran a 128-action wiring check on each of the four development games. This
short smoke test is not performance evidence or a cross-game evaluation claim.
