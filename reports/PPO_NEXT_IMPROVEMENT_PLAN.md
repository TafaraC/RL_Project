# Conditional plan for PPO improvement 2

**Superseded on 23 October after completion and visual diagnosis.** The full
novelty experiment completed. See `PPO_VISUAL_DIAGNOSTICS.md` for the confirmed
8×8 click-coverage blocker and repeated-toggle evidence, and
`PPO_ACTION_MEMORY_PLAN.md` for the implemented, controlled next experiment.
The historical planning snapshot below is retained for provenance; its running
job counts and "not implemented" statements are no longer current.

Planning snapshot: 23 October 2026. This is a decision plan, not permission to
change the current experiment mid-run. No second variant has been implemented.

## Evidence available now

The current pipeline has completed 21/48 jobs and is training dc22 seed 0.
On ft09, all three novelty checkpoints have completed both evaluation modes:

| Training seed | Baseline sampled levels/episodes | Novelty sampled | Novelty greedy |
|---|---|---|---|
| 0 | 2/20 | 4/20 | 0/20 |
| 1 | 0/20 | 4/20 | 0/20 |
| 2 | 3/20 | 20/20 | 0/20 |

These count episodes with at least one level completed, not game wins. Sampled
success rises from 5/60 to 28/60, but seed 2 contributes most of the improvement.
The other games' novelty results are pending. Do not declare broad improvement
or infer causality from the aggregate alone. No full-game wins are established.
Sampling is the primary protocol; greedy is a diagnostic, not a requirement for
a valid stochastic PPO policy. Making greedy performance nonzero is not the goal.

## Finish and inspect before selecting an intervention

1. Confirm all 48 jobs finish, 12 improved checkpoints exist, each evaluation has
   20 episodes, losses are finite and recorded settings match the fixed protocol.
2. Compare baseline versus novelty separately by game, training seed and action
   selection mode. Report mean and variability across three training seeds; the
   20 episodes from one checkpoint are not 20 independently trained models.
3. Compare level completion, game completion, and actions used on successes and
   failures. Use recorded level-completion action counts for efficiency among
   successes; total episode actions alone can stay 1,000 after a level is cleared.
4. Inspect representative successes and failures on development games. Existing
   logs only give aggregate no-change counts, not repeated same-action streaks;
   collect short diagnostic trajectories before claiming a repeated-action cause.
5. Inspect separate intrinsic/extrinsic returns and learning stability. If bonus
   saturation is implicated, record cap-hit timing and reward from repeated vs
   new observations in a small diagnostic. Current episode totals cannot establish
   which observations earned the bonus. Never treat unchanged pixels as proof of
   unchanged hidden state; repeated actions or waiting may be necessary.

## Decision branches

| Completed evidence | Next step |
|---|---|
| Novelty improves sampled task completion but failed trajectories repeatedly choose the same action without visible progress | Preferred candidate: action-history features for PPO (below). |
| Results differ sharply across training seeds, without evidence of repeated-action failures | Inspect update KL, entropy, loss and available training curves first. Do not invent a memory explanation or select the best seed as the result. A later update-stability change needs an observed instability. |
| Bonus routinely saturates on familiar screens before meaningful progress | Candidate instead: award novelty only on first visits or actual screen changes. Freeze this as a separate reward-design experiment; do not combine it with memory in the same next comparison. Disclose overlap with Nathan's first-visit idea. |
| Three development games remain at zero task progress for both agents | Keep the failure visible. Check trajectories and whether the fixed coarse click abstraction prevents useful interactions. Do not scan or tune held-out games. A finer action abstraction is a separate later hypothesis, not an automatic addition. |
| Novelty regresses task completion | Retain the negative result and test the next candidate against both plain PPO and novelty PPO; do not call the combination an improvement without evidence. |
| All comparisons are inconclusive or incomplete | Finish verification and report uncertainty rather than adding several mechanisms at once. |

## Preferred candidate: small action-history input

Hypothesis, conditional on trajectory evidence: the four-frame observation does
not tell the network which actions produced identical frames. Supplying recent
action information may let the policy avoid repeating unsuccessful choices while
still allowing legitimate repeated actions and waiting.

Keep the current CNN, and concatenate a short vector to its shared feature output:
- One-hot encoding of the previous agent-selected action (all zero at episode start).
- A capped, normalized count of consecutive agent steps with unchanged latest grids.

The actor and critic consume the same extra information. This is a modest state-
representation change, not an RNN, learned world model, or new RL algorithm. Do not
hard-ban actions merely because the visible screen stayed the same. The old PPO
baseline and novelty checkpoints remain untouched; train the new architecture from
scratch. Nathan's DQN networks and experiments remain unchanged.

Implementation requirements to settle before coding:
- Define the streak cap and scaling once before experiments; no parameter sweep.
- Store the exact history vector used for each collected action in the PPO rollout,
  alongside its fixed old log probability and legal-action mask.
- Construct next-state history consistently for value bootstrapping; clear history
  after episode resets and game-over/explicit RESET boundaries using explicit reset
  information. Do not infer reset events from changes in pixels.
- Use identical history updates during training, sampled/greedy evaluation, and
  submission. Training-only novelty must not enter evaluation rewards or state.
- Version checkpoints with feature dimensions; retain loading of old checkpoints.
- Add a small controlled history-dependent task test, boundary tests and an end-to-end
  GPU save/load/evaluate smoke check before full runs.

## Comparison and ablation matrix

| Variant | Novelty | Action history | Purpose |
|---|---|---|---|
| PPO baseline | Off | Off | Existing reference |
| PPO improvement 1 | On | Off | Current experiment |
| PPO improvement 2 | On | On | Incremental comparison |
| History-only PPO | Off | On | Isolate memory contribution / remove novelty |

Only execute this matrix if action history is the selected intervention. If reward
redesign is selected instead, define its own matrix before starting. Do not implement
both as one 'improvement 2'. Reuse comparable frozen evaluations rather than retrain
completed baselines merely for this matrix.

Keep four development games, training seeds 0/1/2, 50,000 requested steps, 1,000 actions
per episode, GPU, and PPO update settings unchanged. New candidate and history-only
ablation require 24 training runs in total. Use sampled evaluation as primary, greedy
as secondary, 20 episodes with sampling seeds 100-119 per checkpoint. Record actual
interaction count, runtime, GPU usage and software versions. Engine layout seeding
remains a disclosed limitation; do not silently change it for only one variant.

Accept an improvement based on task completion and efficiency across training seeds,
not just reduced no-change rate, increased shaped return, or one successful checkpoint.
With only three training seeds, show individual results and avoid overstating statistical
certainty. Keep held-out environments untouched until the designs are frozen. The final
report still needs official scoring evidence; these local completion results are not
Kaggle scores.

## Project scope and schedule

Prefer one explainable intervention, one matched ablation, and a short evidence table
(limitation -> hypothesis -> change -> result). Four days remain before the stated
27 October deadline; avoid a large architecture search. Do not alter the running
novelty experiment or start the second experiment solely on this planning document.
