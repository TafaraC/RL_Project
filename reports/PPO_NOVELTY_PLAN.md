# PPO improvement 1: episodic observation counts

Branch: experiment/ppo-novelty-reward, based on Nathan2 commit e542c3c.
Protocol fixed on 23 October 2026, before improved training results.

## Student-project scope and attribution

This builds on our existing custom PPO. Its CNN, clipped objective, GAE, optimizer,
learning rate, rollout length and update epochs are unchanged. Nathan2 supplies the
shared environment, training/evaluation scripts, and DQN improvements. Nathan's
DQN networks, first-visit novelty reward and no-change memory remain unchanged.
His stochastic-evaluation option is reused; we add explicit per-episode Torch
seeding after checkpoint construction and save evaluation settings/checkpoint hashes.

Our additional PPO code is one small Gym wrapper, rl_agents/ppo/novelty.py.
It counts exact packed bytes of the newest visible grid (not hidden engine state):

    count[next_screen] += 1
    bonus = min(0.01 / sqrt(count[next_screen]), remaining_episode_bonus)
    training_reward = original_reward + bonus

The total bonus is capped at 0.5 per episode, below a single level-completion reward
of 1. The initial screen starts with count 1. Counts persist across automatic and
explicit resets inside an episode, and clear only at the next episode boundary.
The wrapper never runs during evaluation. Baseline PPO checkpoints remain compatible.
This is different from Nathan's one-time-per-screen DQN bonus. No no-change masking
is enabled for this PPO experiment, so only one intervention is being tested.

Students should be able to explain the wrapper above, why PPO receives the combined
reward during GAE, and why evaluation must measure task completion rather than the
combined training return. Code and experiment preparation were AI-assisted; do not
claim unassisted authorship or algorithmic novelty. Group contributions must reflect
actual work performed.

## Motivation and limitations

The frozen GPU baseline solved no levels in greedy evaluation, despite occasional
training success. A diagnostic first samples the existing learned PPO distribution,
because training sampled actions whereas prior evaluation used argmax. This diagnostic
is not counted as an algorithm improvement and requires no retraining.

Hypothesis: count-based rewards may encourage exploration beyond repetitive screens
when completion rewards are sparse. The method can reward irrelevant animation and
still gives a diminishing bonus on repeated screens; the cap limits, but does not
eliminate, this weakness. Higher novelty reward or screen diversity is not proof of
better task performance. A null or negative result is valid evidence.

## Fixed experiment

- Games: ft09, dc22, g50t, m0r0 only. No held-out games or game scan.
- Three independent training seeds: 0, 1, 2.
- Same 50,000 requested training steps, 1,000 actions per episode and GPU as baseline.
- Runner finishes the final episode; record actual interaction counts.
- Beta 0.01 and cap 0.5 fixed before results, with no sweep.
- Primary evaluation: sample policy; secondary: greedy policy. Report both, never
  compare improved sampled results only against baseline greedy results.
- Twenty episodes per checkpoint/mode. Policy-sampling seeds 100 through 119.
  These seed Torch sampling, not ARC engine layouts.
- Log combined return, original shaped/extrinsic return and novelty reward separately.
  None of these is an official ARC score.
- Baseline sampled evaluation runs first, then 12 fresh novelty runs with sampled
  and greedy evaluation after each. Existing frozen baseline greedy CSVs are retained.
- The baseline without the wrapper is the no-bonus ablation. No second improvement
  is bundled into this first one.

## Run and inspect

    .\.venv\Scripts\python.exe scripts/run_ppo_novelty.py

Outputs: runs/ppo_novelty_20261023. protocol.json records every command; status.json
records the current job and completed jobs. Logs, checkpoints and episode CSVs stay
under runs/ and are not committed. A failed partial training job restarts from scratch;
this simple runner does not claim exact optimizer-state resumption. Completed jobs
have marker files and can be skipped on rerun with the same unchanged protocol.

The earlier GPU baseline remains in runs/ppo_baseline_cuda_20261013. The DQN LFS
results archive was not needed for implementing or selecting this PPO variant.
