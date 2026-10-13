# PPO baseline validation — 13 October 2026

## Scope

Keep plain PPO as the second baseline. DQN outperforming PPO is acceptable;
no exploration, reward, architecture or hyperparameter improvements were made.
Evaluation remains deferred at the user's request.

## Failure diagnosis

Nine training runs finished and saved checkpoints. The m0r0 seed-0 run logged
48 episodes and 47,712 steps before an ARC metadata connection timed out.
Arcade.make returned None, causing an AttributeError on reset. This was an
infrastructure failure, not a PPO loss or CUDA failure. There is no saved
checkpoint for that interrupted run, so it must restart from seed 0.
The original CPU logs are retained in runs/ppo_baseline.

The runner now supports cached offline games, checks for failed game loading,
and defaults to explicit CUDA training. CUDA requests fail clearly if unavailable.
All twelve runs will restart on CUDA in runs/ppo_baseline_cuda_20261013,
keeping CPU results and the failed attempt separate. Comparing runtime across these devices requires
explicit hardware disclosure.

## Reference comparison

Fetched origin/main and origin/Nathan on 13 October. Neither contains committed
result CSVs. Nathan contains the DQN runner and evaluator. Reference results below
come from the supplied PROGRESS_UPDATE.md dated 4 October; they are not independently
reproduced here. PPO counts are computed from local training episode CSVs.

| Game | PPO episodes with a level, seeds 0 / 1 / 2 | DQN reference |
|---|---|---|
| ft09 | 3 / 3 / 7 (51 episodes per seed) | 38 / 33 / 32 (51 per seed) |
| dc22 | 0 / 0 / 0 | 0 across 153 episodes |
| g50t | 0 / 0 / 1 | 2 across 153 episodes |
| m0r0 | incomplete; seed 0 had 0 / 48 | 0 across 153 episodes |

PPO training samples its learned categorical policy; DQN uses epsilon-greedy
exploration. Training success counts therefore do not directly compare final
policy quality. The report's claim of no learning signal on three games should
be qualified: g50t has rare recorded successes in both implementations.
No official ARC score or final PPO evaluation is claimed.

## Functional checks

PPO uses fixed old log probabilities, the same legal-action masks during sampling
and updates, clipped surrogate loss, GAE, value regression and entropy regularization.
It zeroes terminal bootstrap values and stops GAE across episode boundaries while
bootstrapping time limits. Rollouts are discarded after each on-policy update.

Added a controlled two-action reward test: the probability of the rewarded action
must increase after PPO updates, with a third action masked out. This checks policy
learning rather than merely a change to value-network weights. Existing tests check
checkpoint round trips, finite updates, partial rollouts and runner integration.
Update diagnostics now record losses, entropy, approximate KL and clipping fraction;
these add observability without changing the training objective.

## GPU setup

Use the official PyTorch CUDA 13.0 wheel via requirements-cuda.txt:

    .\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt

Installation reference: https://pytorch.org/get-started/locally/
Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB, driver 610.62.

Run all twelve fresh GPU training jobs (no evaluation):

    .\.venv\Scripts\python.exe scripts/run_ppo_experiments.py --training-only --device cuda --games ft09,dc22,g50t,m0r0 --out-dir runs/ppo_baseline_cuda_20261013

Offline mode requires the cached game files already present on this machine.
Keep all existing CPU results as historical evidence; use the fresh all-GPU series
for consistent compute comparisons within PPO. Do not describe these checkpoints as exact
training-resumption files; they currently contain inference weights and config only.

Validation: PyTorch 2.14.1+cu130 detects the RTX 3050. All 20 tests passed, including
the CUDA policy-learning test, on 13 October 2026.
