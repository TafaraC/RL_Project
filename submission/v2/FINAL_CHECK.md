# Final submission check — 27 October 2026

The current deliverable is `WitsC.zip` in this directory, built from `WitsC/`. The report source is in `report_source/`.

| Assignment requirement | Check |
|---|---|
| RLC 2026 format, no cover, at most eight main pages | Eight pages including references; unchanged style, fonts and margins. All pages visually checked. |
| Four authors and student emails | Nathan Le Roux 2717495, Tafara Chibanda 2683459, Samuel Snyman 2677730 and Joshua Snyman 2677722 appear on page 1. |
| Two learning baselines and learning objectives | DQN and PPO, with update rules, network, reward, resets and hyperparameters in Sections 2–3. |
| Two substantive changes for each algorithm | Novelty and no-change memory for DQN; novelty and action-history input for PPO. Combined versions and ablations are evaluated. |
| Random reference, three independent seeds, variability | Table 1 gives all three seeds and sample SD. Three trained seeds are distinct from repeated evaluation episodes. |
| Fair comparison and budgets | Games, seeds, action limits and hardware recorded. Historical DQN uses ten stochastic rollouts outside ft09, versus twenty for PPO; report and evidence guide now disclose this. DQN evaluation epsilon was selected on development behaviour. |
| Learning curves, costs, success/failure evidence | Two figures, completion and RHAE tables, training/evaluation costs and trajectory diagnosis in the report. Consolidated episode tables supplied. |
| Held-out environments | 270 episodes, six stages, three training seeds, five rollouts per checkpoint/game. Combined DQN was not tested on held-out games; this limits the final-agent comparison. |
| Local ARC score and scoring context | 258 RHAE rows reproduced exactly from supplied logs and baselines. Game versions identified. Counts omit initial RESET, so these are local reconstructions, not official gateway scores. |
| Reuse, AI assistance, contributions | References and contributions included. Development source inspection disclosed; no hidden state is a policy input. |
| Notebook, model files and dependencies | All nine notebooks and weights match the previously validated exports. CUDA and offline Kaggle execution instructions supplied; Windows training requirements are not installed in Kaggle. |
| Paper Track text under 1,500 words | Saved text included; publication still pending. |
| Official submission ID, timestamp, score and Paper Track URL | Missing; stated in the README. Local tests do not satisfy these external evidence requirements. |

Verification: 49 packaged-source tests passed in 16.77 seconds. Human baselines match the downloaded game metadata. All 3,183 development and 270 held-out episode rows use the stated 1,000-action budget and have consistent completion records. The original nine-model CUDA validation remains applicable because notebook and weight bytes are unchanged. Kaggle Linux dependency installation and live gateway execution remain unverified.

The core implementation and analysis are present, with the experimental limitations above. This check does not claim that all competition participation or official-scoring requirements have been fulfilled.
