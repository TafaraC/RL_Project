# WitsC submission packaging status

Updated 27 October 2026. Active branch: `codex/kaggle-moodle-package`.

## Objective

Produce a Moodle-ready folder with an offline ARC-AGI-3 Kaggle notebook, model assets, course report, evidence and lecturer instructions. The lecturer may submit on the team's behalf. Preserve progress on GitHub at meaningful checkpoints.

## Verified starting point

- PPO development evidence is published on main at 6b9f9af.
- Official starter builder retrieved to `runs/kaggle_reference/build_notebook.py`.
- Official framework retrieved to `runs/kaggle_reference/ARC-AGI-3-Agents-main`.
- Official flow: install competition-provided wheels, register Agent, use gateway during competition rerun; ordinary Save & Run All emits the starter placeholder parquet. A successful ordinary run is not an official score.
- No official Kaggle execution or score has been obtained.
- Local training uses Python 3.14; competition guidance specifies Python 3.12. Package must avoid local Windows wheels.

## Completed checkpoints

- Eight self-contained notebooks: five PPO conditions and three DQN stages. Lossless NumPy weights embedded; readable duplicate exports in `kaggle/assets/` for audit, excluded from Moodle to avoid duplicate weight storage.
- Local CUDA checks passed: embedded archive integrity, notebook syntax, exact original/exported network parity, memory observation parity, resets and the official Agent.main loop for 1,000 actions on each development game (32 model/game checks).
- Full DQN archive recovered from Git LFS and verified via ZIP extraction. Original DQN evaluations and all training-cost data exported.
- Frozen held-out run complete: 18 jobs, 270 episodes, three training seeds per stage, five rollouts per checkpoint/game. Only DQN baseline seed 1 clears a tr87 level once; no game wins.
- RLC 2026 report source and initial seven-page PDF built; final held-out paragraph and author emails added, awaiting final compile/visual check.
- Nomination fixed from development evidence: DQN baseline ft09 seed 1, epsilon 0.3. PPO alternative: novelty ft09 seed 2. No selection based on held-out outcomes.

## Remaining

Verify Save & Run All output locally with explicit platform substitutions; finish updated Paper Track text, lecturer instructions and Moodle archive; compile/render final report; audit package and push. Actual Kaggle Linux runtime/gateway execution and external publication are not available locally and must not be called verified. No official score/entry ID or Paper Track publication URL is available.
