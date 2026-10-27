## Latest deliverable: cleaned v2

The current submission is `submission/v2/WitsC.zip`, built from the user's simpler `v2/WitsC` layout. It supersedes the earlier package described below. The report uses the assignment heading and direct wording, retains v2 local RHAE results and is eight pages including references. Final checks: 49 packaged-source tests passed; 258 local RHAE rows reproduced; all notebook/model bytes match validated exports; all report pages visually reviewed. The package contains no checksum manifest. Internal archive checks are saved under ignored `runs/`. See `submission/v2/FINAL_CHECK.md` for requirement coverage and outstanding external submission evidence.

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

- Final packaging: nine small notebooks (five PPO, four DQN) with inference source embedded. Required weights are in `submission/WitsC/kaggle/model_dataset.zip`, to upload as a dataset and attach. This supersedes the oversized embedded-weight notebooks in early checkpoints. Each notebook is under 1 MB (about 8.5 KB) and verifies its required weight hash.
- Local CUDA checks passed: source archive integrity, attached model hashes, Python 3.12 syntax, exact original/exported network parity, memory observation parity, resets and the official Agent.main loop for 1,000 actions on each development game (36 model/game checks). Nominee commit-mode execution writes/reads the actual parquet schema under local path substitutions; offline pip and live Kaggle are explicitly untested.
- Full DQN archive recovered from Git LFS and verified via ZIP extraction. Original DQN evaluations and all training-cost data exported.
- Frozen held-out run complete: 18 jobs, 270 episodes, three training seeds per stage, five rollouts per checkpoint/game. Only DQN baseline seed 1 clears a tr87 level once; no game wins.
- Final seven-page RLC 2026 report compiled and all pages visually inspected, including held-out results, student emails and source/AI disclosures. Cover page omitted; references begin on page 7.
- Supplemental evaluation finished: 240 random episodes (ft09 successes 2,5,8 /20; other games zero) and 60 combined DQN episodes (19,8,18 /20). Original configurations and results exported.
- Nomination fixed from development evidence: DQN baseline ft09 seed 1, epsilon 0.3. PPO alternative: novelty ft09 seed 2. No selection based on held-out outcomes.

## Final local deliverable

`submission/WitsC.zip` is assembled (approximately 77 MB) and its 697 files pass ZIP CRC, all-file SHA256, notebook/weight hash, source-parity, Python 3.12 syntax and GPU/offline metadata checks. `submission/WitsC_PACKAGE_AUDIT.json` records the exact archive hash. The paper is 1,328 whitespace-delimited words. Project tests: 45 passed. The folder and rebuild scripts are saved in Git; the outer ZIP is reproducibly assembled locally to avoid another duplicate binary in Git.

Actual Kaggle Linux runtime/gateway execution and external publication are not available locally and must not be called verified. No official score/entry ID or Paper Track publication URL is available. `submission/WitsC/START_HERE.md` is the lecturer handoff; nominated notebook is `dqn_baseline.ipynb`, PPO alternative is `ppo_novelty.ipynb`. Both require the supplied model dataset to be uploaded and attached, plus official competition data. Final archive builder and auditor are in `scripts/`.
