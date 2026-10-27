# WitsC — Moodle submission package

Course: COMS4061A/COMS7071A. Deadline: **27 October 2026, 23:59 SAST**.

## What to submit

Upload `WitsC.zip` to Moodle. It contains this folder, the final report, runnable notebooks, the required trained-model dataset, source code, experiment evidence and contribution statements. No API credentials or private game sources are included.

## Nominated leaderboard notebook

**`kaggle/dqn_baseline.ipynb`** is the nominated entry: DQN, ft09-trained seed 1, epsilon 0.3, 8-by-8 click lattice, frozen weights, 1,000 actions after the initial reset. Selection used development results, not held-out outcomes. Its coarse lattice has a known limitation on ft09 level 2; the package does not claim complete-game solving.

**`kaggle/ppo_novelty.ipynb`** is the principal PPO alternative. Other notebooks reproduce the corresponding DQN/PPO stages and combined DQN ablation for inspection; they are not additional nominations. Each small notebook contains its inference source and loads its selected weights from the supplied model dataset. Do not upload a local fake score file.

## For the lecturer: executing the nominated notebook

1. If authorised to submit on WitsC's behalf, create a Kaggle dataset by uploading **`kaggle/model_dataset.zip`**. Preserve its extracted folder structure, including `witsc_models_manifest.json` and the named model folders. Then open ARC Prize 2026 ARC-AGI-3 and create/import `kaggle/dqn_baseline.ipynb`.
2. Attach **both** the official ARC-AGI-3 competition dataset **and** the WitsC model dataset just created. Set accelerator to **T4 GPU**, internet **OFF**, and use the competition's supported Python environment. Do not install this project's Windows requirements/wheels. If Kaggle preserves the upload as a ZIP rather than extracting it, extract it before uploading; the notebook must see the marker and model files under `/kaggle/input`.
3. Save & Run All. The notebook checks its embedded payload, loads the actual official Agent interface, performs CUDA model inference and prints the runtime versions. Any missing dependency or CUDA error is fatal.
4. When the notebook run succeeds, choose **Submit to Competition**, selecting **`submission.parquet`**. The commit-mode parquet is the official starter's placeholder, not an actual score. The competition rerun connects to the gateway, plays its games and lets the gateway produce the scored file.
5. Preserve the notebook URL, submission ID, timestamp and official score. The supplied `COMPETITION_EVIDENCE.json` intentionally has null values for evidence that does not yet exist.

## Dependency and validation boundaries

Local training used Windows PyTorch 2.14.1+cu130; the competition uses Linux and its own installed PyTorch. We export float32 NumPy weights without pickle and use ordinary Conv2d/ReLU/Linear operations. The notebook does not depend on Gymnasium or install local training packages. It installs only `arc-agi` and `python-dotenv` from the competition's offline wheel directory; NumPy, PyTorch, pandas and parquet support must be provided by the Kaggle image.

Local checks cover nine models: exact exported-network parity, frame/action-memory parity, resets, full official Agent.main loops on the four development games using CUDA, and actual commit-mode parquet writing/reading for the nominee with local paths substituted. **The offline wheel installation, Kaggle Linux image and live competition gateway have not been executed here.** This package is ready for the lecturer's submission attempt, not evidence of a successful official score.

## Contents

- `report/WitsC_Report.pdf`: course report in RLC 2026 style, no cover page; source and figures alongside it.
- `kaggle/`: nine small notebooks, the complete `model_dataset.zip` to attach, and selection/hash manifest.
- `source/`: project implementations, tests and reproduction scripts, with package guidance.
- `evidence/`: original evaluation CSVs, training logs/configurations, cost tables and final transfer results.
- `paper_track/WRITEUP.md`: condensed write-up under 1,500 words, saved for Paper Track publication.
- `validation/`: actual local test evidence and explicit unverified items.
- `CONTRIBUTIONS.md`, `COMPETITION_EVIDENCE.json`, `SHA256SUMS.txt`: attribution, honest submission status and integrity manifest.

The lecturer's feedback permits delivering the full package through Moodle and leaves proxy-scoring permission to the lecturer. If permitted, an invalid package may receive zero leaderboard marks; if not permitted, the lecturer will remove that component and rescale. There is no fabricated submission identifier or score in this archive. Paper Track publication remains an account-level action; no publication URL has been supplied.
