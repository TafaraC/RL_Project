# WitsC PPO results

These are frozen-policy evaluations on four development games used for training, not official Kaggle scores or held-out results.

- [Verified results and per-seed comparison](EVIDENCE.md)
- [Machine-readable results with provenance](verified_results.csv)
- [Original evaluation CSV copies](evaluation_logs/)
- [Paper write-up](WRITEUP.md)
- [Visual diagnosis](ft09_diagnostic.png)
- [Submission notes](SUBMISSION_NOTES.md)

The exported evidence covers 108 evaluation files and 2,160 episodes. The five PPO conditions use three training seeds per game. Novelty on the original click grid achieved the highest pooled first-level completion on ft09 (28/60); none of the evaluated policies won a full game. See the evidence table for seed variability and unsuccessful improvements.

The CSV copies make the published aggregate independently inspectable without downloading checkpoints. Training logs and model weights are not included in this export. To rebuild the export from the original local runs, execute `python scripts/build_paper_evidence.py` from the repository root.
