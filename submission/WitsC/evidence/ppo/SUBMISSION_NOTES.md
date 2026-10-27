# WitsC Paper Track handoff

Prepared 26 October 2026. `WRITEUP.md` is the concise paper text; `EVIDENCE.md` and `verified_results.csv` preserve numerical provenance. No submission has been made.

## Submission route

On 27 October 2026, the team clarified that an external GitHub link is accepted for the Paper Track. Prepare the write-up using https://github.com/TafaraC/RL_Project as its code link. Do not treat the missing scored Kaggle entry as a blocker to preparing the paper or invent an entry ID, score or timestamp.

Verification note: the publicly accessible [Paper Prize page](https://arcprize.org/competitions/2026/paper), checked on 27 October, still states that a corresponding Kaggle code submission is required. The current Kaggle submission-form requirements could not be independently retrieved. The team's reported submission route and the public prize-eligibility wording therefore remain distinct; successful write-up publication should not itself be represented as confirmed prize eligibility.

Our frozen evaluations reuse development games on which agents trained. The write-up labels them accordingly. Held-out testing and official scoring remain absent, independently of how the code is linked.

## Information still needed for publication

- Kaggle account/handle, registered team identity and permitted submission route.
- External code link: https://github.com/TafaraC/RL_Project. Include an official score only if one actually exists.
- Paper Track write-up URL after publication.
- Final course report attachment/link: the concise write-up does not replace the separate RLC-format report.
- Team review of attribution, AI disclosure and source-informed diagnostics; student email addresses for the course report.

## Files and verification

- `WRITEUP.md`: copyable paper under the assignment's 1,500-word limit, including references.
- `EVIDENCE.md`: generated table with per-seed counts and scope limitations.
- `verified_results.csv`: one row per audited completed evaluation, with original source path.
- `ft09_diagnostic.png`: visual explanation from recorded gameplay, optional illustration.
- `python scripts/build_paper_evidence.py`: regenerate the evidence from local run files without launching training or evaluation.

Raw run files and checkpoints are local artifacts and may be ignored by Git. The repository link alone does not distribute those artifacts; attach the exported CSV and preserve raw logs/checkpoints when assembling the final course archive. Historical DQN/random figures were not promoted into a matched results table because their complete comparable evidence was unavailable.

## Contribution statement

Nathan Le Roux (2717495): DQN implementation and associated experiments.

Tafara Chibanda (2683459): PPO development coordination and experimental iteration with AI assistance.

Samuel Snyman (2677730) and Joshua Snyman (2677722): setup and documentation, as confirmed by the team.

AI assistance: Claude scaffold assistance and Codex implementation, debugging, experiment analysis and writing assistance. Team members should verify this description against their actual use before submission.
