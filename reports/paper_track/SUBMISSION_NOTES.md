# WitsC Paper Track handoff

Updated 27 October 2026. `WRITEUP.md` is the concise paper text; `EVIDENCE.md` and `verified_results.csv` preserve PPO numerical provenance. Complete DQN and held-out evidence is in `submission/WitsC/evidence`. No external Paper Track submission has been made.

## Submission route

On 27 October 2026, the team clarified that an external GitHub link is accepted for the Paper Track. Prepare the write-up using https://github.com/TafaraC/RL_Project as its code link. Do not treat the missing scored Kaggle entry as a blocker to preparing the paper or invent an entry ID, score or timestamp.

Verification note: the publicly accessible [Paper Prize page](https://arcprize.org/competitions/2026/paper), checked on 27 October, still states that a corresponding Kaggle code submission is required. The current Kaggle submission-form requirements could not be independently retrieved. The team's reported submission route and the public prize-eligibility wording therefore remain distinct; successful write-up publication should not itself be represented as confirmed prize eligibility.

Development evaluations reuse games on which agents trained. A separate frozen held-out evaluation is now complete (270 episodes). No official score exists. The lecturer has requested the complete Kaggle package through Moodle and will determine whether proxy scoring is permitted; the final package records that route separately from Paper Track publication.

## Information still needed for publication

- Kaggle account/handle, registered team identity and permitted submission route.
- External code link: https://github.com/TafaraC/RL_Project. Include an official score only if one actually exists.
- Paper Track write-up URL after publication.
- Attach/link the course report supplied in the Moodle package.
- Team review of attribution, AI disclosure and source-informed diagnostics. Student emails use Wits' published student-number convention.

## Files and verification

- `WRITEUP.md`: copyable paper under the assignment's 1,500-word limit, including references.
- `EVIDENCE.md`: generated table with per-seed counts and scope limitations.
- `verified_results.csv`: one row per audited completed evaluation, with original source path.
- `ft09_diagnostic.png`: visual explanation from recorded gameplay, optional illustration.
- `python scripts/build_paper_evidence.py`: regenerate the evidence from local run files without launching training or evaluation.

Published notebooks now include a separate model dataset containing all required weights; attach it in Kaggle. Complete DQN archive recovery enabled the updated comparison; historical evaluation configuration gaps remain disclosed. Per-episode evaluation CSVs, training logs, configuration records and held-out evidence are included in the final package.

## Contribution statement

Nathan Le Roux (2717495): DQN implementation and associated experiments.

Tafara Chibanda (2683459): PPO development coordination and experimental iteration with AI assistance.

Samuel Snyman (2677730) and Joshua Snyman (2677722): setup and documentation, as confirmed by the team.

AI assistance: Claude scaffold assistance and Codex implementation, debugging, experiment analysis and writing assistance. Team members should verify this description against their actual use before submission.
