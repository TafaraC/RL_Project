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

## Remaining

Implement portable inference and notebook builder; include checkpoint assets; test the actual official Agent loop locally; evaluate frozen selection on held-out games without tuning; assemble course report and complete archive; validate contents and publish progress. Record genuine blockers rather than calling untested Kaggle execution verified.
