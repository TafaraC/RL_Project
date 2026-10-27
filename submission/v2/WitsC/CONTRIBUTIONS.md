# Contributions

| Member | Student email | Contribution |
|---|---|---|
| Nathan Le Roux | 2717495@students.wits.ac.za | DQN implementation and experiments |
| Tafara Chibanda | 2683459@students.wits.ac.za | PPO development, experimental iteration and submission integration |
| Samuel Snyman | 2677730@students.wits.ac.za | Setup and documentation |
| Joshua Snyman | 2677722@students.wits.ac.za | Setup and documentation |

## AI assistance

Claude assisted with the initial code scaffold. Codex assisted with implementation, debugging, experiment analysis, packaging and drafting of the report. All authors are responsible for the submitted code, results and claims.

## Reused resources

- PyTorch, NumPy and Gymnasium.
- `arc-agi`, `arcengine` and the official ARC-AGI-3 Kaggle starter and Agent framework.
- The RLC 2026 report style.

No external pretrained model or policy is used. Both learning agents and all improvements were implemented by the team.

## Use of game information

Development-game source and state were inspected to diagnose agent failures (report, Section 6). Agents never use this information. They act only on public frames, legal actions and their own past actions. No held-out game source or solution was inspected.
