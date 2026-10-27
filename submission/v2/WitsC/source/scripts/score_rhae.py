"""Score recorded evaluation episodes with the ARC-AGI-3 RHAE metric.

Uses the official methodology (docs.arcprize.org/methodology):
  level score = min(1.15, (human baseline actions / agent actions)^2)
  game score  = sum over solved levels of (level number x level score)
                divided by the sum of all level numbers (unsolved levels count as 0)
  overall     = mean of game scores

Human baselines are the per-level `baseline_actions` published in each game's
metadata.json. They are copied to evidence/human_baselines.csv. No game is replayed.

Usage (from the source folder):
  python scripts/score_rhae.py --baselines ../evidence/human_baselines.csv \
      --evidence ../evidence --out ../evidence/rhae_scores.csv
or read the baselines straight from downloaded games with --games-dir environment_files.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.rhae import LevelRecord, rhae_game_score  # noqa: E402


def load_baselines(games_dir: Path | None, table: Path | None) -> dict[str, list[int]]:
    """Map short game id (e.g. ft09) to its per-level human baseline action counts."""
    baselines = {}
    if games_dir:
        for meta in sorted(games_dir.glob("*/*/metadata.json")):
            data = json.loads(meta.read_text())
            baselines[data["game_id"].split("-")[0]] = [int(a) for a in data["baseline_actions"]]
    if table:
        levels = defaultdict(dict)
        with open(table, newline="") as f:
            for row in csv.DictReader(f):
                levels[row["game"]][int(row["level"])] = int(row["human_baseline_actions"])
        for game, by_level in levels.items():
            baselines[game] = [by_level[i] for i in sorted(by_level)]
    return baselines


def episode_score(row: dict, baseline: list[int]) -> float:
    """RHAE game score for one episode.

    `actions_at_level_completions` holds the cumulative action count at each level
    completion; the actions spent on level l are the difference from the previous one.
    """
    marks = [int(a) for a in row["actions_at_level_completions"].split(";") if a]
    records, previous = [], 0
    for level, cumulative in enumerate(marks, start=1):
        records.append(LevelRecord(level_index=level, actions_used=max(cumulative - previous, 1),
                                   human_baseline=baseline[level - 1], solved=True))
        previous = cumulative
    return rhae_game_score(records, total_levels=len(baseline))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games-dir", type=Path, help="folder holding <game>/<version>/metadata.json")
    p.add_argument("--baselines", type=Path, help="CSV of per-level human baselines")
    p.add_argument("--evidence", type=Path, required=True,
                   help="folder holding evaluation_episodes_*.csv")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not (args.games_dir or args.baselines):
        p.error("give --games-dir or --baselines")
    baselines = load_baselines(args.games_dir, args.baselines)
    scores = defaultdict(list)  # (split, condition, policy, game, seed) -> episode scores
    for split, name in [("development", "evaluation_episodes_development.csv"),
                        ("held-out", "evaluation_episodes_holdout.csv")]:
        with open(args.evidence / name, newline="") as f:
            for row in csv.DictReader(f):
                key = (split, row["condition"], row["eval_policy"], row["game"], row["training_seed"])
                scores[key].append(episode_score(row, baselines[row["game"]]))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "condition", "eval_policy", "game", "training_seed", "episodes",
                    "rhae_mean_percent", "rhae_max_percent", "levels_in_game"])
        for (split, cond, pol, game, seed), vals in scores.items():
            w.writerow([split, cond, pol, game, seed, len(vals),
                        round(100 * sum(vals) / len(vals), 4), round(100 * max(vals), 4),
                        len(baselines[game])])
    print(f"Wrote {len(scores)} rows to {args.out}")


if __name__ == "__main__":
    main()
