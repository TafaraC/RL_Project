"""Evaluate a saved DQN checkpoint with NO exploration and NO learning.

    python scripts/eval_checkpoint.py --checkpoint runs_final/dqn_seed0/dqn.pt \
        --games ft09 --episodes 20 --seed 0 --max-actions 1000 --out-dir eval_final --tag dqn_seed0_dev

Use the same --max-actions, --click-grid and --history-len the checkpoint was
trained with. Writes <out-dir>/<tag>/episodes.csv in the same format as
run_baseline.py, so random, training-time and greedy results can be compared
with one analysis script. The policy is greedy (epsilon = 0) and parameters
are frozen: nothing persists or adapts between episodes.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baseline import make_env  # noqa: E402


def evaluate(arc, args, agent=None):
    out = Path(args.out_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    games = args.games.split(",")
    fields = ["game", "episode", "actions_used", "levels_completed", "won",
              "level_completion_actions", "wall_s", "n_distinct_actions", "noop_frac",
              "top_action", "top_action_frac"]
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for game in games:
            for ep in range(args.episodes):
                env = make_env(arc, game, args, args.seed + ep)
                if agent is None:
                    from rl_agents.dqn.dqn_agent import DQNAgent, DQNConfig
                    agent = DQNAgent(env.observation_space.shape, env.n_actions,
                                     DQNConfig(device=args.device), inference_only=True)
                    agent.load(args.checkpoint)
                obs, info = env.reset(seed=args.seed + ep)
                done, prev, level_actions = False, info["levels_completed"], []
                counts, noops, steps = Counter(), 0, 0
                while not done:
                    if args.eval_epsilon > 0 and rng.random() < args.eval_epsilon:
                        a = int(rng.choice(np.flatnonzero(info["action_mask"])))
                    else:
                        a = agent.act(obs, info["action_mask"], greedy=True)
                    prev_obs = obs
                    obs, _, term, trunc, info = env.step(a)
                    counts[a] += 1
                    steps += 1
                    # newest grid = last 16 channels of the stacked observation
                    noops += int(np.array_equal(obs[-16:], prev_obs[-16:]))
                    if info["levels_completed"] > prev:
                        level_actions += [info["actions_used"]] * (info["levels_completed"] - prev)
                        prev = info["levels_completed"]
                    done = term or trunc
                w.writerow({"game": game, "episode": ep, "actions_used": info["actions_used"],
                            "levels_completed": info["levels_completed"],
                            "won": int(info["raw_state"].name == "WIN"),
                            "level_completion_actions": ";".join(map(str, level_actions)),
                            "wall_s": round(time.time() - t0, 1),
                            "n_distinct_actions": len(counts),
                            "noop_frac": round(noops / max(steps, 1), 3),
                            "top_action": counts.most_common(1)[0][0] if counts else "",
                            "top_action_frac": round(counts.most_common(1)[0][1] / max(steps, 1), 3) if counts else ""})
                f.flush()
    print(f"Wrote {out / 'episodes.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-actions", type=int, default=1000)
    p.add_argument("--click-grid", type=int, default=8)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="eval")
    p.add_argument("--eval-epsilon", type=float, default=0.0,
                   help="probability of a random legal action at evaluation time (0 = purely greedy)")
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    import arc_agi
    from arc_agi import OperationMode
    evaluate(arc_agi.Arcade(operation_mode=OperationMode.NORMAL), args)
