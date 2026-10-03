"""Run a baseline (random reference policy or DQN) on ARC-AGI-3 games and log
per-episode results to CSV.

    python scripts/run_baseline.py --agent random --games ls20,vc33 --episodes 20 --seed 0
    python scripts/run_baseline.py --agent dqn    --games ls20      --env-steps 50000 --seed 0

Needs network access to the ARC-AGI-3 engine (run locally after `make setup`,
or on Kaggle). The CSV records SHAPED training return for diagnosing learning;
it is NOT the ARC-AGI-3 score. Report official scores via evaluation/rhae.py
with real human baselines, and say whether they are local or Kaggle scores.

Assumption to verify on first run: calling env.reset() repeatedly on one
arc_agi environment object starts a fresh attempt. If your arc_agi version
disagrees, build a new env with arc.make(game_id) per episode instead.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.arc_gym_env import ArcAgi3GymEnv  # noqa: E402


def make_env(arc, game_id, args, seed):
    return ArcAgi3GymEnv(
        arc.make(game_id, render_mode=None), game_id=game_id,
        click_grid=args.click_grid, history_len=args.history_len,
        max_actions=args.max_actions, seed=seed,
    )


def run(arc, args):
    out = Path(args.out_dir) / f"{args.agent}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    games = args.games.split(",")
    agent = None
    log_path = out / "episodes.csv"
    fields = ["game", "episode", "env_steps_total", "actions_used", "levels_completed",
              "won", "shaped_return", "epsilon", "wall_s"]
    total_steps, t0 = 0, time.time()
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        ep = 0
        while True:
            if args.agent == "random" and ep >= args.episodes * len(games):
                break
            if args.agent == "dqn" and total_steps >= args.env_steps:
                break
            game = games[ep % len(games)]
            env = make_env(arc, game, args, args.seed + ep)
            if args.agent == "dqn" and agent is None:
                import torch  # noqa: F401  (only needed for the learning baseline)
                from rl_agents.dqn.dqn_agent import DQNAgent, DQNConfig
                agent = DQNAgent(env.observation_space.shape, env.n_actions,
                                 DQNConfig(seed=args.seed, device=args.device))
            obs, info = env.reset(seed=args.seed + ep)
            ret, done = 0.0, False
            while not done:
                mask = info["action_mask"]
                if args.agent == "random":
                    a = int(rng.choice(np.flatnonzero(mask)))
                else:
                    a = agent.act(obs, mask)
                nobs, r, term, trunc, ninfo = env.step(a)
                if agent is not None:
                    # `done` for bootstrapping = true termination only (WIN), not truncation
                    agent.observe(obs, a, r, nobs, term, mask, ninfo["action_mask"])
                    agent.maybe_train()
                obs, info, ret, done = nobs, ninfo, ret + r, term or trunc
                total_steps += 1
            w.writerow({
                "game": game, "episode": ep, "env_steps_total": total_steps,
                "actions_used": info["actions_used"], "levels_completed": info["levels_completed"],
                "won": int(info["raw_state"].name == "WIN"), "shaped_return": round(ret, 4),
                "epsilon": round(agent.epsilon(), 4) if agent else "",
                "wall_s": round(time.time() - t0, 1),
            })
            f.flush()
            ep += 1
    if agent is not None:
        agent.save(str(out / "dqn.pt"))
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["random", "dqn"], required=True)
    p.add_argument("--games", required=True, help="comma-separated short game ids, e.g. ls20,vc33")
    p.add_argument("--episodes", type=int, default=20, help="random agent: episodes per game")
    p.add_argument("--env-steps", type=int, default=50_000, help="dqn: total environment steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-actions", type=int, default=80)
    p.add_argument("--click-grid", type=int, default=8)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()
    import arc_agi
    from arc_agi import OperationMode
    run(arc_agi.Arcade(operation_mode=OperationMode.NORMAL), args)
