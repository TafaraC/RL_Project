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
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.arc_gym_env import ArcAgi3GymEnv  # noqa: E402


def make_env(arc, game_id, args, seed):
    raw_env = arc.make(game_id, render_mode=None)
    if raw_env is None:
        raise RuntimeError(f"Could not load {game_id}; check cached files or network access")
    env = ArcAgi3GymEnv(
        raw_env, game_id=game_id,
        click_grid=args.click_grid, history_len=args.history_len,
        max_actions=args.max_actions, seed=seed,
        novelty_bonus=getattr(args, "novelty_bonus", 0.0),
        novelty_cap=getattr(args, "novelty_cap", 0.5),
        noop_masking=getattr(args, "noop_masking", False),
    )
    if getattr(args, 'ppo_novelty_bonus', 0.0) > 0:
        if args.agent != 'ppo' or getattr(args, 'novelty_bonus', 0.0) != 0 or getattr(args, 'noop_masking', False):
            raise ValueError('PPO count novelty must be used alone with PPO')
        from rl_agents.ppo.novelty import EpisodicNovelty
        env = EpisodicNovelty(env, args.ppo_novelty_bonus, getattr(args, 'ppo_novelty_cap', .5))
    return env


def run(arc, args):
    if args.device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; install requirements-cuda.txt')
    out = Path(args.out_dir) / f"{args.agent}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    import platform
    import importlib.metadata
    runtime = {'python': platform.python_version(), 'device': args.device,
               'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'arcengine', 'arc-agi')}}
    if args.device.startswith('cuda'):
        runtime['gpu'] = torch.cuda.get_device_name()
        runtime['cuda_runtime'] = torch.version.cuda
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    rng = np.random.default_rng(args.seed)
    games = args.games.split(",")
    agent = None
    log_path = out / "episodes.csv"
    fields = ["game", "episode", "env_steps_total", "actions_used", "levels_completed",
              "won", "shaped_return", "epsilon", "wall_s", "level_completion_actions", "novelty_bonus_total", "extrinsic_return"]
    total_steps, t0 = 0, time.time()
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        ep = 0
        while True:
            if args.agent == "random" and ep >= args.episodes * len(games):
                break
            if args.agent != "random" and total_steps >= args.env_steps:
                break
            game = games[ep % len(games)]
            env = make_env(arc, game, args, args.seed + ep)
            if args.agent == "dqn" and agent is None:
                import torch  # noqa: F401  (only needed for the learning baseline)
                from rl_agents.dqn.dqn_agent import DQNAgent, DQNConfig
                cfg_kwargs = {"seed": args.seed, "device": args.device}
                if getattr(args, "buffer_capacity", None) is not None:
                    cfg_kwargs["buffer_capacity"] = args.buffer_capacity
                agent = DQNAgent(env.observation_space.shape, env.n_actions, DQNConfig(**cfg_kwargs))
            if args.agent == "ppo" and agent is None:
                from rl_agents.ppo.ppo_agent import PPOAgent, PPOConfig
                agent = PPOAgent(env.observation_space.shape, env.n_actions,
                                 PPOConfig(seed=args.seed, device=args.device))
            obs, info = env.reset(seed=args.seed + ep)
            ret, done = 0.0, False
            prev_levels, level_actions = info["levels_completed"], []
            while not done:
                mask = info["action_mask"]
                if args.agent == "random":
                    a = int(rng.choice(np.flatnonzero(mask)))
                else:
                    a = agent.act(obs, mask)
                nobs, r, term, trunc, ninfo = env.step(a)
                if agent is not None:
                    # `done` for bootstrapping = true termination only (WIN), not truncation
                    agent.observe(obs, a, r, nobs, term, mask, ninfo["action_mask"],
                                  **({"truncated": trunc} if args.agent == "ppo" else {}))
                    metrics = agent.maybe_train()
                    if metrics is not None and args.agent == 'ppo':
                        with (out / 'updates.jsonl').open('a') as updates:
                            updates.write(json.dumps({'env_steps': total_steps + 1, **metrics}) + chr(10))
                if ninfo["levels_completed"] > prev_levels:
                    # cumulative actions_used when each new level was completed
                    # (needed later to compute per-level RHAE efficiency)
                    level_actions += [ninfo["actions_used"]] * (ninfo["levels_completed"] - prev_levels)
                    prev_levels = ninfo["levels_completed"]
                obs, info, ret, done = nobs, ninfo, ret + r, term or trunc
                total_steps += 1
            w.writerow({
                "game": game, "episode": ep, "env_steps_total": total_steps,
                "actions_used": info["actions_used"], "levels_completed": info["levels_completed"],
                "won": int(info["raw_state"].name == "WIN"), "shaped_return": round(ret, 4),
                "epsilon": round(agent.epsilon(), 4) if args.agent == "dqn" else "",
                "wall_s": round(time.time() - t0, 1),
                "level_completion_actions": ";".join(map(str, level_actions)),
                "novelty_bonus_total": round(info.get("novelty_bonus_total", 0.0), 4),
                "extrinsic_return": round(ret - info.get("novelty_bonus_total", 0.0), 4),
            })
            f.flush()
            ep += 1
    if agent is not None:
        if args.agent == "ppo":
            agent.maybe_train(force=True)
        agent.save(str(out / f"{args.agent}.pt"))
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["random", "dqn", "ppo"], required=True)
    p.add_argument("--games", required=True, help="comma-separated short game ids, e.g. ls20,vc33")
    p.add_argument("--episodes", type=int, default=20, help="random agent: episodes per game")
    p.add_argument("--env-steps", type=int, default=50_000, help="dqn: total environment steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-actions", type=int, default=80)
    p.add_argument("--click-grid", type=int, default=8)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--ppo-novelty-bonus", type=float, default=0.0, help="PPO only: count-based observation bonus (0 = off)")
    p.add_argument("--ppo-novelty-cap", type=float, default=0.5)
    p.add_argument("--novelty-bonus", type=float, default=0.0,
                   help="Improvement 1: reward for the first visit to each screen per episode (0 = off)")
    p.add_argument("--novelty-cap", type=float, default=0.5,
                   help="max total novelty reward per episode")
    p.add_argument("--noop-masking", action="store_true",
                   help="Improvement 2: mask actions already known to change nothing on this screen")
    p.add_argument("--buffer-capacity", type=int, default=None,
                   help="DQN replay buffer size in transitions (default: DQNConfig default, 10,000). "
                        "50,000 needs about 3 GiB of RAM with bit-packed observations")
    p.add_argument("--device", default="cuda")
    p.add_argument("--offline", action="store_true", help="Use cached games without API requests")
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()
    import arc_agi
    from arc_agi import OperationMode
    run(arc_agi.Arcade(operation_mode=OperationMode.OFFLINE if args.offline else OperationMode.NORMAL), args)
