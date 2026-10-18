"""Screen games with the random policy to see where a learning signal exists.

    python scripts/scan_games.py --unused --episodes 10 --max-actions 1000 --out-dir scan

Why: on 3 of the 4 development games both baselines complete nothing, so any
tuning or improvement can only be judged on ft09. This script finds unused
public games where random play occasionally completes a level, so they can be
added to the DEVELOPMENT set (never the held-out set). Selecting games by random
performance is a choice you must disclose: log every game added in SPLIT.md with
the date and the scan results. Held-out games (cd82, tr87, vc33) are excluded.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baseline import make_env  # noqa: E402

DEV = ["dc22", "ft09", "g50t", "m0r0"]
HELD_OUT = ["cd82", "tr87", "vc33"]
ALL_PUBLIC = ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 "
              "sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30").split()


def scan(arc, args):
    games = args.games.split(",") if args.games else [g for g in ALL_PUBLIC if g not in DEV + HELD_OUT]
    bad = [g for g in games if g in HELD_OUT]
    if bad:
        raise SystemExit(f"Refusing to scan held-out games: {bad}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fields = ["game", "episode", "actions_used", "levels_completed", "noop_frac",
              "unique_frames", "n_legal_actions", "wall_s"]
    t0 = time.time()
    with open(out / "scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for game in games:
            for ep in range(args.episodes):
                env = make_env(arc, game, args, args.seed + ep)
                obs, info = env.reset(seed=args.seed + ep)
                n_legal = int(info["action_mask"].sum())
                seen, noops, steps, done = {hash(obs[-16:].tobytes())}, 0, 0, False
                while not done:
                    a = int(rng.choice(np.flatnonzero(info["action_mask"])))
                    prev = obs
                    obs, _, term, trunc, info = env.step(a)
                    noops += int(np.array_equal(obs[-16:], prev[-16:]))
                    seen.add(hash(obs[-16:].tobytes()))
                    steps += 1
                    done = term or trunc
                w.writerow({"game": game, "episode": ep, "actions_used": info["actions_used"],
                            "levels_completed": info["levels_completed"],
                            "noop_frac": round(noops / max(steps, 1), 3), "unique_frames": len(seen),
                            "n_legal_actions": n_legal, "wall_s": round(time.time() - t0, 1)})
                f.flush()
    print(f"Wrote {out / 'scan.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="", help="comma-separated ids; default = all games not in dev/held-out")
    p.add_argument("--unused", action="store_true", help="(default behaviour) scan the unused public games")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-actions", type=int, default=1000)
    p.add_argument("--click-grid", type=int, default=8)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--out-dir", default="scan")
    args = p.parse_args()
    import arc_agi
    from arc_agi import OperationMode
    scan(arc_agi.Arcade(operation_mode=OperationMode.NORMAL), args)
