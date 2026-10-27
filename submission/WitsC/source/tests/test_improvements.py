"""Tests for the two optional improvements (novelty bonus, no-change memory).
No network, no torch: mock ARC engine only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from arcengine import GameAction, GameState

from environment.arc_gym_env import ArcAgi3GymEnv, NoopMemory, GRID_SIZE
from tests.test_arc_gym_env import FakeRaw


class ScriptedArc:
    """grid_values[i] is the colour shown after the i-th call (call 0 = RESET)."""

    def __init__(self, grid_values):
        self.values, self.n = list(grid_values), 0
        self.ids = [a.value for a in GameAction]

    def step(self, action, data=None, reasoning=None):
        v = self.values[min(self.n, len(self.values) - 1)]
        self.n += 1
        return FakeRaw(v, GameState.NOT_FINISHED, 0, self.ids)


def make(values, **kw):
    return ArcAgi3GymEnv(ScriptedArc(values), click_grid=1, history_len=1, max_actions=50, **kw)


def test_flags_off_matches_baseline_reward_and_mask():
    env = make([0, 1, 1, 2])
    env.reset()
    _, r, *_ = env.step(1)
    assert abs(r - (-0.002)) < 1e-9
    assert env.step(1)[4]["action_mask"].all()


def test_novelty_bonus_first_visit_only():
    env = make([0, 1, 1, 2], novelty_bonus=0.01)
    env.reset()
    _, r1, *_ = env.step(1)   # new screen 1 -> bonus
    _, r2, *_ = env.step(1)   # same screen again -> no bonus
    _, r3, *_ = env.step(1)   # new screen 2 -> bonus
    assert abs(r1 - (-0.002 + 0.01)) < 1e-9
    assert abs(r2 - (-0.002)) < 1e-9
    assert abs(r3 - (-0.002 + 0.01)) < 1e-9


def test_novelty_bonus_is_capped_per_episode():
    env = make(list(range(16)), novelty_bonus=0.1, novelty_cap=0.25)
    env.reset()
    total = 0.0
    for _ in range(10):
        _, r, _, _, info = env.step(1)
        total += r + 0.002
    assert abs(total - 0.25) < 1e-9
    assert abs(info["novelty_bonus_total"] - 0.25) < 1e-9


def test_novelty_counts_reset_each_episode():
    env = make([0, 1, 0, 1], novelty_bonus=0.01)
    env.reset()
    _, r, *_ = env.step(1)
    assert r > 0
    env.arc_env.n = 0
    env.reset()
    _, r, *_ = env.step(1)
    assert r > 0  # screen 1 is novel again in the new episode


def test_noop_masking_removes_repeated_noop_then_falls_back():
    env = make([0] * 40, noop_masking=True)  # screen never changes
    _, info = env.reset()
    n = env.n_actions
    seen_masks = []
    for a in range(n):
        assert info["action_mask"][a], "untried action must stay legal"
        _, _, term, trunc, info = env.step(a)
        seen_masks.append(int(info["action_mask"].sum()))
        if term or trunc:
            break
    assert seen_masks[0] == n - 1          # the action just tried is now masked
    assert seen_masks[-1] == n             # everything tried -> fallback to full mask


def test_noop_masking_ignores_actions_that_changed_the_screen():
    env = make([0, 1, 2, 3], noop_masking=True)
    env.reset()
    _, _, _, _, info = env.step(1)
    assert info["action_mask"].all()


def test_noop_memory_apply_never_returns_empty_mask():
    m = NoopMemory()
    mask = np.array([True, True, False])
    m.record(7, 0)
    m.record(7, 1)
    assert m.apply(mask, 7).tolist() == mask.tolist()
    assert m.apply(mask, 99).tolist() == mask.tolist()
    m.clear()
    m.record(7, 0)
    assert m.apply(mask, 7).tolist() == [False, True, False]




# ---- end-to-end checks that the flags are actually wired into the scripts ----
def test_cli_flags_exist_in_scripts():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    expected = {
        "run_baseline.py": ["--novelty-bonus", "--novelty-cap", "--noop-masking", "--buffer-capacity"],
        "eval_checkpoint.py": ["--noop-masking", "--stochastic", "--eval-epsilon"],
        "scan_games.py": ["--episodes", "--games"],
    }
    for script, flags in expected.items():
        out = subprocess.run([sys.executable, str(root / "scripts" / script), "--help"],
                             capture_output=True, text=True, cwd=root).stdout
        for flag in flags:
            assert flag in out, f"{script} is missing {flag}"


def test_runner_logs_novelty_bonus():
    import csv
    import tempfile
    from argparse import Namespace
    from scripts.run_baseline import run

    class Arcade:
        def make(self, *a, **k):
            return ScriptedArc(list(range(16)))

    def run_once(bonus):
        with tempfile.TemporaryDirectory() as d:
            args = Namespace(agent="random", out_dir=d, seed=0, games="mock", episodes=1, env_steps=0,
                             device="cpu", click_grid=1, history_len=1, max_actions=10,
                             novelty_bonus=bonus, novelty_cap=0.5, noop_masking=False)
            run(Arcade(), args)
            return float(next(csv.DictReader(open(Path(d) / "random_seed0" / "episodes.csv")))["novelty_bonus_total"])

    assert run_once(0.0) == 0.0
    assert run_once(0.05) > 0.0


def test_submission_agent_noop_memory_breaks_the_loop():
    import types
    from rl_agents.policy_agent import make_policy_agent_class
    mod = types.ModuleType("agents.agent")

    class Agent:
        def __init__(self, *a, **k):
            pass

    mod.Agent = Agent
    saved = {k: sys.modules.get(k) for k in ("agents", "agents.agent")}
    sys.modules["agents"], sys.modules["agents.agent"] = types.ModuleType("agents"), mod
    try:
        class Policy:  # always takes the first legal action: a stand-in for a stuck policy
            def __init__(self, *a, **k):
                pass

            def load(self, path):
                pass

            def act(self, obs, mask, greedy):
                return int(np.flatnonzero(mask)[0])

        env = ArcAgi3GymEnv(ScriptedArc([0] * 10), click_grid=1, history_len=1, max_actions=20)
        env.reset()
        frame = env._last_frame
        plain = make_policy_agent_class(Policy, "x", click_grid=1, history_len=1)()
        assert plain.choose_action([], frame) is plain.choose_action([], frame)   # stuck
        masked = make_policy_agent_class(Policy, "x", click_grid=1, history_len=1, noop_masking=True)()
        first, second = masked.choose_action([], frame), masked.choose_action([], frame)
        assert first is not second                                                 # loop broken
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All improvement tests passed.")
