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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All improvement tests passed.")
