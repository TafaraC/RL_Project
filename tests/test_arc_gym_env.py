"""Unit tests for ArcAgi3GymEnv against a MOCK arc_env.

These do not touch the network or the real ARC-AGI-3 game engine (this
sandbox can't reach three.arcprize.org). They only prove the adapter's
wiring is correct: action encode/decode, observation stacking, masking,
reward shaping, and the GAME_OVER-soft-reset / WIN-terminates logic.

Once you have API access (locally via the Kaggle starter's `make setup`,
or on Kaggle itself), swap MockArcEnv for the real
`arc_agi.Arcade().make(game_id)` object -- ArcAgi3GymEnv's code doesn't
change at all, because it only ever calls `.step(action, data=, reasoning=)`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from arcengine import GameAction, GameState

from environment.arc_gym_env import ArcAgi3GymEnv, GRID_SIZE, N_COLORS


class FakeRaw:
    """Stands in for arcengine.FrameDataRaw."""

    def __init__(self, grid_value, state, levels_completed, available_actions):
        self.game_id = "mockgame"
        self.frame = [np.full((GRID_SIZE, GRID_SIZE), grid_value, dtype=np.int64)]
        self.state = state
        self.levels_completed = levels_completed
        self.win_levels = 1
        self.guid = "g"
        self.full_reset = False
        self.available_actions = available_actions


class MockArcEnv:
    """Scripted fake game: RESET -> NOT_FINISHED, then GAME_OVER once,
    then eventually WIN, to exercise every branch of step()."""

    def __init__(self):
        self.n_calls = 0
        # ids: RESET=0, ACTION1..5=1..5, ACTION6=6, ACTION7=7
        self.all_ids = [a.value for a in GameAction]

    def step(self, action, data=None, reasoning=None):
        self.n_calls += 1
        if action is GameAction.RESET:
            return FakeRaw(0, GameState.NOT_FINISHED, 0, self.all_ids)
        if self.n_calls == 2:
            # First real action -> die once, to exercise the soft-reset path.
            return FakeRaw(1, GameState.GAME_OVER, 0, self.all_ids)
        if self.n_calls >= 5:
            return FakeRaw(2, GameState.WIN, 1, self.all_ids)
        return FakeRaw(1, GameState.NOT_FINISHED, 0, self.all_ids)


def test_reset_shape_and_mask():
    env = ArcAgi3GymEnv(MockArcEnv(), click_grid=4, history_len=3, max_actions=20)
    obs, info = env.reset()
    assert obs.shape == (3 * N_COLORS, GRID_SIZE, GRID_SIZE)
    assert obs.dtype == np.float32
    assert info["action_mask"].shape == (env.n_actions,)
    assert info["action_mask"].all()  # mock reports every action legal


def test_action_decode_simple_and_complex():
    env = ArcAgi3GymEnv(MockArcEnv(), click_grid=4)
    a, data = env._decode(0)
    assert a.is_simple() and data == {}
    # first click cell -> ACTION6 with a pixel inside the 64x64 canvas
    a, data = env._decode(env.n_simple)
    assert a is GameAction.ACTION6
    assert 0 <= data["x"] < GRID_SIZE and 0 <= data["y"] < GRID_SIZE


def test_game_over_soft_resets_not_episode_end():
    env = ArcAgi3GymEnv(MockArcEnv(), click_grid=2, max_actions=20)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # ACTION1 -> scripted GAME_OVER
    assert terminated is False
    assert truncated is False
    assert reward < 0  # game_over_penalty + action_cost, no level bonus
    assert info["raw_state"] is GameState.NOT_FINISHED  # env auto-RESET after the death


def test_win_terminates():
    env = ArcAgi3GymEnv(MockArcEnv(), click_grid=2, max_actions=20)
    env.reset()
    obs, r, term, trunc, info = env.step(1)  # -> GAME_OVER, soft reset (call #2)
    for _ in range(10):
        obs, r, term, trunc, info = env.step(1)
        if term:
            break
    assert term is True
    assert info["raw_state"] is GameState.WIN
    assert r > 0  # win_bonus dominates


def test_truncation_on_action_budget():
    env = ArcAgi3GymEnv(MockArcEnv(), click_grid=2, max_actions=3)
    env.reset()
    trunc = False
    for _ in range(5):
        obs, r, term, trunc, info = env.step(1)
        if trunc or term:
            break
    assert trunc is True


if __name__ == "__main__":
    test_reset_shape_and_mask()
    test_action_decode_simple_and_complex()
    test_game_over_soft_resets_not_episode_end()
    test_win_terminates()
    test_truncation_on_action_budget()
    print("All adapter wiring tests passed.")
