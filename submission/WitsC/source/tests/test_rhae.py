import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.rhae import (
    rhae_level_score, rhae_game_score, extract_level_records, LevelRecord,
)


class FakeFrame:
    def __init__(self, levels_completed):
        self.levels_completed = levels_completed


def test_official_worked_examples():
    # From docs.arcprize.org/methodology.md
    assert abs(rhae_level_score(10, 10) - 1.0) < 1e-9
    assert abs(rhae_level_score(10, 20) - 0.25) < 1e-9
    assert abs(rhae_level_score(10, 100) - 0.01) < 1e-9


def test_cap_at_1_15_not_1_0():
    # AI is 4x more efficient than the human baseline -> ratio^2 = 16, must clip to 1.15
    assert rhae_level_score(40, 10) == 1.15


def test_unsolved_levels_cap_game_score_below_100_percent():
    # Game has 3 levels; only level 1 (weight 1) solved perfectly.
    levels = [LevelRecord(level_index=1, actions_used=10, human_baseline=10, solved=True)]
    score = rhae_game_score(levels, total_levels=3)
    # denom = 1+2+3 = 6, numer = 1*1.0 = 1.0 -> 1/6
    assert abs(score - (1 / 6)) < 1e-9


def test_extract_level_records_from_frame_history():
    # RESET frame (0 completed), 3 actions to clear level 1, 2 more to clear level 2.
    frames = [
        FakeFrame(0),  # after RESET
        FakeFrame(0), FakeFrame(0), FakeFrame(1),  # 3 actions -> level 1 done at idx 3
        FakeFrame(1), FakeFrame(2),                # 2 actions -> level 2 done at idx 5
    ]
    baselines = {1: 3, 2: 2}
    records = extract_level_records(frames, win_levels=2, human_baselines=baselines)
    assert len(records) == 2
    assert records[0].level_index == 1 and records[0].actions_used == 3
    assert records[1].level_index == 2 and records[1].actions_used == 2
    # Both levels matched human baseline exactly -> both level scores == 1.0
    game_score = rhae_game_score(records, total_levels=2)
    assert abs(game_score - 1.0) < 1e-9  # perfect, all levels solved at human parity


if __name__ == "__main__":
    test_official_worked_examples()
    test_cap_at_1_15_not_1_0()
    test_unsolved_levels_cap_game_score_below_100_percent()
    test_extract_level_records_from_frame_history()
    print("All RHAE tests passed.")
