import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "score_rhae", Path(__file__).resolve().parents[1] / "scripts" / "score_rhae.py")
score_rhae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_rhae)

BASELINE = [43, 12, 23, 28, 65, 37]  # ft09: six levels, weights sum to 21


def test_unsolved_episode_scores_zero():
    assert score_rhae.episode_score({"actions_at_level_completions": ""}, BASELINE) == 0.0


def test_first_level_at_human_parity():
    row = {"actions_at_level_completions": "43"}
    assert score_rhae.episode_score(row, BASELINE) == pytest.approx(1 / 21)


def test_level_score_is_capped():
    row = {"actions_at_level_completions": "9"}
    assert score_rhae.episode_score(row, BASELINE) == pytest.approx(1.15 / 21)


def test_later_levels_use_actions_since_previous_completion():
    row = {"actions_at_level_completions": "86;110"}  # level 1 in 86, level 2 in 24
    expected = (1 * (43 / 86) ** 2 + 2 * (12 / 24) ** 2) / 21
    assert score_rhae.episode_score(row, BASELINE) == pytest.approx(expected)
