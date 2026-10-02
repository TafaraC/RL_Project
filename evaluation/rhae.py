"""Official ARC-AGI-3 scoring (RHAE), verified against
docs.arcprize.org/methodology.md (fetched 27 Sep 2026 -- re-check before
your final report in case it has since changed):

  * Per-level score: RHAE_l = min(1.15, (human_baseline_l / ai_actions_l)^2)
    -- capped at 1.15x human baseline, not 1.0, so an agent that finds a
    shortcut can score slightly above "human parity".
  * Per-game score: weighted average of per-level scores using the
    1-indexed level number as weight (later, harder levels count more).
    Unsolved levels contribute 0 to the numerator but their weight still
    counts in the denominator -- so a game score cannot reach 100%
    without clearing every level, regardless of how efficient the solved
    levels were.
        RHAE(g) = sum_{l in solved} l * RHAE_l  /  sum_{l=1..L_g} l
  * Headline score: unweighted mean of RHAE(g) over all games evaluated.

This module does NOT know the real human-baseline action counts -- those
are published by the ARC Prize Foundation per game/level and must be
pulled from the competition once you have API access (this sandbox
can't reach three.arcprize.org). Treat `human_baselines` below as
something you fill in from the real data, not something to invent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rhae_level_score(human_baseline: int, ai_actions: int) -> float:
    if ai_actions <= 0:
        return 0.0
    return min(1.15, (human_baseline / ai_actions) ** 2)


@dataclass
class LevelRecord:
    level_index: int  # 1-indexed
    actions_used: int
    human_baseline: int
    solved: bool

    @property
    def score(self) -> float:
        return rhae_level_score(self.human_baseline, self.actions_used) if self.solved else 0.0


def rhae_game_score(levels: list[LevelRecord], total_levels: int) -> float:
    """`total_levels` (L_g) may exceed len(levels) if the agent never even
    reached later levels -- those still count in the denominator."""
    denom = total_levels * (total_levels + 1) / 2  # sum_{l=1..L_g} l
    if denom == 0:
        return 0.0
    numer = sum(rec.level_index * rec.score for rec in levels if rec.solved)
    return numer / denom


def extract_level_records(frames: list, win_levels: int,
                           human_baselines: dict[int, int]) -> list[LevelRecord]:
    """Walk an Agent's `.frames` history (list[FrameData], one entry per
    action taken, as populated by `agents.agent.Agent.append_frame`) and
    slice it into per-level action counts.

    `human_baselines` maps 1-indexed level number -> human baseline action
    count for this game; missing levels fall back to the largest known
    baseline (a conservative placeholder -- replace with real data).
    """
    records: list[LevelRecord] = []
    prev_levels_completed = 0
    prev_boundary_idx = 0
    fallback_baseline = max(human_baselines.values()) if human_baselines else 1

    for i, frame in enumerate(frames):
        lc = getattr(frame, "levels_completed", 0)
        if lc > prev_levels_completed:
            for level_idx in range(prev_levels_completed + 1, lc + 1):
                actions_used = i - prev_boundary_idx
                records.append(LevelRecord(
                    level_index=level_idx,
                    actions_used=max(actions_used, 1),
                    human_baseline=human_baselines.get(level_idx, fallback_baseline),
                    solved=True,
                ))
            prev_boundary_idx = i
            prev_levels_completed = lc
    return records


@dataclass
class GameEvalResult:
    game_id: str
    seed: int
    levels: list[LevelRecord]
    total_levels: int
    actions_used: int
    won: bool

    @property
    def rhae(self) -> float:
        return rhae_game_score(self.levels, self.total_levels)


@dataclass
class EvaluationReport:
    per_run: list[GameEvalResult] = field(default_factory=list)

    def add(self, result: GameEvalResult) -> None:
        self.per_run.append(result)

    def summary_by_game(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for g in sorted({r.game_id for r in self.per_run}):
            runs = [r for r in self.per_run if r.game_id == g]
            scores = [r.rhae for r in runs]
            out[g] = {
                "n_seeds": len(runs),
                "rhae_mean": float(np.mean(scores)),
                "rhae_std": float(np.std(scores)),
                "win_rate": float(np.mean([r.won for r in runs])),
                "mean_actions_used": float(np.mean([r.actions_used for r in runs])),
            }
        return out

    def overall(self) -> dict:
        per_game = self.summary_by_game()
        if not per_game:
            return {}
        return {
            "n_games": len(per_game),
            "rhae_mean_over_games": float(np.mean([v["rhae_mean"] for v in per_game.values()])),
            "win_rate_over_games": float(np.mean([v["win_rate"] for v in per_game.values()])),
        }


def evaluate_policy(agent_cls, arc_arcade, game_ids: list[str], seeds: list[int],
                     human_baselines: dict[str, dict[int, int]], max_actions: int = 80,
                     **agent_kwargs) -> EvaluationReport:
    """Runs `agent_cls` (an `agents.agent.Agent` subclass) on every
    (game, seed) pair, scores it with the real RHAE methodology above, and
    returns a report. Needs network access to the ARC-AGI-3 engine -- run
    this in your local dev setup or on Kaggle, not in this sandbox.
    `human_baselines`: {game_id: {level_index: human_action_count}}.
    """
    report = EvaluationReport()
    for game_id in game_ids:
        baselines = human_baselines.get(game_id, {})
        for seed in seeds:
            env = arc_arcade.make(game_id, render_mode=None)
            agent = agent_cls(
                card_id="eval", game_id=game_id, agent_name=f"eval.{game_id}.{seed}",
                ROOT_URL="http://localhost", record=False, arc_env=env, tags=["eval"],
                **agent_kwargs,
            )
            agent.MAX_ACTIONS = max_actions
            agent.main()
            final = agent.frames[-1]
            total_levels = getattr(final, "win_levels", final.levels_completed) or 1
            levels = extract_level_records(agent.frames, total_levels, baselines)
            report.add(GameEvalResult(
                game_id=game_id, seed=seed, levels=levels, total_levels=total_levels,
                actions_used=agent.action_counter, won=final.state.name == "WIN",
            ))
    return report
