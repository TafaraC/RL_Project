"""Gymnasium adapter for ARC-AGI-3.

The competition's own interface is NOT a Gymnasium environment: agents
subclass `agents.agent.Agent` (from the ARC-AGI-3-Agents framework) and
implement `choose_action(frames, latest_frame) -> GameAction` /
`is_done(frames, latest_frame) -> bool`. Standard RL libraries (Stable
Baselines3, CleanRL, our own DQN loop) expect `gymnasium.Env`'s
`reset()/step()` contract instead, so this module is the adapter the
assignment brief asks for ("If your RL library requires Gymnasium,
implement an adapter and document its semantics").

This file documents every non-obvious design decision inline, because the
report needs to justify them, not just state them.

WHAT THIS WRAPS
----------------
We wrap an `arcengine.EnvironmentWrapper`-like object that exposes
`.step(action, data=..., reasoning=...) -> FrameDataRaw` and comes from
`arc_agi.Arcade().make(game_id, ...)`. That object talks to the real
ARC-AGI-3 game engine (locally in dev, or via the Kaggle gateway at
submission time). This wrapper never itself talks to the network -- it
only calls whatever `arc_env` you hand it, so it works unmodified whether
`arc_env` is a live game, a local one, or (see tests/) a mock used for
unit testing the wiring without any network access.

MDP DESIGN DECISIONS (put these in the report's Environment section)
----------------------------------------------------------------------
1. Observation: `FrameData.frame` is a list of 64x64 int grids (values
   0-15, i.e. 16 colours). We one-hot each grid to (16, 64, 64) and stack
   the last `history_len` *post-action* observations along the channel
   axis, giving shape (history_len * 16, 64, 64). This is the same
   "frame stacking" trick classic Atari DQN uses to give a feed-forward
   CNN a short memory window over an otherwise partially-observable,
   rule-undiscovered environment -- ARC-AGI-3 gives no explicit game
   description, so recent history is the only cheap way to disambiguate
   e.g. "did that last click do anything".
2. Action space: GameAction has 6 simple actions (RESET, ACTION1-5,
   ACTION7) and one complex action (ACTION6, which additionally needs an
   (x, y) point on the 64x64 canvas). Exposing all 4096 (x, y) pairs as
   separate discrete actions would make exploration extremely hard for a
   tabular-epsilon-greedy DQN baseline, so we discretise ACTION6's click
   target onto a coarser `click_grid x click_grid` lattice (default 8x8 =
   64 cells, each action clicking that cell's centre pixel). This is a
   documented action abstraction, allowed under the brief's "action
   abstractions ... compatible with the competition rules" clause, and is
   itself a natural ablation: compare click_grid=8 vs 16 vs 64 later.
   Total action space size = 6 + click_grid**2 (default 6+64 = 70).
3. Action masking: `FrameData.available_actions` lists the GameAction ids
   that are legal in the current state. We expose this every step as
   `info["action_mask"]`, a boolean vector over the full action space
   (all click cells share ACTION6's legality). Baselines are free to
   ignore it; masked epsilon-greedy / masked policy heads are one of the
   documented "substantive improvements" this scaffolding leaves room
   for.
4. Episode boundary vs. soft reset: the starter kit's own reference agent
   treats GAME_OVER as "issue RESET and keep going" and only treats WIN
   as done -- because a fixed per-game action budget (MAX_ACTIONS) is
   spent whether you die once or five times, and RHAE scoring cares about
   total actions used to reach WIN, not about "episodes". We mirror that:
     - GAME_OVER  -> we auto-submit RESET internally, return
                     terminated=False, and let the reward term below
                     penalise it. From the RL algorithm's point of view
                     this is one long MDP with soft resets, not a new
                     episode.
     - WIN        -> terminated=True.
     - action budget (`max_actions`) reached -> truncated=True.
   This is a deliberate, reportable choice -- an alternative (treating
   GAME_OVER as a hard episode end) is a one-line change and worth an
   ablation if time allows.
5. Reward shaping (TRAINING SIGNAL ONLY): ARC-AGI-3 has no explicit
   reward channel. We shape:
       +1.0  per additional level completed this step
       +5.0  bonus the step WIN is reached
       -0.05 GAME_OVER (discourage dying)
       -0.002 per action taken (mild pressure toward action-efficiency,
              in the same spirit as RHAE's efficiency term, but this is
              NOT the official score)
   Per the brief ("Do not present shaped training return as the
   ARC-AGI-3 score"), the evaluation harness (evaluation/rhae.py) must
   compute and report the real RHAE score separately from whatever
   return curves you plot during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This adapter requires gymnasium: pip install gymnasium"
    ) from e

# Real types from the installed `arcengine` package (see agents/agent.py
# in ARC-AGI-3-Agents for how the official framework uses them).
from arcengine import FrameData, GameAction, GameState

N_COLORS = 16
GRID_SIZE = 64

# GameAction ids that don't need extra (x, y) data. Public (not
# underscore-prefixed) because agents/policy_agent.py needs the same
# encode/decode logic at inference time.
SIMPLE_ACTIONS = [a for a in GameAction if a.is_simple()]
COMPLEX_ACTIONS = [a for a in GameAction if a.is_complex()]
assert len(COMPLEX_ACTIONS) == 1 and COMPLEX_ACTIONS[0] is GameAction.ACTION6, (
    "This adapter assumes ACTION6 is the only complex (pointer) action; "
    "re-check against the installed arcengine version if this fires."
)
_SIMPLE_ACTIONS = SIMPLE_ACTIONS  # backwards-compat alias used internally below


@dataclass
class RewardConfig:
    level_bonus: float = 1.0
    win_bonus: float = 5.0
    game_over_penalty: float = 0.05
    action_cost: float = 0.002


class ArcAgi3GymEnv(gym.Env):
    """gymnasium.Env wrapper around one ARC-AGI-3 game.

    Parameters
    ----------
    arc_env:
        The object returned by `arc_agi.Arcade().make(game_id, ...)`.
        Must expose `.step(action, data=dict, reasoning=dict|None) ->
        FrameDataRaw` -- this matches `agents.agent.Agent.do_action_request`
        in the official framework, so anything that plugs into the
        official `Agent` class plugs in here too.
    game_id:
        Short game id (e.g. "ls20"), used only for logging/seeding.
    click_grid:
        Side length of the coarse lattice ACTION6 clicks are snapped to.
    history_len:
        Number of most recent one-hot grids stacked as the observation.
    max_actions:
        Per-episode action budget (mirrors `Agent.MAX_ACTIONS`); use the
        same number you'll submit with, so training-time efficiency
        pressure matches evaluation conditions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        arc_env: Any,
        game_id: str = "unknown",
        click_grid: int = 8,
        history_len: int = 4,
        max_actions: int = 80,
        reward_config: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not 1 <= click_grid <= GRID_SIZE or history_len < 1 or max_actions < 1:
            raise ValueError("Invalid click grid, history length, or action budget")
        self._episode_done = False
        self.arc_env = arc_env
        self.game_id = game_id
        self.click_grid = click_grid
        self.history_len = history_len
        self.max_actions = max_actions
        self.reward_config = reward_config or RewardConfig()
        self._rng = np.random.default_rng(seed)

        self.n_simple = len(_SIMPLE_ACTIONS)
        self.n_click_cells = click_grid * click_grid
        self.n_actions = self.n_simple + self.n_click_cells
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.history_len * N_COLORS, GRID_SIZE, GRID_SIZE),
            dtype=np.float32,
        )

        self._history: list[np.ndarray] = []
        self._action_counter = 0
        self._last_levels_completed = 0
        self._last_frame: Optional[FrameData] = None

    # ------------------------------------------------------------------
    # gymnasium.Env API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        frame = self._request(GameAction.RESET)
        self._episode_done = False
        self._action_counter = 0
        self._last_levels_completed = frame.levels_completed
        self._last_frame = frame
        self._history = []
        obs = self._push_and_stack(frame)
        info = self._info(frame)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._last_frame is None:
            raise RuntimeError("Call reset() before step().")

        if self._episode_done:
            raise RuntimeError("Episode ended; call reset() before step()")
        game_action, action_data = self._decode(action)
        if game_action is GameAction.RESET:
            self._history = []
        frame = self._request(game_action, action_data)
        self._action_counter += 1

        reward = self._reward(frame)
        terminated = frame.state is GameState.WIN
        truncated = self._action_counter >= self.max_actions

        if frame.state is GameState.GAME_OVER and not terminated and not truncated:
            # Soft reset: mirrors the starter-kit reference agent, which
            # RESETs on GAME_OVER and keeps consuming the same action
            # budget rather than ending the episode. See module docstring
            # point 4.
            frame = self._request(GameAction.RESET)
            self._action_counter += 1
            reward -= self.reward_config.action_cost
            self._history = []
            terminated = frame.state is GameState.WIN
            truncated = self._action_counter >= self.max_actions

        self._last_levels_completed = frame.levels_completed
        self._last_frame = frame
        obs = self._push_and_stack(frame)
        info = self._info(frame)
        self._episode_done = terminated or truncated
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _request(self, game_action: GameAction, data: Optional[dict] = None) -> FrameData:
        raw = self.arc_env.step(game_action, data=data or {}, reasoning=None)
        return FrameData(
            game_id=getattr(raw, "game_id", self.game_id),
            frame=[np.asarray(g).tolist() for g in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )

    def _decode(self, action: int) -> tuple[GameAction, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if action < self.n_simple:
            return _SIMPLE_ACTIONS[action], {}
        cell = action - self.n_simple
        cx, cy = cell % self.click_grid, cell // self.click_grid
        # Snap to the cell's centre pixel in the 64x64 canvas.
        px = int((cx + 0.5) * GRID_SIZE / self.click_grid)
        py = int((cy + 0.5) * GRID_SIZE / self.click_grid)
        px, py = min(px, GRID_SIZE - 1), min(py, GRID_SIZE - 1)
        return GameAction.ACTION6, {"x": px, "y": py}

    def _one_hot(self, grid: list[list[int]]) -> np.ndarray:
        arr = np.asarray(grid, dtype=np.int64)
        oh = np.zeros((N_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        rows, cols = np.indices(arr.shape)
        oh[arr, rows, cols] = 1.0
        return oh

    def _push_and_stack(self, frame: FrameData) -> np.ndarray:
        latest = frame.frame[-1] if frame.frame else [[0] * GRID_SIZE] * GRID_SIZE
        self._history.append(self._one_hot(latest))
        if len(self._history) > self.history_len:
            self._history.pop(0)
        pad = self.history_len - len(self._history)
        stacked = ([np.zeros((N_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)] * pad
                   + self._history)
        return np.concatenate(stacked, axis=0)

    def _reward(self, frame: FrameData) -> float:
        rc = self.reward_config
        r = -rc.action_cost
        delta_levels = frame.levels_completed - self._last_levels_completed
        if delta_levels > 0:
            r += rc.level_bonus * delta_levels
        if frame.state is GameState.WIN:
            r += rc.win_bonus
        if frame.state is GameState.GAME_OVER:
            r -= rc.game_over_penalty
        return float(r)

    def _info(self, frame: FrameData) -> dict:
        mask = np.zeros(self.n_actions, dtype=bool)
        for legal_id in frame.available_actions:
            legal_action = GameAction.from_id(legal_id) if hasattr(GameAction, "from_id") else None
            if legal_action in _SIMPLE_ACTIONS:
                mask[_SIMPLE_ACTIONS.index(legal_action)] = True
            elif legal_action is GameAction.ACTION6:
                mask[self.n_simple:] = True
        return {
            "action_mask": mask,
            "actions_used": self._action_counter,
            "levels_completed": frame.levels_completed,
            "win_levels": frame.win_levels,
            "game_id": self.game_id,
            "raw_state": frame.state,
        }
