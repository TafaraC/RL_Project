"""Bridges a trained policy back into the *official* submission interface.

Training happens against `ArcAgi3GymEnv` (Gymnasium API) because that's
what standard RL code expects. But the actual Kaggle submission -- and
the starter kit's `agent/my_agent.py` -- must implement the competition's
own `Agent` interface (`is_done` / `choose_action` operating on
`list[FrameData]`), not gymnasium's. This file is the thin adapter back
the other way, so `agent/my_agent.py` on Kaggle can just be:

    from rl_agents.policy_agent import make_policy_agent_class
    MyAgent = make_policy_agent_class(DQNAgent, "checkpoints/dqn_ls20.pt",
                                       click_grid=8, history_len=4)

Note: this project's own package is named `rl_agents` (not `agents`) on
purpose -- the ARC-AGI-3-Agents framework already owns a top-level
`agents` package (`agents.agent.Agent`, imported below), and both end up
on `sys.path` together once this is wired into the Kaggle starter kit.
Reusing the name would shadow one or the other depending on import order.

Keeping this separate from ArcAgi3GymEnv means the same trained checkpoint
can be evaluated three ways with identical action-selection code: inside
the Gym loop during training-time evaluation, inside this Agent wrapper
for local `play_local.py` runs, and inside this same wrapper when it ships
to Kaggle -- so "local score" and "Kaggle score" differences trace back to
the environment, not to two different inference implementations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from environment.arc_gym_env import (GRID_SIZE, N_COLORS, SIMPLE_ACTIONS as _SIMPLE_ACTIONS,
                                     NoopMemory, frame_key)


def make_policy_agent_class(agent_cls, checkpoint_path: str, click_grid: int = 8,
                             history_len: int = 4, device: str = "cpu",
                             noop_masking: bool = False, ppo_action_memory: bool = False,
                             stochastic: bool = False, max_actions: int = 80):
    """Returns an `agents.agent.Agent` subclass driven by a trained policy.

    `agent_cls` is e.g. `rl_agents.dqn.dqn_agent.DQNAgent` -- anything exposing
    `.act(obs, action_mask, greedy=True) -> int` and `.load(path)`.
    """
    from agents.agent import Agent  # official framework, only needed at submission time

    n_actions = len(_SIMPLE_ACTIONS) + click_grid * click_grid
    obs_shape = (history_len * N_COLORS, GRID_SIZE, GRID_SIZE)
    if ppo_action_memory:
        from rl_agents.ppo.action_memory import ActionMemory, MEMORY_CHANNELS
        obs_shape = (obs_shape[0] + MEMORY_CHANNELS, GRID_SIZE, GRID_SIZE)
        if noop_masking:
            raise ValueError('Do not combine action memory and no-op masking')

    class PolicyAgent(Agent):
        MAX_ACTIONS = max_actions

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._history: list[np.ndarray] = []
            # Improvement 2: per-game-attempt memory of actions that changed nothing on a screen.
            self._noop = NoopMemory()
            self._prev_key = None
            self._prev_action = None
            self._action_memory = ActionMemory(click_grid) if ppo_action_memory else None
            self._memory_grid = None
            self._memory_action = None
            policy_kwargs = {'action_memory': True} if ppo_action_memory else {}
            self._policy = agent_cls(obs_shape, n_actions, device=device, inference_only=True, **policy_kwargs)
            self._policy.load(checkpoint_path)

        def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
            return latest_frame.state is GameState.WIN

        def _one_hot(self, grid) -> np.ndarray:
            arr = np.asarray(grid, dtype=np.int64)
            oh = np.zeros((N_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
            rows, cols = np.indices(arr.shape)
            oh[arr, rows, cols] = 1.0
            return oh

        def _obs(self, frame: FrameData) -> np.ndarray:
            latest = frame.frame[-1] if frame.frame else [[0] * GRID_SIZE] * GRID_SIZE
            self._history.append(self._one_hot(latest))
            if len(self._history) > history_len:
                self._history.pop(0)
            pad = history_len - len(self._history)
            stacked = ([np.zeros((N_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)] * pad
                       + self._history)
            return np.concatenate(stacked, axis=0)

        def _mask(self, frame: FrameData) -> np.ndarray:
            mask = np.zeros(n_actions, dtype=bool)
            n_simple = len(_SIMPLE_ACTIONS)
            for legal_id in frame.available_actions:
                legal = GameAction.from_id(legal_id)
                if legal in _SIMPLE_ACTIONS:
                    mask[_SIMPLE_ACTIONS.index(legal)] = True
                elif legal is GameAction.ACTION6:
                    mask[n_simple:] = True
            return mask

        def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self._history = []
                self._prev_key = None
                if self._action_memory is not None:
                    self._action_memory.clear()
                    self._memory_grid = self._memory_action = None
                return GameAction.RESET

            obs = self._obs(latest_frame)
            if self._action_memory is not None:
                if self._memory_action is not None:
                    self._action_memory.update(self._memory_action, self._memory_grid, obs[-16:])
                self._memory_grid = obs[-16:].copy()
                obs = self._action_memory.augment(obs)
            mask = self._mask(latest_frame)
            if noop_masking:
                key = frame_key(latest_frame.frame[-1]) if latest_frame.frame else 0
                if self._prev_key is not None and key == self._prev_key:
                    self._noop.record(self._prev_key, self._prev_action)
                mask = self._noop.apply(mask, key)
            action_id = self._policy.act(obs, mask, greedy=not stochastic)
            self._memory_action = action_id
            if noop_masking:
                self._prev_key, self._prev_action = key, action_id
            if action_id < len(_SIMPLE_ACTIONS) and _SIMPLE_ACTIONS[action_id] is GameAction.RESET:
                self._history = []
                if self._action_memory is not None:
                    self._action_memory.clear()
                    self._memory_grid = self._memory_action = None

            n_simple = len(_SIMPLE_ACTIONS)
            if action_id < n_simple:
                return _SIMPLE_ACTIONS[action_id]
            cell = action_id - n_simple
            cx, cy = cell % click_grid, cell // click_grid
            px = min(int((cx + 0.5) * GRID_SIZE / click_grid), GRID_SIZE - 1)
            py = min(int((cy + 0.5) * GRID_SIZE / click_grid), GRID_SIZE - 1)
            action = GameAction.ACTION6
            action.set_data({"x": px, "y": py})
            return action

    return PolicyAgent
