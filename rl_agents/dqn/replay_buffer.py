"""Fixed-size uniform replay buffer.

Kept deliberately simple for the baseline. Prioritized replay is a
reasonable "substantive improvement" candidate for the iterative-
improvement phase of the assignment (cite Schaul et al. 2016 if you use it)
-- don't build it into the baseline itself, or you lose the baseline->
improvement comparison the rubric wants.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], n_actions: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.action_mask = np.zeros((capacity, n_actions), dtype=np.bool_)
        self.next_action_mask = np.zeros((capacity, n_actions), dtype=np.bool_)
        self._idx = 0
        self._full = False
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.capacity if self._full else self._idx

    def add(self, obs, action, reward, next_obs, done, action_mask, next_action_mask) -> None:
        i = self._idx
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.action_mask[i] = action_mask
        self.next_action_mask[i] = next_action_mask
        self._idx += 1
        if self._idx >= self.capacity:
            self._idx = 0
            self._full = True

    def sample(self, batch_size: int) -> Batch:
        n = len(self)
        idx = self._rng.integers(0, n, size=batch_size)
        return Batch(
            obs=self.obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
            action_mask=self.action_mask[idx],
            next_action_mask=self.next_action_mask[idx],
        )
