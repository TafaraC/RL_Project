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
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.obs_shape = obs_shape
        self.capacity = capacity
        self.obs = [None] * capacity
        self.next_obs = [None] * capacity
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
        packed_obs = self._pack(obs)
        packed_next = self._pack(next_obs)
        self.obs[i] = packed_obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = packed_next
        self.done[i] = done
        self.action_mask[i] = action_mask
        self.next_action_mask[i] = next_action_mask
        self._idx += 1
        if self._idx >= self.capacity:
            self._idx = 0
            self._full = True

    def sample(self, batch_size: int) -> Batch:
        n = len(self)
        if n == 0 or batch_size <= 0:
            raise ValueError("sampling requires data and a positive batch size")
        idx = self._rng.integers(0, n, size=batch_size)
        return Batch(
            obs=self._unpack(self.obs, idx),
            action=self.action[idx],
            reward=self.reward[idx],
            next_obs=self._unpack(self.next_obs, idx),
            done=self.done[idx],
            action_mask=self.action_mask[idx],
            next_action_mask=self.next_action_mask[idx],
        )

    def _pack(self, obs):
        arr = np.asarray(obs)
        if arr.shape != self.obs_shape or not np.all((arr == 0) | (arr == 1)):
            raise ValueError("replay observations must be binary with the configured shape")
        return np.packbits(arr.reshape(-1).astype(np.uint8))

    def _unpack(self, storage, indices):
        packed = np.stack([storage[i] for i in indices])
        bits = np.unpackbits(packed, axis=1, count=int(np.prod(self.obs_shape)))
        return bits.reshape(len(indices), *self.obs_shape).astype(np.float32)
