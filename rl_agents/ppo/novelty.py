"""PPO improvement 1: episodic counts of visible screens, training only."""
from collections import Counter
import math
import gymnasium as gym
import numpy as np


class EpisodicNovelty(gym.Wrapper):
    """Add beta/sqrt(N(screen)) to training reward, capped per episode.

    Only the newest visible grid is used, never engine state. Counts persist
    through automatic/explicit in-episode resets and reset only on env.reset().
    This wrapper is not used at evaluation or submission time.
    """
    def __init__(self, env, beta=.01, cap=.5):
        super().__init__(env)
        if not math.isfinite(beta) or not math.isfinite(cap) or beta < 0 or cap < 0:
            raise ValueError('Novelty beta and cap must be finite and nonnegative')
        self.n_actions = env.action_space.n
        self.beta, self.cap = beta, cap
        self.counts = Counter()
        self.total = 0.

    @staticmethod
    def key(obs):
        # Exact packed bytes avoid hash collisions and ignore stacked history.
        return np.packbits(obs[-16:].astype(np.uint8)).tobytes()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.counts = Counter({self.key(obs): 1})
        self.total = 0.
        return obs, {**info, 'novelty_bonus_total': 0., 'intrinsic_reward': 0.}

    def step(self, action):
        obs, extrinsic, terminated, truncated, info = self.env.step(action)
        key = self.key(obs)
        self.counts[key] += 1
        intrinsic = min(self.beta / math.sqrt(self.counts[key]), max(0., self.cap-self.total))
        self.total += intrinsic
        return obs, extrinsic + intrinsic, terminated, truncated, {
            **info, 'extrinsic_reward': extrinsic, 'intrinsic_reward': intrinsic,
            'novelty_bonus_total': self.total, 'unique_observations': len(self.counts)}
