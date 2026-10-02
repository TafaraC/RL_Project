"""DQN baseline (course-covered algorithm A).

Standard DQN (Mnih et al., 2015): CNN Q-network, target network updated
by periodic hard copies, uniform replay, epsilon-greedy exploration -- with
one ARC-AGI-3-specific addition: illegal actions (per
`info["action_mask"]` from ArcAgi3GymEnv) are excluded both when acting
and when bootstrapping the TD target, otherwise the agent wastes most of
its budget on actions the current game state doesn't support (e.g.
clicking when only RESET is legal).

This is intentionally a plain baseline -- no Double DQN, no dueling heads,
no prioritized replay, no n-step returns. Those are exactly the kind of
"substantive improvement" the assignment wants you to add *on top of*
this baseline, one at a time, each with its own before/after evaluation.
Adding several of them here would collapse the baseline -> improvement 1
-> improvement 2 comparison the rubric is graded on.
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 64
    buffer_capacity: int = 10_000
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_freq: int = 1_000  # hard update every N gradient steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    grad_clip: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class DQNAgent:
    """Owns the network, target network, optimizer and replay buffer, and
    exposes `act()` / `observe()` / `train_step()` so a plain Python loop
    can drive it against `ArcAgi3GymEnv`. Kept separate from that loop
    (see scripts/train_dqn.py) so the same class can later be reused for
    evaluation with epsilon=0.
    """

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, config: DQNConfig | None = None, *, device: str | None = None, inference_only: bool = False):
        config = copy.deepcopy(config) if config is not None else DQNConfig()
        if device is not None:
            config.device = device
        self.cfg = config
        self.n_actions = n_actions
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.device = torch.device(config.device)
        in_channels = obs_shape[0]
        self.q = QNetwork(in_channels, n_actions).to(self.device)
        self.target_q = copy.deepcopy(self.q).to(self.device)
        self.target_q.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=config.lr)
        self.buffer = None if inference_only else ReplayBuffer(config.buffer_capacity, obs_shape, n_actions, seed=config.seed)

        self._env_steps = 0
        self._grad_steps = 0

    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        frac = min(1.0, self._env_steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, action_mask: np.ndarray, greedy: bool = False) -> int:
        legal = np.flatnonzero(action_mask)
        if legal.size == 0:
            raise ValueError("No legal actions available")
        if not greedy and random.random() < self.epsilon():
            return int(np.random.choice(legal))

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        q = self.q(obs_t).squeeze(0).cpu().numpy()
        q_masked = np.where(action_mask, q, -np.inf)
        return int(np.argmax(q_masked))

    def observe(self, obs, action, reward, next_obs, done, action_mask, next_action_mask) -> None:
        if self.buffer is None:
            raise RuntimeError("Inference-only agent cannot observe training transitions")
        self.buffer.add(obs, action, reward, next_obs, done, action_mask, next_action_mask)
        self._env_steps += 1

    def maybe_train(self) -> dict | None:
        if self.buffer is None or len(self.buffer) < max(1, self.cfg.learning_starts):
            return None
        if self._env_steps % self.cfg.train_freq != 0:
            return None
        return self._train_step()

    def _train_step(self) -> dict:
        batch = self.buffer.sample(self.cfg.batch_size)
        obs = torch.from_numpy(batch.obs).to(self.device)
        next_obs = torch.from_numpy(batch.next_obs).to(self.device)
        action = torch.from_numpy(batch.action).long().to(self.device)
        reward = torch.from_numpy(batch.reward).to(self.device)
        done = torch.from_numpy(batch.done).float().to(self.device)
        next_mask = torch.from_numpy(batch.next_action_mask).to(self.device)

        q_values = self.q(obs).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_q(next_obs)
            next_q = next_q.masked_fill(~next_mask, float("-inf"))
            # If a next state somehow has no legal action recorded, fall
            # back to the unmasked max rather than propagating -inf/NaN.
            no_legal = ~next_mask.any(dim=1)
            if no_legal.any():
                next_q[no_legal] = self.target_q(next_obs)[no_legal]
            next_v = next_q.max(dim=1).values
            target = reward + self.cfg.gamma * (1.0 - done) * next_v

        loss = F.smooth_l1_loss(q_values, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self._grad_steps += 1

        if self._grad_steps % self.cfg.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return {"loss": loss.item(), "epsilon": self.epsilon(), "grad_steps": self._grad_steps}

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"q": self.q.state_dict(), "cfg": asdict(self.cfg)}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(ckpt["q"])
        self.target_q.load_state_dict(ckpt["q"])
