"""Plain masked PPO: clipped surrogate, GAE, shared CNN actor/critic."""
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from rl_agents.dqn.q_network import QNetwork


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip_range: float = .2
    value_coef: float = .5
    entropy_coef: float = .01
    rollout_steps: int = 128
    batch_size: int = 32
    epochs: int = 4
    grad_clip: float = .5
    seed: int = 0
    device: str = 'cpu'


def compute_gae(rewards, values, next_values, terminated, boundaries, gamma, lam):
    """Bootstrap time limits, but never carry advantages across episode resets."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (1 - terminated[i]) * next_values[i] - values[i]
        carry = delta + gamma * lam * (1 - boundaries[i]) * carry
        advantages[i] = carry
    return advantages, advantages + np.asarray(values, dtype=np.float32)


class PPOAgent:
    def __init__(self, obs_shape, n_actions, config=None, *, device=None, inference_only=False, action_memory=False):
        self.cfg = PPOConfig(**asdict(config)) if config else PPOConfig()
        if device is not None:
            self.cfg.device = device
        if min(self.cfg.rollout_steps, self.cfg.batch_size, self.cfg.epochs) < 1:
            raise ValueError('Rollout, batch size and epochs must be positive')
        torch.manual_seed(self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        self.obs_shape, self.n_actions = tuple(int(dim) for dim in obs_shape), int(n_actions)
        from rl_agents.ppo.action_memory import SCHEMA
        self.observation_schema = SCHEMA if action_memory else 'pixels_v1'
        # Same convolutional architecture as DQN; last output is the state value.
        self.network = QNetwork(obs_shape[0], n_actions + 1).to(self.device)
        self.optimizer = None if inference_only else torch.optim.Adam(self.network.parameters(), lr=self.cfg.lr)
        self.rollout = []
        self.pending = None

    def _forward(self, obs, mask):
        output = self.network(obs)
        if not mask.any(dim=-1).all():
            raise ValueError('No legal actions available')
        return Categorical(logits=output[:, :-1].masked_fill(~mask, -torch.inf)), output[:, -1]

    @torch.no_grad()
    def act(self, obs, action_mask, greedy=False):
        dist, value = self._forward(torch.as_tensor(obs, device=self.device).unsqueeze(0),
                                    torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0))
        action = dist.probs.argmax(-1) if greedy else dist.sample()
        if self.optimizer is not None:
            self.pending = (int(action.item()), float(dist.log_prob(action).item()), float(value.item()))
        return int(action.item())

    @torch.no_grad()
    def observe(self, obs, action, reward, next_obs, terminated, action_mask, next_action_mask, *, truncated=False):
        if self.optimizer is None or self.pending is None or action != self.pending[0]:
            raise RuntimeError('Training observe must follow act on the same transition')
        if not np.all((obs == 0) | (obs == 1)):
            raise ValueError('Packed rollout requires binary observations')
        next_value = 0. if terminated else float(self.network(torch.as_tensor(next_obs, device=self.device).unsqueeze(0))[0, -1].item())
        _, logp, value = self.pending
        self.rollout.append((np.packbits(obs.astype(np.uint8)), action, reward, value, next_value,
                             bool(terminated), bool(terminated or truncated), action_mask.copy(), logp))
        self.pending = None

    def maybe_train(self, force=False):
        if not self.rollout or (not force and len(self.rollout) < self.cfg.rollout_steps):
            return None
        packed, actions, rewards, values, next_values, terminal, boundaries, masks, logps = zip(*self.rollout)
        advantages, returns = compute_gae(rewards, values, next_values, terminal, boundaries, self.cfg.gamma, self.cfg.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        packed = np.stack(packed)
        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=self.device)
        actions, masks = tensor(actions, torch.long), tensor(np.stack(masks), torch.bool)
        old_logps, advantages, returns = tensor(logps), tensor(advantages), tensor(returns)
        losses, policy_losses, value_losses, entropies, kls, clips = [], [], [], [], [], []
        for _ in range(self.cfg.epochs):
            order = self.rng.permutation(len(self.rollout))
            for start in range(0, len(order), self.cfg.batch_size):
                idx = order[start:start + self.cfg.batch_size]
                obs = np.unpackbits(packed[idx], axis=1, count=int(np.prod(self.obs_shape))).reshape(-1, *self.obs_shape)
                dist, value = self._forward(tensor(obs), masks[idx])
                ratio = (dist.log_prob(actions[idx]) - old_logps[idx]).exp()
                surrogate = torch.minimum(ratio * advantages[idx], ratio.clamp(1-self.cfg.clip_range, 1+self.cfg.clip_range) * advantages[idx])
                policy_loss = -surrogate.mean()
                value_loss = (value-returns[idx]).square().mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite PPO loss')
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                kls.append(((ratio - 1) - ratio.log()).mean().item())
                clips.append(((ratio-1).abs() > self.cfg.clip_range).float().mean().item())
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                losses.append(loss.item())
        self.rollout.clear()
        return {key: float(np.mean(data)) for key, data in zip(
            ['loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction'],
            [losses, policy_losses, value_losses, entropies, kls, clips])}

    def save(self, path):
        torch.save({'algorithm': 'ppo', 'network': self.network.state_dict(), 'cfg': asdict(self.cfg),
                    'observation_schema': self.observation_schema,
                    'obs_shape': self.obs_shape, 'n_actions': self.n_actions}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get('observation_schema', 'pixels_v1') != self.observation_schema:
            raise ValueError('Checkpoint observation schema does not match action-memory setting')
        if checkpoint['algorithm'] != 'ppo' or tuple(checkpoint['obs_shape']) != self.obs_shape or checkpoint['n_actions'] != self.n_actions:
            raise ValueError('Checkpoint architecture does not match environment')
        self.network.load_state_dict(checkpoint['network'])
