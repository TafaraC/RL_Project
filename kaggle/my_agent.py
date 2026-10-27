"""Portable frozen PPO inference for the official ARC-AGI-3 Agent interface.

No training dependencies, private state, game-ID routing, or network calls.
Weights are NumPy arrays exported losslessly from our own PPO checkpoints.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from arcengine import GameAction, GameState
from agents.agent import Agent

SIMPLE = [GameAction.RESET, GameAction.ACTION1, GameAction.ACTION2,
          GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]


class Network(nn.Module):
    def __init__(self, channels, outputs):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, 32, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4096, 512),
                                  nn.ReLU(inplace=True), nn.Linear(512, outputs))

    def forward(self, obs):
        return self.head(self.conv(obs))


class MyAgent(Agent):
    # Official Agent.main uses <= MAX_ACTIONS. Initial RESET plus 1,000
    # subsequent actions reproduces the project's initial-reset exclusion.
    MAX_ACTIONS = 1000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = Path(os.environ.get('WITSC_ASSETS', Path(__file__).parent))
        self.config = json.loads((root / 'model.json').read_text())
        self.grid = self.config['click_grid']
        self.memory = self.config['action_memory']
        self.device = torch.device(os.environ.get('WITSC_DEVICE', 'cuda'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA required: enable a Kaggle GPU accelerator')
        self.algorithm = self.config.get('algorithm', 'ppo')
        self.network = Network(80 if self.memory else 64, 7 + self.grid ** 2 + (self.algorithm == 'ppo'))
        with np.load(root / 'weights.npz', allow_pickle=False) as arrays:
            self.network.load_state_dict({k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}, strict=True)
        self.network.to(self.device).eval()
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(os.environ.get('WITSC_SEED', '100')))
        self.action_counter = 0
        self.banned = {}
        self.noop_key = self.noop_action = None
        self.clear_history()

    def clear_history(self):
        self.history = []
        self.previous_action = None
        self.previous_grid = None
        self.repeats = self.unchanged = 0
        self.noop_key = self.noop_action = None

    def is_done(self, frames, latest_frame):
        return latest_frame.state == GameState.WIN

    def observation(self, frame):
        raw = np.asarray(frame.frame[-1] if len(frame.frame) else np.zeros((64, 64)), dtype=np.int64)
        if raw.shape != (64, 64) or raw.min() < 0 or raw.max() > 15:
            raise ValueError('Unexpected official frame shape or colour range')
        hot = np.eye(16, dtype=np.float32)[raw].transpose(2, 0, 1)
        self.history = (self.history + [hot])[-4:]
        obs = np.concatenate([np.zeros_like(hot)] * (4-len(self.history)) + self.history)
        if self.memory:
            planes = np.zeros((16, 64, 64), dtype=np.float32)
            if self.previous_action is not None:
                self.unchanged = min(4, self.unchanged+1) if np.array_equal(raw, self.previous_grid) else 0
                a = self.previous_action
                if a < 7:
                    planes[a] = 1
                else:
                    x, y = (a-7) % self.grid, (a-7) // self.grid
                    planes[7, int(y*64/self.grid):int((y+1)*64/self.grid),
                           int(x*64/self.grid):int((x+1)*64/self.grid)] = 1
                planes[8:8+self.repeats] = 1
                planes[12:12+self.unchanged] = 1
            obs = np.concatenate([planes, obs])
        self.previous_grid = raw.copy()
        return obs

    @torch.inference_mode()
    def choose_action(self, frames, latest_frame):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.clear_history()
            return GameAction.RESET
        obs = self.observation(latest_frame)
        mask = np.zeros(7+self.grid**2, dtype=bool)
        for action_id in latest_frame.available_actions:
            action = GameAction.from_id(action_id)
            if action in SIMPLE:
                mask[SIMPLE.index(action)] = True
            elif action == GameAction.ACTION6:
                mask[7:] = True
        if not mask.any():
            raise ValueError('Official interface returned no legal actions')
        if self.config.get('noop_masking', False):
            key = self.previous_grid.tobytes()
            if self.noop_key == key and self.noop_action is not None:
                self.banned.setdefault(key, set()).add(self.noop_action)
            filtered = mask.copy()
            for action_id in self.banned.get(key, set()):
                filtered[action_id] = False
            if filtered.any():
                mask = filtered
            self.noop_key = key
        output = self.network(torch.from_numpy(obs).unsqueeze(0).to(self.device))[0]
        if self.algorithm == 'ppo':
            output = output[:-1]
        logits = output.masked_fill(~torch.from_numpy(mask).to(self.device), -torch.inf)
        if self.algorithm == 'dqn':
            epsilon = self.config.get('eval_epsilon', .3)
            probs = torch.from_numpy(mask).to(self.device, dtype=torch.float32)
            probs = epsilon * probs / probs.sum()
            probs[logits.argmax()] += 1-epsilon
        else:
            probs = logits.softmax(-1)
        if not torch.isfinite(probs).all():
            raise ValueError('Non-finite policy distribution')
        a = int(torch.multinomial(probs, 1, generator=self.generator).item())
        self.repeats = min(4, self.repeats+1) if a == self.previous_action else 1
        self.previous_action = a
        self.noop_action = a
        if a < 7:
            if SIMPLE[a] == GameAction.RESET:
                self.clear_history()
            return SIMPLE[a]
        cell = a-7
        action = GameAction.ACTION6
        action.set_data({'x': min(int((cell % self.grid + .5)*64/self.grid), 63),
                         'y': min(int((cell // self.grid + .5)*64/self.grid), 63)})
        return action
