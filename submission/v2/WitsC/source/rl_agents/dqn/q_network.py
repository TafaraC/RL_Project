"""CNN Q-network for the DQN baseline (course-covered algorithm A).

Input: (batch, history_len * 16, 64, 64) one-hot grid stack, as produced
by ArcAgi3GymEnv. Output: one Q-value per discrete action (see
environment/arc_gym_env.py for how the action space is constructed:
6 simple actions + click_grid**2 pointer cells).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, grid_size: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # 64->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32->16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 16->8
            nn.ReLU(inplace=True),
        )
        conv_out = 64 * (grid_size // 8) * (grid_size // 8)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))
