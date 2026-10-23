"""Small, observation-only action history shared by training and submission."""
import gymnasium as gym
import numpy as np
from environment.arc_gym_env import SIMPLE_ACTIONS

STREAK_CAP = 4
MEMORY_CHANNELS = len(SIMPLE_ACTIONS) + 1 + 2 * STREAK_CAP
SCHEMA = 'previous_action_repeat_nochange_v1'


class ActionMemory:
    """Binary planes keep PPO's existing packed rollout representation valid.

    Simple actions use one constant plane each; a click marks its lattice cell.
    Repeat and no-change streaks use four thermometer-coded planes each. No
    action is banned and no reward, game identity or private state is inspected.
    """
    def __init__(self, click_grid):
        self.click_grid = click_grid
        self.clear()

    def clear(self):
        self.previous_action = None
        self.repeats = self.unchanged = 0

    def update(self, action, before_grid, after_grid, *, reset=False):
        if reset:
            self.clear()
            return
        self.repeats = min(STREAK_CAP, self.repeats+1) if action == self.previous_action else 1
        self.unchanged = min(STREAK_CAP, self.unchanged+1) if np.array_equal(before_grid, after_grid) else 0
        self.previous_action = int(action)

    def augment(self, obs):
        planes = np.zeros((MEMORY_CHANNELS, 64, 64), dtype=np.float32)
        n_simple = len(SIMPLE_ACTIONS)
        if self.previous_action is not None:
            if self.previous_action < n_simple:
                planes[self.previous_action] = 1
            else:
                cell = self.previous_action - n_simple
                x, y = cell % self.click_grid, cell // self.click_grid
                x0, x1 = int(x*64/self.click_grid), int((x+1)*64/self.click_grid)
                y0, y1 = int(y*64/self.click_grid), int((y+1)*64/self.click_grid)
                planes[n_simple, y0:y1, x0:x1] = 1
            planes[n_simple+1:n_simple+1+self.repeats] = 1
            offset = n_simple+1+STREAK_CAP
            planes[offset:offset+self.unchanged] = 1
        # Keep newest 16 colour channels last: reward/evaluation see pixels only.
        return np.concatenate((planes, obs), axis=0)


class ActionHistoryEnv(gym.Wrapper):
    def __init__(self, env, click_grid):
        super().__init__(env)
        self.n_actions = int(env.action_space.n)
        self.memory = ActionMemory(click_grid)
        self.observation_space = gym.spaces.Box(0., 1.,
            shape=(env.observation_space.shape[0]+MEMORY_CHANNELS, 64, 64), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.memory.clear()
        self.previous_grid = obs[-16:].copy()
        return self.memory.augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.memory.update(action, self.previous_grid, obs[-16:], reset=info['history_reset'])
        self.previous_grid = obs[-16:].copy()
        return self.memory.augment(obs), reward, terminated, truncated, info
