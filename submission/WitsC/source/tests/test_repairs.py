import sys
import types
import numpy as np
import pytest
from arcengine import GameAction, GameState
from environment.arc_gym_env import ArcAgi3GymEnv, SIMPLE_ACTIONS
from tests.test_arc_gym_env import MockArcEnv, FakeRaw
from rl_agents.dqn.replay_buffer import ReplayBuffer
from rl_agents.dqn.dqn_agent import DQNAgent, DQNConfig
from evaluation.rhae import extract_level_records


def test_replay_roundtrip_padding_and_overwrite():
    buffer = ReplayBuffer(1, (3, 2, 2), 2)
    obs = np.zeros((3, 2, 2), dtype=np.float32)
    obs[1, 0, 0] = 1
    mask = np.array([True, False])
    buffer.add(obs, 0, 1, 1-obs, False, mask, mask)
    batch = buffer.sample(2)
    np.testing.assert_array_equal(batch.obs[0], obs)
    np.testing.assert_array_equal(batch.next_obs[0], 1-obs)
    assert batch.obs.dtype == np.float32
    assert buffer.obs[0].nbytes == 2
    buffer.add(1-obs, 0, 2, obs, True, mask, mask)
    assert len(buffer) == 1
    assert buffer.sample(1).reward[0] == 2
    with pytest.raises(ValueError):
        buffer.add(obs + .5, 0, 0, obs, False, mask, mask)


def test_reset_history_matches_submission(monkeypatch):
    from rl_agents.policy_agent import make_policy_agent_class
    module = types.ModuleType('agents.agent')
    class Agent:
        def __init__(self, *args, **kwargs): pass
    module.Agent = Agent
    monkeypatch.setitem(sys.modules, 'agents', types.ModuleType('agents'))
    monkeypatch.setitem(sys.modules, 'agents.agent', module)
    class Policy:
        def __init__(self, *args, **kwargs):
            assert kwargs == {'device': 'cpu', 'inference_only': True}
        def load(self, path): pass
        def act(self, obs, mask, greedy):
            self.obs = obs
            return SIMPLE_ACTIONS.index(GameAction.ACTION1)
    agent = make_policy_agent_class(Policy, 'unused')()
    env = ArcAgi3GymEnv(MockArcEnv())
    obs, _ = env.reset()
    agent.choose_action([], env._last_frame)
    np.testing.assert_array_equal(obs, agent._policy.obs)
    obs, reward, _, _, info = env.step(SIMPLE_ACTIONS.index(GameAction.ACTION1))
    # Submission clears history when choosing RESET after GAME_OVER.
    dead = FakeRaw(1, GameState.GAME_OVER, 0, [1])
    assert agent.choose_action([], dead) is GameAction.RESET
    agent.choose_action([], env._last_frame)
    np.testing.assert_array_equal(obs, agent._policy.obs)
    assert info['actions_used'] == 2
    assert reward == pytest.approx(-.054)
    assert not obs[:-16].any()


def test_bounds_and_episode_end():
    env = ArcAgi3GymEnv(MockArcEnv(), max_actions=1)
    env.reset()
    with pytest.raises(ValueError): env.step(-1)
    env.step(1)
    with pytest.raises(RuntimeError): env.step(1)


def test_checkpoint_and_training(tmp_path):
    cfg = DQNConfig(buffer_capacity=2, batch_size=1, learning_starts=1, train_freq=1, device='cpu')
    agent = DQNAgent((16, 64, 64), 2, cfg)
    obs = np.zeros((16, 64, 64), dtype=np.float32)
    obs[0] = 1
    mask = np.array([True, False])
    agent.observe(obs, 0, 1, obs, True, mask, np.zeros(2, dtype=bool))
    assert np.isfinite(agent.maybe_train()['loss'])
    checkpoint = tmp_path / 'policy.pt'
    agent.save(checkpoint)
    loaded = DQNAgent((16, 64, 64), 2, device='cpu', inference_only=True)
    loaded.load(checkpoint)
    assert loaded.buffer is None
    assert loaded.act(obs, mask, greedy=True) == 0
    with pytest.raises(ValueError): loaded.act(obs, np.zeros(2, dtype=bool), greedy=True)


def test_missing_baselines_rejected():
    with pytest.raises(ValueError): extract_level_records([], 2, {1: 3})
