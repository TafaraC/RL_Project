import numpy as np
import torch
import pytest
from rl_agents.ppo.ppo_agent import PPOAgent, PPOConfig, compute_gae


def test_gae_boundaries():
    adv, ret = compute_gae([1, 2, 3], [0, 0, 0], [10, 20, 30],
                           [False, True, False], [True, True, False], .9, 1.)
    np.testing.assert_allclose(adv, [10, 2, 30])
    np.testing.assert_allclose(ret, adv)


def test_ppo_update_mask_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    agent = PPOAgent((16, 64, 64), 3, PPOConfig(rollout_steps=3, batch_size=2, epochs=2))
    obs = np.zeros((16, 64, 64), dtype=np.float32)
    obs[0] = 1
    mask = np.array([False, True, False])
    before = [p.detach().clone() for p in agent.network.parameters()]
    for i in range(3):
        action = agent.act(obs, mask)
        assert action == 1
        agent.observe(obs, action, 1., obs, i == 2, mask, mask, truncated=i == 1)
    assert np.isfinite(agent.maybe_train()['loss'])
    assert not agent.rollout
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.network.parameters()))
    path = tmp_path / 'ppo.pt'
    agent.save(path)
    loaded = PPOAgent((16, 64, 64), 3, inference_only=True)
    loaded.load(path)
    for a, b in zip(agent.network.parameters(), loaded.network.parameters()):
        torch.testing.assert_close(a, b)
    assert loaded.act(obs, mask, greedy=True) == 1
    with pytest.raises(ValueError): loaded.act(obs, np.zeros(3, dtype=bool))
    action = agent.act(obs, mask)
    agent.observe(obs, action, 1., obs, False, mask, mask)
    assert agent.maybe_train() is None
    assert np.isfinite(agent.maybe_train(force=True)['loss'])


def test_runner_and_evaluator(tmp_path):
    from argparse import Namespace
    from scripts.run_baseline import run
    from scripts.eval_checkpoint import evaluate
    from tests.test_arc_gym_env import MockArcEnv
    class Arcade:
        def make(self, *args, **kwargs): return MockArcEnv()
    args = Namespace(agent='ppo', out_dir=str(tmp_path), seed=0, games='mock',
                     episodes=1, env_steps=3, device='cpu', click_grid=2,
                     history_len=1, max_actions=3)
    run(Arcade(), args)
    args.checkpoint = str(tmp_path / 'ppo_seed0' / 'ppo.pt')
    args.tag = 'evaluation'
    args.eval_epsilon = 0.
    evaluate(Arcade(), args)
    assert (tmp_path / 'evaluation' / 'episodes.csv').exists()
