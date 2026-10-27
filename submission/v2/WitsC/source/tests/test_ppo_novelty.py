import math
import numpy as np
import pytest
from rl_agents.ppo.novelty import EpisodicNovelty
from tests.test_improvements import make


def test_counts_decay_and_cap_with_separate_rewards():
    env = EpisodicNovelty(make([0, 1, 1, 2]), beta=.1, cap=.2)
    env.reset()
    a = env.step(1)
    b = env.step(1)
    c = env.step(1)
    assert a[4]['intrinsic_reward'] == pytest.approx(.1)
    assert b[4]['intrinsic_reward'] == pytest.approx(.1/math.sqrt(2))
    assert c[4]['novelty_bonus_total'] == pytest.approx(.2)
    assert a[1] == pytest.approx(a[4]['extrinsic_reward']+a[4]['intrinsic_reward'])
    assert env.step(1)[4]['intrinsic_reward'] == 0


def test_counts_survive_soft_reset_but_reset_between_episodes():
    from environment.arc_gym_env import ArcAgi3GymEnv
    from tests.test_arc_gym_env import MockArcEnv
    env = EpisodicNovelty(ArcAgi3GymEnv(MockArcEnv()), beta=.1)
    env.reset()
    step = env.step(1)  # GAME_OVER followed by automatic reset to initial screen
    assert step[4]['intrinsic_reward'] == pytest.approx(.1/math.sqrt(2))
    env.reset()
    assert sum(env.counts.values()) == 1
    assert env.total == 0


def test_zero_bonus_matches_original():
    plain = make([0, 1, 1])
    wrapped = EpisodicNovelty(make([0, 1, 1]), beta=0)
    np.testing.assert_array_equal(plain.reset()[0], wrapped.reset()[0])
    assert plain.step(1)[1] == wrapped.step(1)[1]


def test_ppo_only_wiring_and_reward_log(tmp_path):
    from argparse import Namespace
    import csv
    from scripts.run_baseline import run, make_env
    from tests.test_improvements import ScriptedArc
    class Arcade:
        def make(self, *args, **kwargs): return ScriptedArc([0, 1, 1, 2])
    args = Namespace(agent='ppo', out_dir=str(tmp_path), seed=0, games='mock',
                     episodes=1, env_steps=3, device='cpu', click_grid=1,
                     history_len=1, max_actions=3, ppo_novelty_bonus=.01, ppo_novelty_cap=.5)
    run(Arcade(), args)
    with (tmp_path/'ppo_seed0/episodes.csv').open() as f:
        row = next(csv.DictReader(f))
    assert float(row['novelty_bonus_total']) > 0
    assert float(row['shaped_return']) == pytest.approx(float(row['extrinsic_return'])+float(row['novelty_bonus_total']), abs=.00011)
    from rl_agents.ppo.ppo_agent import PPOAgent
    loaded = PPOAgent((16, 64, 64), 8, inference_only=True)
    loaded.load(tmp_path/'ppo_seed0/ppo.pt')
    args.agent = 'dqn'
    with pytest.raises(ValueError): make_env(Arcade(), 'mock', args, 0)


def test_experiment_plan_keeps_dqn_and_heldout_untouched():
    from scripts.run_ppo_novelty import jobs
    plan = list(jobs())
    assert len(plan) == 48
    assert all('baseline_' in name and '--stochastic' in command for name, command in plan[:12])
    for name, command in plan:
        assert command[command.index('--agent')+1] == 'ppo'
        assert command[command.index('--games')+1] in ['ft09', 'dc22', 'g50t', 'm0r0']
        if 'train' not in name:
            assert '--ppo-novelty-bonus' not in command


def test_sampled_evaluation_reproducible(tmp_path):
    from argparse import Namespace
    from scripts.eval_checkpoint import evaluate
    from tests.test_improvements import ScriptedArc
    from rl_agents.ppo.ppo_agent import PPOAgent
    import csv
    import torch
    class Arcade:
        def make(self, *args, **kwargs): return ScriptedArc([0, 1, 1, 2])
    checkpoint = tmp_path / 'ppo.pt'
    agent = PPOAgent((16, 64, 64), 8, inference_only=True)
    agent.save(checkpoint)
    args = Namespace(agent='ppo', checkpoint=str(checkpoint), games='mock', episodes=2,
                     seed=100, max_actions=10, click_grid=1, history_len=1, device='cpu',
                     out_dir=str(tmp_path), tag='first', stochastic=True, eval_epsilon=0.)
    evaluate(Arcade(), args)
    torch.manual_seed(9999)
    args.tag = 'second'
    evaluate(Arcade(), args)
    def rows(tag):
        with (tmp_path/tag/'episodes.csv').open() as f:
            return [{k:v for k,v in row.items() if k != 'wall_s'} for row in csv.DictReader(f)]
    assert rows('first') == rows('second')
    args.ppo_novelty_bonus = .01
    with pytest.raises(ValueError, match='disabled'): evaluate(Arcade(), args)
