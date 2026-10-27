import sys
import types
import numpy as np
import pytest
import torch
from arcengine import GameAction, GameState
from environment.arc_gym_env import ArcAgi3GymEnv, SIMPLE_ACTIONS
from rl_agents.ppo.action_memory import ActionMemory, ActionHistoryEnv, MEMORY_CHANNELS, STREAK_CAP
from rl_agents.ppo.novelty import EpisodicNovelty
from rl_agents.ppo.ppo_agent import PPOAgent, PPOConfig
from tests.test_improvements import make
from tests.test_arc_gym_env import MockArcEnv


def test_history_distinguishes_identical_screens_without_masking():
    a = ActionHistoryEnv(make([0]*10), 1)
    b = ActionHistoryEnv(make([0]*10), 1)
    a.reset(); b.reset()
    oa, _, _, _, ia = a.step(1)
    ob, _, _, _, ib = b.step(2)
    np.testing.assert_array_equal(oa[-16:], ob[-16:])
    assert not np.array_equal(oa[:MEMORY_CHANNELS], ob[:MEMORY_CHANNELS])
    assert ia['action_mask'].all() and ib['action_mask'].all()
    assert set(np.unique(oa)) <= {0, 1}


def test_repeat_streak_detects_changing_screen_cycles_and_is_capped():
    memory = ActionMemory(16)
    obs = np.zeros((16,64,64), np.float32)
    for i in range(10):
        memory.update(1, np.full((64,64),i%2), np.full((64,64),(i+1)%2))
    assert memory.repeats == STREAK_CAP and memory.unchanged == 0
    memory.update(2, obs, obs)
    assert memory.repeats == 1 and memory.unchanged == 1
    memory.update(1, obs, obs, reset=True)
    assert not memory.augment(obs).any()


def test_click_plane_and_newest_pixels_are_preserved():
    memory = ActionMemory(16)
    obs = np.zeros((64,64,64), np.float32); obs[-1]=1
    memory.update(len(SIMPLE_ACTIONS)+255, obs, obs)
    augmented=memory.augment(obs)
    assert augmented.shape==(64+MEMORY_CHANNELS,64,64)
    plane=augmented[len(SIMPLE_ACTIONS)]
    assert plane.sum()==16 and plane[60:64,60:64].all()
    np.testing.assert_array_equal(augmented[MEMORY_CHANNELS:],obs)


def test_soft_explicit_and_episode_resets_clear_memory():
    env=ActionHistoryEnv(ArcAgi3GymEnv(MockArcEnv(),max_actions=20),8)
    assert not env.reset()[0][:MEMORY_CHANNELS].any()
    obs,_,_,_,info=env.step(1)  # death + reset
    assert info['history_reset'] and not obs[:MEMORY_CHANNELS].any()
    assert env.step(2)[0][:MEMORY_CHANNELS].any()
    assert not env.step(0)[0][:MEMORY_CHANNELS].any()
    assert not env.reset()[0][:MEMORY_CHANNELS].any()


def test_memory_does_not_change_novelty_reward():
    plain=EpisodicNovelty(make([0,0,1,0]))
    history=ActionHistoryEnv(EpisodicNovelty(make([0,0,1,0])),1)
    plain.reset();history.reset()
    for action in (1,2,1):
        a=plain.step(action); b=history.step(action)
        assert a[1]==b[1]
        assert a[4]['intrinsic_reward']==b[4]['intrinsic_reward']


def test_submission_observations_match_training_across_soft_reset(monkeypatch):
    from rl_agents.policy_agent import make_policy_agent_class
    module=types.ModuleType('agents.agent')
    module.Agent=type('Agent',(),{'__init__':lambda self,*a,**k:None})
    monkeypatch.setitem(sys.modules,'agents',types.ModuleType('agents'))
    monkeypatch.setitem(sys.modules,'agents.agent',module)
    captured=[]
    class Policy:
        def __init__(self,*a,**k): assert k['action_memory']
        def load(self,path): pass
        def act(self,obs,mask,greedy):
            captured.append(obs.copy()); assert not greedy
            return 7
    submission=make_policy_agent_class(Policy,'unused',click_grid=16,history_len=4,
                     ppo_action_memory=True,stochastic=True,max_actions=1000)()
    assert submission.MAX_ACTIONS==1000
    raw=MockArcEnv()
    env=ActionHistoryEnv(ArcAgi3GymEnv(MockArcEnv(),click_grid=16,history_len=4,max_actions=1000),16)
    obs,_=env.reset(); frame=raw.step(GameAction.RESET)
    for _ in range(3):
        submission.choose_action([],frame)
        np.testing.assert_array_equal(captured[-1],obs)
        obs,_,_,_,_=env.step(7)
        frame=raw.step(GameAction.ACTION6)
        if frame.state is GameState.GAME_OVER:
            assert submission.choose_action([],frame) is GameAction.RESET
            frame=raw.step(GameAction.RESET)


def test_cuda_memory_update_checkpoint_and_strict_schema(tmp_path):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(2)
    env=ActionHistoryEnv(make([0,0,1,0,1]),1)
    obs,info=env.reset()
    agent=PPOAgent(obs.shape,env.n_actions,PPOConfig(rollout_steps=4,batch_size=2,epochs=1),
                   device='cuda',action_memory=True)
    for _ in range(4):
        action=agent.act(obs,info['action_mask'])
        next_obs,reward,term,trunc,next_info=env.step(action)
        agent.observe(obs,action,reward,next_obs,term,info['action_mask'],next_info['action_mask'],truncated=trunc)
        obs,info=next_obs,next_info
    assert all(np.isfinite(v) for v in agent.maybe_train().values())
    path=tmp_path/'memory.pt';agent.save(path)
    loaded=PPOAgent(obs.shape,env.n_actions,device='cuda',inference_only=True,action_memory=True)
    loaded.load(path)
    assert loaded.act(obs,info['action_mask'],greedy=True)==agent.act(obs,info['action_mask'],greedy=True)
    with pytest.raises(ValueError,match='schema'):
        PPOAgent(obs.shape,env.n_actions,inference_only=True).load(path)


def test_memory_experiment_protocol():
    from scripts.run_ppo_action_memory import jobs
    plan=list(jobs())
    assert len(plan)==108
    for name,command in plan:
        assert command[command.index('--device')+1]=='cuda'
        assert command[command.index('--click-grid')+1]=='16'
        assert command[command.index('--games')+1] in ('ft09','dc22','g50t','m0r0')
        assert ('--ppo-action-memory' in command)==(not name.startswith('grid_control_'))
        if not name.endswith('_train'): assert '--ppo-novelty-bonus' not in command


def test_policy_learns_from_history_when_pixels_are_identical():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(2)
    pixels=np.zeros((16,64,64),np.float32);pixels[0]=1
    contexts=[]
    for previous in (1,2):
        memory=ActionMemory(1);memory.update(previous,pixels,pixels)
        contexts.append(memory.augment(pixels))
    mask=np.zeros(8,dtype=bool);mask[1:3]=True
    agent=PPOAgent(contexts[0].shape,8,PPOConfig(rollout_steps=64,batch_size=32,epochs=2),
                   device='cuda',action_memory=True)
    def correct_probabilities():
        with torch.no_grad():
            dist,_=agent._forward(torch.as_tensor(np.stack(contexts),device='cuda'),
                                 torch.as_tensor(np.stack([mask,mask]),device='cuda'))
            return np.array([dist.probs[0,2].item(),dist.probs[1,1].item()])
    before=correct_probabilities()
    # A one-step contextual task: repeat has zero reward; switch has reward one.
    # Pixels are identical; only the recorded preceding action distinguishes cases.
    for _ in range(8):
        for i in range(64):
            context=i%2;obs=contexts[context];correct=2-context
            action=agent.act(obs,mask)
            agent.observe(obs,action,float(action==correct),obs,True,mask,mask)
        agent.maybe_train()
    after=correct_probabilities()
    assert np.all(after>before+.15), (before,after)
    assert np.all(after>.65), after
