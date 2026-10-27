"""Verify embedded notebook assets, checkpoint parity and real official loop."""
import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actions', type=int, default=64)
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    framework = ROOT/'runs/kaggle_reference/ARC-AGI-3-Agents-main'
    (framework/'agents/__init__.py').write_text('from .agent import Agent\n')
    sys.path.insert(0, str(framework))
    spec = importlib.util.spec_from_file_location('witsc_agent', ROOT/'kaggle/my_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from arcengine import FrameData, GameState
    from arc_agi import Arcade, OperationMode
    from rl_agents.ppo.ppo_agent import PPOAgent
    from rl_agents.ppo.action_memory import ActionMemory
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT/'environment_files'))
    out = ROOT/'submission/WitsC/validation'
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for model, config in json.loads((ROOT/'submission/WitsC/kaggle/manifest.json').read_text()).items():
        notebook = json.loads((ROOT/f'submission/WitsC/kaggle/{model}.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(cell['source'], '<notebook cell>', 'exec')
        # Execute the actual extraction cell, changing only its Linux temp path.
        scope = {}
        extraction = notebook['cells'][1]['source'].replace("Path('/tmp/witsc_assets')", f'Path({str(ROOT / "runs/kaggle_extracted" / model)!r})')
        exec(extraction, scope)
        assets = scope['root']
        assert hashlib.sha256((assets/'weights.npz').read_bytes()).hexdigest() == config['weights_sha256']
        assert (assets/'my_agent.py').read_bytes() == (ROOT/'kaggle/my_agent.py').read_bytes()
        os.environ['WITSC_ASSETS'] = str(assets)
        os.environ['WITSC_DEVICE'] = 'cuda'
        agent = module.MyAgent(card_id='', game_id='probe', agent_name='myagent', ROOT_URL='', record=False, arc_env=None)
        if config['algorithm'] == 'ppo':
            ref = PPOAgent((80 if config['action_memory'] else 64, 64, 64), 7+config['click_grid']**2,
                           device='cuda', inference_only=True, action_memory=config['action_memory'])
        else:
            from rl_agents.dqn.dqn_agent import DQNAgent, DQNConfig
            ref = DQNAgent((64,64,64),71,DQNConfig(device='cuda'),inference_only=True)
        ref.load(ROOT/config['source'])
        history, memory = [], ActionMemory(config['click_grid'])
        previous = None
        for i in range(8):
            raw = np.full((64,64), i % 3, dtype=np.int64)
            hot = np.eye(16, dtype=np.float32)[raw].transpose(2,0,1)
            history = (history + [hot])[-4:]
            expected = np.concatenate([np.zeros_like(hot)]*(4-len(history))+history)
            if config['action_memory']:
                if previous is not None:
                    memory.update(agent.previous_action, previous, hot)
                expected = memory.augment(expected)
            frame = FrameData(frame=[raw.tolist()], state=GameState.NOT_FINISHED, levels_completed=0, available_actions=[1,2,3,4,5,6,7])
            actual = agent.observation(frame)
            np.testing.assert_array_equal(actual, expected)
            x = torch.from_numpy(actual).unsqueeze(0).cuda()
            with torch.inference_mode():
                torch.testing.assert_close(agent.network(x), (ref.network if config['algorithm']=='ppo' else ref.q)(x), rtol=0, atol=0)
            # Advance chosen-action history manually to test identical repeats and clicks.
            a = [1,1,10,10,12,3,3,3][i]
            agent.repeats = min(4, agent.repeats+1) if a == agent.previous_action else 1
            agent.previous_action = a
            previous = hot
        reset_frame = FrameData(frame=[], state=GameState.GAME_OVER, levels_completed=0)
        agent.choose_action([], reset_frame)
        assert not agent.history and agent.previous_action is None
        del agent, ref
        for game in ['ft09','dc22','g50t','m0r0']:
            env = arc.make(game)
            agent = module.MyAgent(card_id='', game_id=game, agent_name='myagent', ROOT_URL='', record=False, arc_env=env)
            agent.MAX_ACTIONS = args.actions
            t = time.monotonic()
            agent.main()  # actual upstream Agent.main, not a mocked interface
            assert agent.action_counter == args.actions+1 or agent.state == GameState.WIN
            results.append(dict(model=model, game=game, actions=agent.action_counter,
                levels=agent.levels_completed, won=agent.state == GameState.WIN,
                seconds=round(time.monotonic()-t, 3), device=str(agent.device),
                check='official Agent.main local game; not a Kaggle gateway test'))
            print(model, game, results[-1], flush=True)
            del agent
    (out/'local_integration.json').write_text(json.dumps(results, indent=2)+'\n')
    (out/'validation_summary.json').write_text(json.dumps(dict(
        embedded_archive_integrity='passed', notebook_syntax='passed',
        lossless_export_network_parity='passed', memory_observation_parity='passed',
        official_agent_loop='passed', cuda_gpu=torch.cuda.get_device_name(),
        python=sys.version, torch=torch.__version__, kaggle_gateway='not run; requires competition access'), indent=2)+'\n')

if __name__ == '__main__':
    main()
