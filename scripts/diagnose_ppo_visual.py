"""Frozen CUDA PPO diagnostics; privileged state is logged, never fed to policy."""
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_baseline import make_env
from rl_agents.ppo.ppo_agent import PPOAgent
from arc_agi import Arcade, OperationMode

OUT = Path('runs/ppo_visual_diagnostics_20261023')


def state(env, data):
    game = env.arc_env._game
    result = {'level_index': int(game.level_index)}
    # Read-only primitive fields: useful for timers/selection/phase without copying engine.
    result['scalars'] = {k: v for k, v in vars(game).items()
                         if type(v) in (int, float, bool, str) and not k.startswith('_')}
    if env.game_id == 'ft09':
        point = game.camera.display_to_grid(data['x'], data['y']) if data else None
        result['target'] = 'outside board'
        if point:
            result['target'] = 'background/feedback'
            for tag, label in [('Hkx', 'puzzle tile'), ('NTi', 'pattern tile'), ('bsT', 'constraint tile')]:
                if game.current_level.get_sprite_at(*point, tag):
                    result['target'] = label
                    break
        result['moves_left'] = game.lpw.dzy
        result['tiles'] = [[s.x, s.y, int(s.pixels[1][1])] for s in game.fhc + game.mou]
    return result


def run():
    if list(OUT.glob('*.json.gz')):
        raise FileExistsError(f'Recorded traces already exist in {OUT}; choose a new output directory')
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    jobs = [('ft09', 2, False), ('ft09', 2, True), ('ft09', 0, False),
            ('dc22', 0, False), ('g50t', 0, False), ('m0r0', 0, False)]
    summaries = []
    for game, seed, greedy in jobs:
        name = f'{game}_seed{seed}_{"greedy" if greedy else "sampled"}'
        checkpoint = Path(f'runs/ppo_novelty_20261023/novelty/{game}/ppo_seed{seed}/ppo.pt')
        env = make_env(arc, game, SimpleNamespace(click_grid=8, history_len=4, max_actions=1000), 100)
        agent = PPOAgent(env.observation_space.shape, env.n_actions, device='cuda', inference_only=True)
        agent.load(checkpoint)
        torch.manual_seed(100)
        obs, info = env.reset(seed=100)
        frames, indices, rows = [], {}, []

        def frame_id(grid):
            flat = np.asarray(grid, dtype=np.uint8).reshape(-1)
            key = flat.tobytes()
            if key not in indices:
                indices[key] = len(frames)
                # Run-length encoding preserves every pixel and keeps inline replays small.
                boundaries = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, len(flat)]
                frames.append([[int(flat[a]), int(b-a)] for a, b in zip(boundaries[:-1], boundaries[1:])])
            return indices[key]

        raw_step = env.arc_env.step
        events = []

        def record_step(action, **kwargs):
            raw = raw_step(action, **kwargs)
            events.append({'action': action.name, 'state': raw.state.name,
                           'level': raw.levels_completed, 'frames': [frame_id(g) for g in raw.frame]})
            return raw

        env.arc_env.step = record_step
        streak = 0
        while True:
            before = frame_id(obs[-16:].argmax(0))
            with torch.no_grad():
                dist, value = agent._forward(torch.as_tensor(obs, device='cuda').unsqueeze(0),
                    torch.as_tensor(info['action_mask'], device='cuda').unsqueeze(0))
                probs = dist.probs[0].cpu().numpy()
                entropy = float(dist.entropy().item())
            action = agent.act(obs, info['action_mask'], greedy=greedy)
            decoded, data = env._decode(action)
            prestate = state(env, data)
            events.clear()
            next_obs, reward, term, trunc, info = env.step(action)
            changed = int(np.count_nonzero(obs[-16:].argmax(0) != next_obs[-16:].argmax(0)))
            streak = streak + 1 if changed == 0 else 0
            top = np.argsort(probs)[-3:][::-1]
            rows.append({'step': len(rows)+1, 'actions_used': info['actions_used'],
                'before': before, 'after': frame_id(next_obs[-16:].argmax(0)), 'action': action,
                'label': decoded.name, 'xy': list(data.values()) if data else None,
                'prob': round(float(probs[action]), 5), 'entropy': round(entropy, 4),
                'top': [[int(a), round(float(probs[a]), 5)] for a in top],
                'value': round(float(value.item()), 5), 'changed': changed, 'streak': streak,
                'levels': info['levels_completed'], 'reward': reward, 'state': prestate,
                'events': list(events)})
            obs = next_obs
            if term or trunc:
                break
        counts = Counter(r['action'] for r in rows)
        summary = {'name': name, 'device': str(next(agent.network.parameters()).device),
            'gpu': torch.cuda.get_device_name(), 'checkpoint': str(checkpoint),
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'sampling_seed': 100, 'steps': len(rows), 'actions_used': info['actions_used'],
            'levels': info['levels_completed'], 'won': info['raw_state'].name == 'WIN',
            'no_change_fraction': sum(r['changed']==0 for r in rows)/len(rows),
            'longest_no_change_streak': max(r['streak'] for r in rows),
            'action_counts': counts, 'targets': Counter(r['state'].get('target', '') for r in rows),
            'deaths': sum(e['state']=='GAME_OVER' for r in rows for e in r['events']),
            'level_actions': [r['actions_used'] for i,r in enumerate(rows) if r['levels'] > (rows[i-1]['levels'] if i else 0)]}
        with gzip.open(OUT / f'{name}.json.gz', 'wt', encoding='utf-8') as f:
            json.dump({'summary': summary, 'frames': frames, 'steps': rows}, f)
        summaries.append(summary)
        (OUT/'summary.json').write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
