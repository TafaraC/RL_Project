"""Fixed PPO diagnostic and first-improvement experiment; no DQN changes."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'runs/ppo_baseline_cuda_20261013'
OUT = ROOT / 'runs/ppo_novelty_20261023'
GAMES = ['ft09', 'dc22', 'g50t', 'm0r0']


def jobs():
    # Diagnose existing PPO before collecting any improved-training results.
    for game in GAMES:
        for seed in range(3):
            checkpoint = BASELINE / game / f'ppo_seed{seed}/ppo.pt'
            yield f'baseline_{game}_{seed}_sampled', evaluation(checkpoint, game, seed, 'baseline', 'sampled')
    for game in GAMES:
        for seed in range(3):
            train_dir = OUT / 'novelty' / game
            checkpoint = train_dir / f'ppo_seed{seed}/ppo.pt'
            yield f'novelty_{game}_{seed}_train', [
                'scripts/run_baseline.py', '--agent', 'ppo', '--games', game,
                '--seed', str(seed), '--env-steps', '50000', '--max-actions', '1000',
                '--device', 'cuda', '--offline', '--ppo-novelty-bonus', '0.01',
                '--ppo-novelty-cap', '0.5', '--out-dir', str(train_dir)]
            for mode in ('sampled', 'greedy'):
                yield f'novelty_{game}_{seed}_{mode}', evaluation(checkpoint, game, seed, 'novelty', mode)


def evaluation(checkpoint, game, seed, variant, mode):
    command = ['scripts/eval_checkpoint.py', '--agent', 'ppo', '--checkpoint', str(checkpoint),
               '--games', game, '--episodes', '20', '--seed', '100', '--max-actions', '1000',
               '--device', 'cuda', '--offline', '--out-dir', str(OUT / variant / game / 'eval'),
               '--tag', f'seed{seed}_{mode}']
    if mode == 'sampled':
        command.append('--stochastic')
    return command


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    for game in GAMES:
        for seed in range(3):
            if not (BASELINE / game / f'ppo_seed{seed}/ppo.pt').is_file():
                raise FileNotFoundError(f'Missing baseline {game} seed {seed}')
    OUT.mkdir(parents=True, exist_ok=True)
    plan = list(jobs())
    (OUT / 'protocol.json').write_text(json.dumps({
        'base_branch': 'origin/Nathan2 e542c3c', 'device': torch.cuda.get_device_name(),
        'primary_evaluation': 'sampled', 'secondary_evaluation': 'greedy',
        'novelty_beta': .01, 'novelty_cap': .5, 'commands': dict(plan)}, indent=2))
    state = {'completed': [], 'total_jobs': len(plan)}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        (OUT / 'status.json').write_text(json.dumps(state, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for name, command in plan:
        marker = OUT / f'{name}.complete'
        if not marker.exists():
            state.update(state='running', current=name)
            save()
            with (OUT / f'{name}.log').open('w') as log:
                result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(state='failed', returncode=result.returncode)
                save()
                raise SystemExit(result.returncode)
            marker.touch()
        state['completed'].append(name)
        save()
    state.update(state='complete')
    save()


if __name__ == '__main__':
    main()
