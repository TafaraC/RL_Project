"""Run the fixed PPO development protocol sequentially; resume completed jobs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'ppo_baseline'


def main(training_only=False, device="cuda", games=None, out_dir=None, evaluation_only=False):
    global OUT
    if out_dir is not None:
        OUT = Path(out_dir).resolve()
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run but unavailable")
    if evaluation_only:
        for game in games or ['ft09', 'dc22', 'g50t', 'm0r0']:
            for seed in range(3):
                checkpoint = OUT / game / f'ppo_seed{seed}' / 'ppo.pt'
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    status = {'protocol': {'games': games or ['ft09', 'dc22', 'g50t', 'm0r0'], 'seeds': [0, 1, 2],
                           'training_steps': 50000, 'max_actions': 1000,
                           'eval_episodes': 20, 'eval_seed': 100, 'eval_epsilon': 0,
                           'device': device, 'threads': 2, 'training_only': training_only, 'evaluation_only': evaluation_only}, 'completed': []}
    def save():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = OUT / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(OUT / 'status.json')
    summaries = []
    for game in status['protocol']['games']:
        for seed in status['protocol']['seeds']:
            base = OUT / game
            checkpoint = base / f'ppo_seed{seed}' / 'ppo.pt'
            evaluation = base / 'eval' / f'seed{seed}'
            marker = base / f'seed{seed}.complete'
            if not marker.exists():
                base.mkdir(parents=True, exist_ok=True)
                commands = [
                    ('train', ['scripts/run_baseline.py', '--agent', 'ppo', '--games', game,
                               '--env-steps', '50000', '--seed', str(seed), '--max-actions', '1000',
                               '--out-dir', str(base), '--device', device, '--offline']),
                    ('evaluate', ['scripts/eval_checkpoint.py', '--agent', 'ppo', '--checkpoint', str(checkpoint),
                                  '--games', game, '--episodes', '20', '--seed', '100', '--max-actions', '1000',
                                  '--out-dir', str(base / 'eval'), '--tag', f'seed{seed}', '--device', device, '--offline'])]
                for phase, command in commands:
                    if training_only and phase == 'evaluate':
                        continue
                    if phase == 'train' and (evaluation_only or checkpoint.exists()):
                        continue
                    status.update(state='running', game=game, seed=seed, phase=phase)
                    save()
                    with (base / f'seed{seed}_{phase}.log').open('w') as log:
                        result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        status.update(state='failed', returncode=result.returncode)
                        save()
                        raise SystemExit(result.returncode)
                if not training_only:
                    marker.touch()
            if training_only:
                status['completed'].append(f'{game}/seed{seed}')
                save()
                continue
            rows = list(csv.DictReader((evaluation / 'episodes.csv').open()))
            summaries.append({'game': game, 'training_seed': seed, 'episodes': len(rows),
                              'episodes_with_level': sum(int(r['levels_completed']) > 0 for r in rows),
                              'wins': sum(int(r['won']) for r in rows),
                              'mean_actions': sum(int(r['actions_used']) for r in rows) / len(rows),
                              'mean_noop_fraction': sum(float(r['noop_frac']) for r in rows) / len(rows)})
            with (OUT / 'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
            status['completed'].append(f'{game}/seed{seed}')
            save()
    status.update(state='complete', phase='complete')
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--training-only", action="store_true")
    modes.add_argument("--evaluation-only", action="store_true")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--games', default='ft09,dc22,g50t,m0r0')
    parser.add_argument('--out-dir', default=str(OUT))
    args = parser.parse_args()
    main(training_only=args.training_only, device=args.device,
         games=args.games.split(','), out_dir=args.out_dir, evaluation_only=args.evaluation_only)
