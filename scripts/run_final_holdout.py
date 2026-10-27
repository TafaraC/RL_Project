"""One frozen transfer evaluation: six stages, three training seeds, 5 rollouts/game."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/final_holdout_20261027'
STAGES={
 'ppo_baseline': ('runs/ppo_baseline_cuda_20261013/ft09/ppo_seed{seed}/ppo.pt', 8, []),
 'ppo_novelty': ('runs/ppo_novelty_20261023/novelty/ft09/ppo_seed{seed}/ppo.pt',8,[]),
 'ppo_memory_novelty': ('runs/ppo_action_memory_20261023/memory_novelty/ft09/ppo_seed{seed}/ppo.pt',16,['--ppo-action-memory']),
 'dqn_baseline': ('runs/dqn_archive/runs_final/ft09/dqn_seed{seed}/dqn.pt',8,[]),
 'dqn_novelty': ('runs/dqn_archive/runs_imp1/ft09/dqn_seed{seed}/dqn.pt',8,[]),
 'dqn_noop': ('runs/dqn_archive/runs_imp2/ft09/dqn_seed{seed}/dqn.pt',8,['--noop-masking']),
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(), stages=STAGES,
        games=['cd82','tr87','vc33'], train_seeds=[0,1,2], sampling_seeds=list(range(100,105)),
        episodes_per_game_per_checkpoint=5, max_actions=1000,
        checkpoint_selection='All ft09-trained seeds; same donor game for every stage. No tuning on held-out outcomes.',
        limitation='Five rather than twenty rollouts per checkpoint due to deadline/compute; only three independent training seeds.',
        adaptation='Frozen parameters; episodic memory only; no held-out source inspection.')
    path=OUT/'protocol.json'
    if not path.exists(): path.write_text(json.dumps(protocol,indent=2))
    state={'completed':[], 'total':18, 'state':'running'}
    env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    for name,(source,grid,extra) in STAGES.items():
        for seed in range(3):
            tag=f'{name}_seed{seed}'
            marker=OUT/f'{tag}.complete'
            state['current']=tag
            (OUT/'status.json').write_text(json.dumps(state,indent=2))
            if not marker.exists():
                algorithm=name.split('_')[0]
                command=[sys.executable,'scripts/eval_checkpoint.py','--agent',algorithm,
                    '--checkpoint',source.format(seed=seed),'--games','cd82,tr87,vc33',
                    '--episodes','5','--seed','100','--max-actions','1000','--click-grid',str(grid),
                    '--device','cuda','--offline','--out-dir',str(OUT),'--tag',tag,*extra]
                command+=['--stochastic'] if algorithm=='ppo' else ['--eval-epsilon','0.3']
                with (OUT/f'{tag}.log').open('w') as log:
                    result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    state.update(state='failed',returncode=result.returncode)
                    (OUT/'status.json').write_text(json.dumps(state,indent=2))
                    raise SystemExit(result.returncode)
                marker.write_text('complete\n')
            state['completed'].append(tag)
    state.update(state='complete',current=None)
    (OUT/'status.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__': main()
