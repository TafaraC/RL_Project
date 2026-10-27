"""Complete inexpensive reference and cumulative DQN ablation on development only."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/final_supplement_20261027'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    commands=[]
    for seed in range(3):
        commands.append((f'random_seed{seed}',[sys.executable,'scripts/run_baseline.py','--agent','random',
            '--games','dc22,ft09,g50t,m0r0','--episodes','20','--seed',str(seed),'--max-actions','1000',
            '--offline','--device','cpu','--out-dir',str(OUT/'random')]))
        commands.append((f'dqn_combined_seed{seed}',[sys.executable,'scripts/eval_checkpoint.py','--agent','dqn',
            '--checkpoint',f'runs/dqn_archive/runs_imp12/ft09/dqn_seed{seed}/dqn.pt',
            '--games','ft09','--episodes','20','--seed','100','--max-actions','1000',
            '--offline','--device','cuda','--eval-epsilon','0.3','--noop-masking',
            '--out-dir',str(OUT),'--tag',f'dqn_combined_seed{seed}']))
    state={'completed':[],'total':len(commands),'state':'running'}
    for name,cmd in commands:
        state['current']=name
        (OUT/'status.json').write_text(json.dumps(state,indent=2))
        with (OUT/f'{name}.log').open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                                  stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(state='failed',returncode=result.returncode)
            (OUT/'status.json').write_text(json.dumps(state,indent=2))
            raise SystemExit(result.returncode)
        state['completed'].append(name)
    state.update(state='complete',current=None)
    (OUT/'status.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()
