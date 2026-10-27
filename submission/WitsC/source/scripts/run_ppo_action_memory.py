"""Fixed CUDA comparison: finer-grid control, memory addition, reward ablation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ppo_action_memory_20261023'
GAMES=('ft09','dc22','g50t','m0r0')
VARIANTS=('grid_control','memory_novelty','memory_only')


def jobs():
    for game in GAMES:
        for seed in range(3):
            for variant in VARIANTS:
                folder=OUT/variant/game
                checkpoint=folder/f'ppo_seed{seed}/ppo.pt'
                common=['--agent','ppo','--games',game,'--max-actions','1000',
                        '--click-grid','16','--device','cuda','--offline']
                if variant!='grid_control': common+=['--ppo-action-memory']
                train=['scripts/run_baseline.py',*common,'--seed',str(seed),'--env-steps','50000','--out-dir',str(folder)]
                if variant!='memory_only': train+=['--ppo-novelty-bonus','0.01','--ppo-novelty-cap','0.5']
                yield f'{variant}_{game}_{seed}_train',train
                for mode in ('sampled','greedy'):
                    command=['scripts/eval_checkpoint.py',*common,'--checkpoint',str(checkpoint),
                             '--episodes','20','--seed','100','--out-dir',str(folder/'eval'),'--tag',f'seed{seed}_{mode}']
                    if mode=='sampled': command+=['--stochastic']
                    yield f'{variant}_{game}_{seed}_{mode}',command


def main():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    OUT.mkdir(parents=True,exist_ok=True)
    plan=list(jobs())
    protocol={'games':GAMES,'variants':VARIANTS,'training_seeds':[0,1,2],
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
              'primary_evaluation':'sampled','secondary_evaluation':'greedy',
              'sampling_seeds':list(range(100,120)), 'engine_layout_seeded':False,
              'action_grid':16,'memory_streak_cap':4,'source_informed_diagnosis':True,
              'policy_inputs':'official observation frames and past chosen actions only',
              'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                  'environment/arc_gym_env.py','rl_agents/ppo/action_memory.py',
                  'rl_agents/ppo/ppo_agent.py','rl_agents/ppo/novelty.py',
                  'rl_agents/dqn/q_network.py','scripts/run_baseline.py','scripts/eval_checkpoint.py')},
              'commands':dict(plan)}
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if any(previous.get(key)!=protocol[key] for key in ('commands','source_sha256')):
            raise RuntimeError('Existing protocol/code differs; choose a new experiment directory')
    protocol_path.write_text(json.dumps(protocol,indent=2))
    state={'completed':[],'total_jobs':len(plan),'pid':os.getpid()}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temp=OUT/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(OUT/'status.json')
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    try:
        for name,command in plan:
            marker=OUT/f'{name}.complete'
            if not marker.exists():
                log_path=OUT/f'{name}.log'
                if log_path.exists():
                    raise RuntimeError(f'Unfinished job {name}: preserve/archive partial outputs before restart')
                state.update(state='running',current=name);save()
                with log_path.open('x') as log:
                    result=subprocess.run([sys.executable,*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError(f'{name} exited {result.returncode}; inspect {log_path}')
                marker.touch()
            state['completed'].append(name);save()
        state.update(state='complete');save()
    except BaseException as error:
        state.update(state='failed',error=str(error));save()
        raise


if __name__=='__main__': main()
