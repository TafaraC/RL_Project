"""Aggregate original DQN/PPO evidence and create report figures; no training."""
import csv
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submission/WitsC/evidence'

def read(p): return list(csv.DictReader(p.open(newline='')))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for p in sorted((ROOT/'runs/dqn_archive').glob('eval*/**/episodes.csv')):
        data=read(p)
        rel=p.relative_to(ROOT/'runs/dqn_archive')
        target=OUT/'dqn'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
        rows.append(dict(experiment=p.parent.name,game=data[0]['game'],episodes=len(data),
            successes=sum(int(x['levels_completed'])>0 for x in data),
            levels=sum(int(x['levels_completed']) for x in data),wins=sum(int(x['won']) for x in data),
            actions=sum(int(x['actions_used']) for x in data),wall_seconds=max(float(x['wall_s']) for x in data),
            source=target.relative_to(OUT).as_posix(),
            limitation='Historical evaluation lacks config.json; mode/seed inferred from archive directory name.'))
    with (OUT/'dqn_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    training=[]
    stages={
        'DQN baseline':'runs/dqn_archive/runs_final',
        'DQN novelty':'runs/dqn_archive/runs_imp1',
        'DQN no-change memory':'runs/dqn_archive/runs_imp2',
        'DQN novelty + memory (ft09 only)':'runs/dqn_archive/runs_imp12',
        'PPO baseline':'runs/ppo_baseline_cuda_20261013',
        'PPO novelty':'runs/ppo_novelty_20261023/novelty',
        'PPO grid control':'runs/ppo_action_memory_20261023/grid_control',
        'PPO memory + novelty':'runs/ppo_action_memory_20261023/memory_novelty',
        'PPO memory only':'runs/ppo_action_memory_20261023/memory_only',
    }
    for name,folder in stages.items():
        for p in sorted((ROOT/folder).glob('*/*_seed*/episodes.csv')):
            data=read(p)
            if not data: continue
            target=OUT/'training'/name.replace(' ','_').replace('/','_')/p.parent.parent.name/p.parent.name
            target.mkdir(parents=True,exist_ok=True)
            for file in p.parent.iterdir():
                if file.name in ['episodes.csv','config.json','runtime.json']:
                    shutil.copyfile(file,target/file.name)
            training.append(dict(stage=name,game=data[0]['game'],seed=int(p.parent.name[-1]),
                episodes=len(data),decisions=int(data[-1]['env_steps_total']),
                wall_seconds=max(float(x['wall_s']) for x in data),
                training_successes=sum(int(x['levels_completed'])>0 for x in data),
                source=(target/'episodes.csv').relative_to(OUT).as_posix()))
    with (OUT/'training_cost.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(training[0]));w.writeheader();w.writerows(training)
    holdout=ROOT/'runs/final_holdout_20261027'
    for folder in ['runs_rand','runs_rand_mask']:
        for p in (ROOT/'runs/dqn_archive'/folder).rglob('*.csv'):
            target=OUT/'random'/p.relative_to(ROOT/'runs/dqn_archive')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    held=[]
    for p in sorted(holdout.glob('*_seed*/episodes.csv')):
        data=read(p)
        for game in ['cd82','tr87','vc33']:
            group=[r for r in data if r['game']==game]
            if len(group)!=5: raise ValueError(f'Incomplete holdout {p}: {game}')
            held.append(dict(stage=p.parent.name.rsplit('_seed',1)[0],seed=int(p.parent.name[-1]),
                game=game,episodes=len(group),successes=sum(int(r['levels_completed'])>0 for r in group),
                levels=sum(int(r['levels_completed']) for r in group),wins=sum(int(r['won']) for r in group),
                actions=sum(int(r['actions_used']) for r in group)))
    with (OUT/'holdout_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(held[0]));w.writeheader();w.writerows(held)
    shutil.copytree(ROOT/'reports/paper_track',OUT/'ppo',dirs_exist_ok=True)
    holdout=ROOT/'runs/final_holdout_20261027'
    if (holdout/'protocol.json').exists():
        shutil.copyfile(holdout/'protocol.json',OUT/'holdout_protocol.json')
        for p in holdout.glob('*_seed*/*.csv'):
            target=OUT/'holdout'/p.parent.name/p.name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    # Primary DQN eps0.3 results; historical greedy jobs have unequal counts.
    print('DQN ft09 sampled diagnostic (epsilon .3)')
    for name in ['dqn_base_plain','dqn_imp1','dqn_imp2','dqn_base_mem']:
        selected=[r for r in rows if r['experiment'].startswith(name+'_ft09') and r['experiment'].endswith('eps0.3')]
        print(name,[(r['experiment'],r['successes'],r['episodes']) for r in selected])
    print('Other DQN games:',[(r['experiment'],r['successes'],r['episodes']) for r in rows if r['game']!='ft09' and r['successes']])
    print('Training cost by stage (hours, decisions):')
    for stage in stages:
        subset=[r for r in training if r['stage']==stage]
        print(stage,round(sum(r['wall_seconds'] for r in subset)/3600,2),sum(r['decisions'] for r in subset))
    # Plot original training learning curves; never label shaped return official score.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figures=ROOT/'reports/course_report/images'
    figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for ax,prefix in zip(axes,['DQN','PPO']):
        for stage in stages:
            if not stage.startswith(prefix) or 'ft09 only' in stage: continue
            curves=[]
            for r in training:
                if r['stage']==stage and r['game']=='ft09':
                    data=read(OUT/r['source'])
                    ys=np.array([int(x['levels_completed'])>0 for x in data],dtype=float)
                    xs=np.array([int(x['env_steps_total']) for x in data])
                    curves.append((xs,np.cumsum(ys)/np.arange(1,len(ys)+1)))
            if curves:
                length=min(len(x) for x,y in curves)
                ys=np.stack([y[:length] for x,y in curves]);xs=curves[0][0][:length]
                ax.plot(xs/1000,ys.mean(0),label=stage.replace(prefix+' ',''))
                ax.fill_between(xs/1000,ys.min(0),ys.max(0),alpha=.12)
        ax.set(title=f'{prefix}: ft09 training',xlabel='Environment decisions (thousands)',ylim=(0,1))
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_ylabel('Cumulative episodes clearing a level')
    fig.savefig(figures/'learning_curves.png',dpi=190);plt.close(fig)
    labels=['Random','DQN base','DQN novelty','DQN memory','PPO base','PPO novelty','PPO grid','PPO memory+nov','PPO memory']
    values=[[5,9,8],[17,20,5],[7,1,5],[3,17,5],[2,0,3],[4,4,20],[0,9,1],[1,1,1],[2,0,4]]
    fig,ax=plt.subplots(figsize=(10,3.4),layout='constrained')
    for i,v in enumerate(values):
        rates=np.array(v)/20
        ax.scatter(np.repeat(i,3)+[-.12,0,.12],rates,marker='o',s=30,color='#267caa')
        ax.plot([i-.25,i+.25],[rates.mean()]*2,color='black',lw=2)
    ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right')
    ax.set(ylabel='Fraction of episodes clearing a level',ylim=(-.03,1.05),title='ft09: each dot is one training seed; bars are means')
    ax.grid(axis='y',alpha=.2)
    fig.savefig(figures/'evaluation.png',dpi=190);plt.close(fig)
    shutil.copyfile(ROOT/'reports/paper_track/ft09_diagnostic.png',figures/'diagnostic.png')

if __name__=='__main__': main()
