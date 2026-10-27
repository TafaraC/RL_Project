"""Execute real notebook commit-mode inference/parquet with local path substitutions.

Only the offline wheel installation is skipped (packages already installed).
This is not a Kaggle Linux/gateway verification.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def main():
    sandbox=ROOT/'runs/notebook_commit_test'
    competition=sandbox/'competition'
    working=sandbox/'working'
    working.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'runs/kaggle_reference/ARC-AGI-3-Agents-main',
                    competition/'ARC-AGI-3-Agents',dirs_exist_ok=True)
    notebook=json.loads((ROOT/'submission/WitsC/kaggle/dqn_baseline.ipynb').read_text())
    assets=sandbox/'assets'
    scope={}
    extraction=notebook['cells'][1]['source'].replace("Path('/tmp/witsc_assets')",f'Path({str(assets)!r})')
    extraction=extraction.replace("Path('/kaggle/input')",f'Path({str(ROOT / "kaggle/assets")!r})')
    exec(extraction,scope)
    runtime=(assets/'runtime.py').read_text()
    for original,replacement in {
        '/kaggle/working':working,
        '/kaggle/input/competitions/arc-prize-2026-arc-agi-3':competition,
        '/tmp/witsc_framework':sandbox/'framework',
        '/tmp/witsc_assets':assets,
    }.items():
        runtime=runtime.replace(original,replacement.as_posix())
    original_run=subprocess.run
    skipped=[]
    def installed(cmd,**kwargs):
        if cmd[1:4]==['-m','pip','install']:
            assert '--no-index' in cmd
            skipped.append(cmd)
            return subprocess.CompletedProcess(cmd,0)
        return original_run(cmd,**kwargs)
    os.environ.pop('KAGGLE_IS_COMPETITION_RERUN',None)
    with patch('subprocess.run',installed):
        exec(compile(runtime,'runtime.py','exec'),{})
    import pandas as pd
    frame=pd.read_parquet(working/'submission.parquet')
    assert list(frame.columns)==['row_id','game_id','end_of_game','score']
    assert frame.to_dict('records')==[dict(row_id='1_0',game_id='1',end_of_game=True,score=1)]
    result=dict(status='passed',notebook='dqn_baseline.ipynb',
        executed='embedded source extraction, attached model discovery/hash, real framework import, GPU model load/action and actual parquet write/read',
        substitutions='Kaggle /tmp and /kaggle paths mapped into runs/notebook_commit_test',
        installation='Skipped offline pip install; used locally installed packages',
        kaggle_linux_verified=False,kaggle_gateway_verified=False,output_schema=list(frame.columns))
    (ROOT/'submission/WitsC/validation/notebook_commit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
