"""Executed by notebook after embedded assets are verified and extracted."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

WORK = Path('/kaggle/working')
INPUT = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3')
if not INPUT.exists():
    INPUT = Path('/kaggle/input/arc-prize-2026-arc-agi-3')
if not INPUT.exists():
    raise FileNotFoundError('Attach the ARC Prize 2026 ARC-AGI-3 competition data')

subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links',
                str(INPUT / 'arc_agi_3_wheels'), 'arc-agi', 'python-dotenv'], check=True)

framework = Path('/tmp/witsc_framework')
shutil.copytree(INPUT / 'ARC-AGI-3-Agents', framework, dirs_exist_ok=True)
assets = Path('/tmp/witsc_assets')
shutil.copyfile(assets / 'my_agent.py', framework / 'agents/templates/my_agent.py')
(framework / 'agents/__init__.py').write_text('''from .agent import Agent, Playback
from .swarm import Swarm
from .templates.my_agent import MyAgent
AVAILABLE_AGENTS = {'myagent': MyAgent}
''')
os.environ.update(WITSC_ASSETS=str(assets), WITSC_DEVICE='cuda', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.path.insert(0, str(framework))
import torch
torch.set_num_threads(2)
from agents.templates.my_agent import MyAgent
from arcengine import FrameData, GameState

# Test actual imports, checkpoint loading, GPU execution and action legality
# on every Save & Run All, rather than only writing a placeholder parquet.
probe = MyAgent(card_id='preflight', game_id='preflight', agent_name='myagent',
                ROOT_URL='', record=False, arc_env=None)
frame = FrameData(frame=[[[0]*64 for _ in range(64)]], state=GameState.NOT_FINISHED,
                  levels_completed=0, available_actions=[1, 2, 3, 4, 5, 6, 7])
action = probe.choose_action([], frame)
assert action.value in frame.available_actions
print(json.dumps({'preflight': 'passed', 'torch': torch.__version__, 'python': sys.version,
                  'gpu': torch.cuda.get_device_name(), 'model': probe.config}))
del probe
torch.cuda.empty_cache()

if os.environ.get('KAGGLE_IS_COMPETITION_RERUN'):
    deadline = time.monotonic()+600
    while True:
        try:
            with urllib.request.urlopen('http://gateway:8001/api/games', timeout=10) as response:
                games = json.load(response)
            if not games:
                raise RuntimeError('Gateway returned an empty game list')
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)
    (framework / '.env').write_text('SCHEME=http\nHOST=gateway\nPORT=8001\n'
        'ARC_API_KEY=test-key-123\nARC_BASE_URL=http://gateway:8001/\n'
        'OPERATION_MODE=online\nENVIRONMENTS_DIR=\nRECORDINGS_DIR=/tmp/witsc_recordings\n')
    # Official runner is multithreaded. Make worker errors fatal instead of
    # silently producing an apparently successful process exit.
    (framework / 'witsc_run.py').write_text('''import os, threading, traceback
def fail(args):
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    os._exit(1)
threading.excepthook = fail
import main
main.main()
''')
    subprocess.run([sys.executable, 'witsc_run.py', '--agent', 'myagent'], cwd=framework,
                   env=dict(os.environ, MPLBACKEND='agg'), check=True, timeout=36000)
    print('Official runner finished. Competition gateway owns scored submission.parquet generation.')
else:
    # Exact official-starter commit-mode schema. Not a score or game result.
    import pandas as pd
    pd.DataFrame([['1_0', '1', True, 1]], columns=[
        'row_id', 'game_id', 'end_of_game', 'score']).to_parquet(WORK/'submission.parquet', index=False)
    print('Save & Run All preflight complete; placeholder parquet is NOT an official score.')
