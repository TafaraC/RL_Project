"""Read-only action coverage audit using a recorded policy trajectory."""
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_baseline import make_env
from arc_agi import Arcade, OperationMode

out = Path('runs/ppo_visual_diagnostics_20261023')
trace = json.load(gzip.open(out/'ft09_seed2_sampled.json.gz', 'rt'))
env = make_env(Arcade(operation_mode=OperationMode.OFFLINE), 'ft09',
               SimpleNamespace(click_grid=8, history_len=4, max_actions=1000), 100)
env.reset(seed=100)
results = []

def audit():
    game = env.arc_env._game
    coverage = {}
    for size in (8, 16, 64):
        hits, tiles = [], set()
        for iy in range(size):
            for ix in range(size):
                x, y = int((ix+.5)*64/size), int((iy+.5)*64/size)
                point = game.camera.display_to_grid(x, y)
                if point:
                    for tag in ('Hkx', 'NTi'):
                        tile = game.current_level.get_sprite_at(*point, tag)
                        if tile:
                            hits.append([x,y])
                            tiles.add((tile.x,tile.y))
                            break
        coverage[size] = {'clicks_hitting_tiles': len(hits), 'reachable_tiles': len(tiles), 'clicks': hits}
    results.append({'level_index': int(game.level_index), 'total_tiles': len(game.fhc+game.mou), 'coverage': coverage})

audit()
for row in trace['steps']:
    _, _, term, trunc, info = env.step(row['action'])
    if info['levels_completed'] == 1:
        audit()
        break
(out/'ft09_click_coverage.json').write_text(json.dumps(results, indent=2))
print(json.dumps([{**r, 'coverage': {k: {a:b for a,b in v.items() if a!='clicks'} for k,v in r['coverage'].items()}} for r in results], indent=2))
