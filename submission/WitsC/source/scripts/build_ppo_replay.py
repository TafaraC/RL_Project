"""Package recorded observations into the replay template; no simulation."""
import base64
import gzip
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from arc_agi.rendering import COLOR_MAP

root = Path('runs/ppo_visual_diagnostics_20261023')
names = ['ft09_seed2_sampled','ft09_seed2_greedy','ft09_seed0_sampled',
         'dc22_seed0_sampled','g50t_seed0_sampled','m0r0_seed0_sampled']
runs = [json.load(gzip.open(root/f'{name}.json.gz', 'rt')) for name in names]
data = {'palette': [COLOR_MAP[i] for i in range(16)], 'runs': runs}
encoded = base64.b64encode(gzip.compress(json.dumps(data,separators=(',',':')).encode())).decode()
template = Path('scripts/ppo_replay_template.html').read_text(encoding='utf-8')
dest = Path('C:/Users/pamnt/.codex/visualizations/2026/10/02/01a0fdec-4c43-7261-ae83-368405e5c19b/ppo-game-replay.html')
html = template.replace('__REPLAY_DATA__', encoded)
assert len(html.encode()) < 1_000_000
dest.write_text(html, encoding='utf-8')
# Actual recorded screenshots for review, including action lattice after level transition.
palette = np.array([[int(COLOR_MAP[i][j:j+2],16) for j in (1,3,5)] for i in range(16)],dtype=np.uint8)
sheet = Image.new('RGB',(960,376),'white')
draw = ImageDraw.Draw(sheet)
for col,(run_index,step,title) in enumerate([(1,10,'Greedy: repeat tile click'),(0,309,'Level 2: current 8x8 grid'),(0,309,'Level 2: proposed 16x16 grid')]):
    run=runs[run_index];row=run['steps'][step]
    grid=np.concatenate([np.full(n,c,dtype=np.uint8) for c,n in run['frames'][row['before']]]).reshape(64,64)
    image=Image.fromarray(palette[grid]).resize((320,320),Image.Resampling.NEAREST)
    pen=ImageDraw.Draw(image)
    if col:
        size=8 if col==1 else 16
        for y in range(size):
            for x in range(size):
                px=int((x+.5)*64/size)*5+2;py=int((y+.5)*64/size)*5+2
                pen.ellipse((px-2,py-2,px+2,py+2),fill='white',outline='black')
    else:
        x,y=[n*5+2 for n in row['xy']]
        pen.line((x-7,y-7,x+7,y+7),fill='white',width=3)
        pen.line((x-7,y+7,x+7,y-7),fill='white',width=3)
    sheet.paste(image,(col*320,36));draw.text((col*320+8,10),title,fill='black')
sheet.save(root/'ft09_diagnostic.png')
print(dest, len(html.encode()), 'bytes')
