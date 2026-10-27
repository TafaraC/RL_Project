"""Build self-contained notebooks from our frozen PPO models. No credentials."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'submission/WitsC/kaggle'
MODELS = {
    'ppo_baseline': ('runs/ppo_baseline_cuda_20261013/ft09/ppo_seed2/ppo.pt', 8, False),
    'ppo_novelty': ('runs/ppo_novelty_20261023/novelty/ft09/ppo_seed2/ppo.pt', 8, False),
    'ppo_grid_control': ('runs/ppo_action_memory_20261023/grid_control/ft09/ppo_seed1/ppo.pt', 16, False),
    'ppo_memory_novelty': ('runs/ppo_action_memory_20261023/memory_novelty/ft09/ppo_seed0/ppo.pt', 16, True),
    'ppo_memory_only': ('runs/ppo_action_memory_20261023/memory_only/ft09/ppo_seed2/ppo.pt', 16, True),
}

def cell(source, kind='code'):
    result = dict(cell_type=kind, metadata={}, source=source)
    if kind == 'code':
        result.update(outputs=[], execution_count=None)
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name, (source, grid, memory) in MODELS.items():
        checkpoint = torch.load(ROOT/source, map_location='cpu', weights_only=True)
        assert checkpoint['n_actions'] == 7+grid**2
        assert checkpoint['obs_shape'][0] == (80 if memory else 64)
        arrays = {k: v.detach().cpu().numpy() for k,v in checkpoint['network'].items()}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        config = dict(name=name, algorithm='ppo', click_grid=grid, action_memory=memory,
                      training_game='ft09', source=source,
                      source_sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
                      seed=checkpoint['cfg']['seed'], sampling_seed=100, max_actions=1000,
                      weights_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                      selection='Development first-level success; tie broken by lowest seed. No held-out selection.')
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('weights.npz', buffer.getvalue())
            z.writestr('model.json', json.dumps(config, indent=2))
            for f in ['my_agent.py', 'runtime.py']:
                z.writestr(f, (ROOT/'kaggle'/f).read_bytes())
        payload = archive.getvalue()
        encoded = base64.b64encode(payload).decode('ascii')
        extraction = ('import base64, hashlib, io, zipfile\nfrom pathlib import Path\n'
            f'payload = base64.b64decode({encoded!r})\n'
            f'assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}\n'
            "root = Path('/tmp/witsc_assets')\nroot.mkdir(parents=True, exist_ok=True)\n"
            "with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n"
            "    assert set(archive.namelist()) == {'weights.npz', 'model.json', 'my_agent.py', 'runtime.py'}\n"
            "    archive.extractall(root)\n"
            "del payload\n")
        nb = dict(nbformat=4, nbformat_minor=4,
            metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                kaggle=dict(accelerator='nvidiaTeslaT4', isInternetEnabled=False, isGpuEnabled=True,
                            language='python', sourceType='notebook')),
            cells=[cell(f'# WitsC: {name}\n\nSelf-contained frozen PPO. Attach ARC-AGI-3 competition data; enable T4 GPU; internet OFF. '
                'Weights and inference source are embedded. Save & Run All tests CUDA inference and emits the official starter placeholder; '
                'only a competition rerun produces a genuine score.\n\n'
                'Original research: https://github.com/TafaraC/RL_Project . No online learning or game-specific routing.\n', 'markdown'),
                cell(extraction), cell("exec(compile((root/'runtime.py').read_text(), str(root/'runtime.py'), 'exec'))")])
        path = OUT/f'{name}.ipynb'
        path.write_text(json.dumps(nb, ensure_ascii=True), encoding='utf-8')
        # Human-readable assets alongside notebook, useful for audit/testing.
        assets = OUT/'assets'/name
        assets.mkdir(parents=True, exist_ok=True)
        (assets/'weights.npz').write_bytes(buffer.getvalue())
        (assets/'model.json').write_text(json.dumps(config, indent=2)+'\n')
        config['notebook_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifests[name] = config
        print(name, path.stat().st_size)
    (OUT/'manifest.json').write_text(json.dumps(manifests, indent=2)+'\n')

if __name__ == '__main__':
    main()
