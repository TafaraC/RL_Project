"""Build the cleaned v2 Moodle ZIP and verify every delivered file and model."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'submission/v2'
OUT = V2 / 'WitsC'


def main():
    shutil.copyfile(V2 / 'report_source/main.pdf', OUT / 'WitsC_Report.pdf')
    files = sorted(p for p in OUT.rglob('*') if p.is_file()
                   and not {'__pycache__', '.pytest_cache'}.intersection(p.parts)
                   and p.suffix not in {'.pyc', '.pyo'} and p.name != 'SHA256SUMS.txt')
    assert all(p.name != '.env' and 'environment_files' not in p.parts for p in files)
    words = len((OUT / 'paper_track_writeup.md').read_text().split())
    assert words < 1500
    checked = []
    for notebook in sorted((OUT / 'kaggle').glob('*.ipynb')):
        nb = json.loads(notebook.read_text())
        assert nb['metadata']['kaggle']['isGpuEnabled']
        assert not nb['metadata']['kaggle']['isInternetEnabled']
        assert notebook.stat().st_size < 1_000_000
        cell = nb['cells'][1]['source']
        tree = ast.parse(cell, feature_version=(3, 12))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets))
        payload = base64.b64decode(ast.literal_eval(assignment.value.args[0]), validate=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as embedded:
            config = json.loads(embedded.read('model.json'))
            asset = OUT / 'kaggle/model_dataset' / notebook.stem
            assert json.loads((asset / 'model.json').read_text()) == config
            for name in ['my_agent.py', 'runtime.py']:
                assert embedded.read(name) == (OUT / 'source/kaggle' / name).read_bytes()
                ast.parse(embedded.read(name).decode(), feature_version=(3, 12))
            # Original validated export provides an independent expected weight digest.
            manifest = json.loads((ROOT / 'submission/WitsC/kaggle/manifest.json').read_text())
            assert hashlib.sha256((asset / 'weights.npz').read_bytes()).hexdigest() == manifest[notebook.stem]['weights_sha256']
            assert hashlib.sha256(notebook.read_bytes()).hexdigest() == manifest[notebook.stem]['notebook_sha256']
        checked.append(notebook.stem)
    assert len(checked) == 9
    sums = ''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.relative_to(OUT).as_posix() + '\n' for p in files)
    archive = V2 / 'WitsC.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, 'WitsC/' + p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        for line in sums.splitlines():
            digest, name = line.split('  ', 1)
            assert hashlib.sha256(z.read('WitsC/' + name)).hexdigest() == digest
    result = dict(status='passed', files=len(files), paper_words=words,
                  notebooks=checked, bytes=archive.stat().st_size,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  checks=['ZIP CRC', 'every delivered file SHA256', 'nine notebook and weight hashes',
                          'embedded source parity', 'Python 3.12 syntax', 'CUDA/offline metadata'],
                  unverified=['Kaggle Linux installation', 'live gateway', 'official score', 'Paper Track publication'])
    (ROOT / 'runs/v2_package_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
