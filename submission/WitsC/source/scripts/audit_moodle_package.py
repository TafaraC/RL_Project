"""Validate the final ZIP, small notebooks and attached model dataset."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    archive=ROOT/'submission/WitsC.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert len(names)==len(set(names))
        assert all(n.startswith('WitsC/') and '..' not in Path(n).parts for n in names)
        assert not any('/.env' in n or '/environment_files/' in n or n.endswith(('.pt','.whl')) for n in names)
        for line in z.read('WitsC/SHA256SUMS.txt').decode().splitlines():
            expected,name=line.split('  ',1)
            assert hashlib.sha256(z.read('WitsC/'+name)).hexdigest()==expected,name
        manifest=json.loads(z.read('WitsC/kaggle/manifest.json'))
        dataset=zipfile.ZipFile(io.BytesIO(z.read('WitsC/kaggle/model_dataset.zip')))
        assert dataset.testzip() is None
        assert 'witsc_models_manifest.json' in dataset.namelist()
        checks=[]
        for name,config in manifest.items():
            raw=z.read(f'WitsC/kaggle/{name}.ipynb')
            assert len(raw)<1_000_000
            assert hashlib.sha256(raw).hexdigest()==config['notebook_sha256']
            nb=json.loads(raw)
            assert nb['metadata']['kaggle']['isGpuEnabled'] is True
            assert nb['metadata']['kaggle']['isInternetEnabled'] is False
            for cell in nb['cells']:
                if cell['cell_type']=='code': ast.parse(cell['source'],feature_version=(3,12))
            tree=ast.parse(nb['cells'][1]['source'])
            assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
            payload=base64.b64decode(ast.literal_eval(assignment.value.args[0]),validate=True)
            with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                assert set(inner.namelist())=={'model.json','my_agent.py','runtime.py'}
                model=json.loads(inner.read('model.json'))
                assert model['name']==name
                assert hashlib.sha256(dataset.read(name+'/weights.npz')).hexdigest()==config['weights_sha256']
                for file in ['my_agent.py','runtime.py']:
                    ast.parse(inner.read(file).decode(),feature_version=(3,12))
                    assert inner.read(file)==z.read('WitsC/source/kaggle/'+file)
            checks.append(name)
        evidence=json.loads(z.read('WitsC/COMPETITION_EVIDENCE.json'))
        assert evidence['nominated_notebook']=='kaggle/dqn_baseline.ipynb'
        assert evidence['official_score'] is None
        words=len(z.read('WitsC/paper_track/WRITEUP.md').decode().split())
        assert words<1500,words
        result=dict(status='passed',files=len(names),notebooks=checks,paper_words=words,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            checked=['ZIP CRC','all file SHA256 hashes','notebook SHA256','attached weight SHA256','notebook source below 1 MB',
                     'Python 3.12 syntax','GPU/offline metadata','embedded source parity',
                     'safe file paths','no local wheels/checkpoint pickle/game-source directories'],
            does_not_verify=['Kaggle Linux dependency installation','live competition gateway','official score'])
        (ROOT/'submission/WitsC_PACKAGE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
