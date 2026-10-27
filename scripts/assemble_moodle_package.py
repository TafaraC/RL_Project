"""Assemble an allowlisted, credential-free Moodle ZIP with integrity hashes."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submission/WitsC'

def copy_tree(source,dest,suffixes):
    for p in source.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in suffixes:
            target=dest/p.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)

def main():
    report=ROOT/'reports/course_report'
    assert (report/'latex/main.pdf').is_file()
    (OUT/'report').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(report/'latex/main.pdf',OUT/'report/WitsC_Report.pdf')
    copy_tree(report/'latex',OUT/'report/latex',{'.tex','.sty','.bst'})
    copy_tree(report/'images',OUT/'report/images',{'.png'})
    shutil.copyfile(report/'TEMPLATE_PROVENANCE.md',OUT/'report/TEMPLATE_PROVENANCE.md')
    copy_tree(ROOT/'reports/paper_track',OUT/'paper_track',{'.md','.csv','.png'})
    for folder in ['environment','rl_agents','scripts','tests','evaluation']:
        copy_tree(ROOT/folder,OUT/'source'/folder,{'.py','.html'})
    for name in ['README.md','SPLIT.md','requirements.txt','requirements-cuda.txt','pytest.ini']:
        shutil.copyfile(ROOT/name,OUT/'source'/name)
    copy_tree(ROOT/'kaggle',OUT/'source/kaggle',{'.py','.md'})
    shutil.copyfile(ROOT/'kaggle/README.md',OUT/'kaggle/EXECUTION_GUIDE.md')
    # All model parameters are embedded in notebooks; duplicate weight assets
    # are retained in the repository's kaggle/assets, not the Moodle package.
    assert not (OUT/'kaggle/assets').exists()
    for p in OUT.rglob('*'):
        if p.is_file():
            if p.name == '.env' or p.suffix in ['.whl','.pt'] or 'environment_files' in p.parts:
                raise ValueError(f'Unexpected private/platform-specific artifact: {p}')
    sums=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS.txt':
            sums.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(OUT).as_posix())
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n')
    archive=ROOT/'submission/WitsC.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,Path('WitsC')/p.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert 'WitsC/report/WitsC_Report.pdf' in z.namelist()
        assert 'WitsC/kaggle/dqn_baseline.ipynb' in z.namelist()
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(sums),
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
