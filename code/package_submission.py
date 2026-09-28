"""Create reproducible source and matching checkpoint ZIPs after final audits."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
out=ROOT.parent/'deliverables'
out.mkdir(exist_ok=True)
freeze=json.loads((ROOT/'results/freeze.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/'submission/model.pt')==freeze['checkpoint_sha256']
for name,expected in freeze['source_sha256'].items():
    assert sha(ROOT/name)==expected, f'Frozen source changed: {name}'
report=ROOT/'output/pdf/MP1_Report.pdf'
assert report.is_file(), 'Report missing'
audit=json.loads((ROOT/'results/benchmark/summary.json').read_text())
assert audit['time_pass'] and audit['ram_pass']
assert json.loads((ROOT/'submission/ASSET_MANIFEST.json').read_text())['pass']
files=[]
for p in ROOT.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if any(part.startswith('.venv') or part in ('downloads','__pycache__','.git','tmp') for part in rel.parts):continue
    if rel.suffix in ('.pyc','.pt'):continue
    if rel.parts[0]=='output' and p!=report:continue
    files.append(p)
with zipfile.ZipFile(out/'MP1_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,'code/'+p.relative_to(ROOT).as_posix())
    z.write(ROOT.parent/'GUIDE.md','GUIDE.md')
with zipfile.ZipFile(out/'MP1_checkpoints.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for rel in ['submission/model.pt','submission/ASSET_MANIFEST.json','runs/paired-baseline-s17/checkpoint.pt']:
        z.write(ROOT/rel,'code/'+rel)
record={'source_zip':sha(out/'MP1_source.zip'),'checkpoint_zip':sha(out/'MP1_checkpoints.zip'),
        'final_checkpoint_sha256':freeze['checkpoint_sha256'],'final_test_bpb':audit['candidate']['bpb']}
(out/'SHA256_AND_SCORE.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
