"""Freeze a validation-selected checkpoint BEFORE test scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reason',required=True)
    a=p.parse_args()
    frozen=ROOT/'results/freeze.json'
    if frozen.exists():
        raise SystemExit('A frozen method already exists; refusing to overwrite it.')
    run=a.run if a.run.is_absolute() else ROOT/a.run
    metrics=json.loads((run/'metrics.json').read_text())
    target=ROOT/'submission/model.pt'
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(run/'best.pt',target)
    source_names=['student.py','model.py','common.py','evaluate.py','train.py',
                  'configs/gated.json','configs/ablation.json','configs/baseline.json']
    record={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'selected_run':run.relative_to(ROOT).as_posix(),'selected_checkpoint':'best.pt',
            'selection_reason':a.reason,'best_step':metrics['best_step'],
            'validation_bpb':metrics['best_validation_bpb'],
            'total_run_train_targets':metrics['train_tokens'],
            'selected_checkpoint_train_targets':metrics['best_step']*32*256,
            'checkpoint_sha256':digest(target),
            'source_sha256':{name:digest(ROOT/name) for name in source_names},
            'test_used_for_selection':False}
    frozen.parent.mkdir(exist_ok=True)
    frozen.write_text(json.dumps(record,indent=2)+'\n')
    # Conservative: count both model modules, the tokenizer and all model configs.
    names=['submission/model.pt','student.py','model.py','data/tokenizer.json',
           'configs/gated.json','configs/ablation.json','configs/baseline.json']
    assets=[{'path':name,'bytes':(ROOT/name).stat().st_size,'sha256':digest(ROOT/name)} for name in names]
    total=sum(row['bytes'] for row in assets)
    asset_record={'files':assets,'total_bytes':total,'limit_bytes':64*1024**2,
                  'pass':total<=64*1024**2,
                  'scope':'Checkpoint and model-specific inference files. Standard software dependencies, benchmark text, and audit metadata excluded.'}
    (ROOT/'submission/ASSET_MANIFEST.json').write_text(json.dumps(asset_record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if not asset_record['pass']:
        raise SystemExit('Asset budget exceeded.')


if __name__=='__main__': main()
