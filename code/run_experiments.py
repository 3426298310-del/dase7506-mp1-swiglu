"""Run paired experiments sequentially; never evaluates the test split."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', default='bf16')
    p.add_argument('--seeds', nargs='+', type=int, default=[17, 29])
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--only', choices=['both', 'baseline', 'gated'], default='both')
    p.add_argument('--tag', default='paired')
    a = p.parse_args()
    Path('results').mkdir(exist_ok=True)
    ledger_path = Path('results/run_ledger.json')
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    for seed in a.seeds:
        for label in ('baseline', 'gated'):
            if a.only != 'both' and label != a.only:
                continue
            run = Path(f'runs/{a.tag}-{label}-s{seed}')
            if (run/'metrics.json').exists():
                print(f'Already completed: {run}', flush=True)
                continue
            command = [sys.executable, 'train.py', '--implementation', 'model' if label=='baseline' else 'student',
                       '--config', f'configs/{"baseline" if label=="baseline" else "gated"}.json',
                       '--seed', str(seed), '--steps', str(a.steps), '--batch-size', '32',
                       '--device', a.device, '--precision', a.precision, '--threads', '4',
                       '--eval-every', '300', '--run-dir', str(run)]
            logpath = Path(f'results/{run.name}.log')
            row = {'run':str(run),'command':command,'started_unix':time.time(),'status':'running'}
            ledger.append(row)
            ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
            print(f'Starting {run}',flush=True)
            with logpath.open('w', encoding='utf-8') as log:
                status = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
            row.update(status='completed' if status==0 else 'failed',returncode=status,
                       elapsed_seconds=time.time()-row['started_unix'])
            ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
            if status:
                raise RuntimeError(f'{run} failed; inspect {logpath}')
            metrics=json.loads((run/'metrics.json').read_text())
            print(json.dumps({'run':str(run),'validation_bpb':metrics['validation']['bpb'],
                              'best_validation_bpb':metrics['best_validation_bpb'],
                              'train_seconds':metrics['train_seconds']}),flush=True)


if __name__ == '__main__':
    main()
