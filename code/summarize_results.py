"""Collect measured metrics, without selecting on the test set."""
import csv
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
rows=[]
for path in sorted((root/'runs').glob('*/metrics.json')):
    m=json.loads(path.read_text())
    if path.parent.name.startswith('smoke'):
        continue
    rows.append(dict(run=path.parent.name,parameters=m['parameters'],seed=m['seed'],
                     train_targets=m['train_tokens'],final_validation_bpb=m['validation']['bpb'],
                     best_validation_bpb=m['best_validation_bpb'],best_step=m['best_step'],
                     train_seconds=m['train_seconds'],process_seconds=m['process_seconds'],
                     precision=m['precision'],device=m['device']))
with (root/'results/experiments.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
means={}
for label in ('baseline','gated'):
    values=[r['final_validation_bpb'] for r in rows if r['run'].startswith('paired-'+label)]
    means[label]={'mean':statistics.mean(values),'values':values}
summary={'experiments':rows,'paired':means,
         'paired_mean_reduction':means['baseline']['mean']-means['gated']['mean'],
         'paired_relative_reduction_percent':100*(1-means['gated']['mean']/means['baseline']['mean']),
         'total_training_targets_excluding_smoke':sum(r['train_targets'] for r in rows),
         'total_training_seconds_excluding_smoke':sum(r['train_seconds'] for r in rows),
         'total_process_seconds_excluding_smoke':sum(r['process_seconds'] for r in rows),
         'best_validation_run':min(rows,key=lambda r:r['best_validation_bpb'])['run']}
(root/'results/experiment_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
