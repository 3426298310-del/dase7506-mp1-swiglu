"""Measure the unchanged evaluator in fresh processes; no scoring logic here."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import psutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output-dir', type=Path, default=Path('results/benchmark'))
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for repeat in range(a.repeats):
        # Alternate order to reduce systematic warmup / temperature bias.
        cases = [('baseline', a.baseline), ('candidate', a.candidate)]
        if repeat % 2:
            cases.reverse()
        for label, checkpoint in cases:
            output = a.output_dir/f'{label}-{repeat}.json'
            command = [sys.executable, 'evaluate.py', '--checkpoint', checkpoint,
                       '--device', 'cpu', '--precision', 'fp32', '--threads', '4',
                       '--split', a.split, '--output', str(output)]
            peak = 0
            max_processes = 0
            start = time.perf_counter()
            with output.with_suffix('.log').open('w', encoding='utf-8') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform=='win32' else 0)
                watched = psutil.Process(proc.pid)
                while proc.poll() is None:
                    try:
                        # Windows venv python.exe is a small launcher; torch runs
                        # in a descendant. Include the entire evaluator tree.
                        processes = [watched] + watched.children(recursive=True)
                        total_peak = 0
                        for process in processes:
                            try:
                                memory = process.memory_info()
                                total_peak += max(memory.rss, getattr(memory, 'peak_wset', 0))
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        max_processes = max(max_processes, len(processes))
                        peak = max(peak, total_peak)
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(.02)
                if proc.returncode:
                    raise RuntimeError(f'Failed: {command}; see {output.with_suffix(".log")}')
            score = json.loads(output.read_text())
            row = dict(label=label, repeat=repeat, bpb=score['bpb'],
                       scoring_seconds=score['seconds'], process_seconds=time.perf_counter()-start,
                       peak_process_ram_bytes=peak, max_evaluator_processes=max_processes, checkpoint=checkpoint)
            records.append(row)
            print(json.dumps(row), flush=True)
    summary = {'split':a.split, 'threads':4, 'precision':'fp32',
               'ram_method':'Conservative sum of per-process peak working sets across launcher and all evaluator descendants, sampled every 20 ms.',
               'records':records}
    for label in ('baseline', 'candidate'):
        rows = [r for r in records if r['label'] == label]
        summary[label] = dict(median_scoring_seconds=statistics.median(r['scoring_seconds'] for r in rows),
                              max_peak_process_ram_bytes=max(r['peak_process_ram_bytes'] for r in rows),
                              bpb=rows[0]['bpb'])
    summary['time_ratio'] = summary['candidate']['median_scoring_seconds']/summary['baseline']['median_scoring_seconds']
    summary['time_pass'] = summary['time_ratio'] <= 5
    summary['ram_pass'] = summary['candidate']['max_peak_process_ram_bytes'] <= 4*1024**3
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
