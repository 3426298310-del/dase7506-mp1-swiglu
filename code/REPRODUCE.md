# Reproduction

Run commands from this `code/` directory. Python 3.12 is required.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Choose ONE torch install, according to hardware:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
# For NVIDIA CUDA training, use this instead:
# python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt -r requirements-tools.txt
python -m unittest discover -s tests -v
python verify_integrity.py
```

Linux activation: `source .venv/bin/activate`. Other commands are the same.
CPU scoring does not need CUDA. GPU-trained checkpoints load on CPU.
The actual environments are recorded in `results/environment-cpu.txt` and
`results/environment-gpu.txt`.

## Controlled training

The command below records the completed historical GPU experiments. It is not
required to reproduce the submitted score. After the reported system interruption,
this machine was not used for further GPU training; use the CPU-only score command
below. If retraining is needed here, use `--device cpu --precision fp32`.

```powershell
python run_experiments.py --device cuda --precision bf16 --seeds 17 29 --steps 1200 --tag paired
```

This runs baseline and gated model at each seed, from random initialization.
All four process 9,830,400 training targets apiece. Use the final checkpoint in
each run for the fixed-budget table; intermediate checkpoint selection is a
separate procedure. `results/run_ledger.json` records exact commands and costs.
On a CPU-only machine, `--device cpu --precision fp32` works but training and
scores can differ slightly from the GPU BF16 training runs.

The supplied baseline also serves as the gate-removal ablation. To run the
explicit disabled-gate branch, use `--implementation student --config
configs/ablation.json` with `train.py` and a fresh output directory.

## Evaluation without retraining

The final bundle provides `submission/model.pt` and all source files.

```powershell
python evaluate.py --checkpoint submission/model.pt --device cpu --precision fp32 --threads 4 --split test --output results/reproduced-test.json
```

Report the `bpb` field from this complete-test output, not `token_ppl`.
No network access is required after installing dependencies and obtaining the bundle.

## Resource measurements

```powershell
python benchmark.py --baseline runs/paired-baseline-s17/checkpoint.pt --candidate submission/model.pt --split test --repeats 3 --output-dir results/benchmark
```

Run only after freezing the method. This launches the unchanged scorer in fresh
processes, alternating baseline/candidate order. On Windows both processes have
Below Normal priority to reduce desktop disruption. Both use four CPU threads and
FP32. The reported ratio uses median scoring time, excluding loading, from three
trials. RAM includes loading and the whole evaluator process tree, not just the
Windows virtual-environment launcher. Every 20 ms, psutil sums the peak working
sets (or current RSS if larger) of the launcher and all descendants; the maximum
sum is a conservative memory estimate. This is a
local audit, not a claim that the instructor's hardware will have identical timing.

`results/freeze.json` records the final predictor and source hashes before test
evaluation. `submission/ASSET_MANIFEST.json` records all submitted inference files.

## Final selection and interrupted extension

The submitted model is `runs/paired-gated-s29/best.pt`, selected at update 1200
using validation BPB only, copied without changes to `submission/model.pt`.
All four paired runs finished. A subsequent 4800-step experiment was launched but
produced no training metrics or checkpoint before a reported black screen/system
restart. Its ledger entry is marked interrupted, not completed. No further GPU
training was attempted; remaining audits use the separate `.venv-cpu` environment.
No claim about the cause of the system failure is made. Reproduction of the
submitted score does not require GPU training.
