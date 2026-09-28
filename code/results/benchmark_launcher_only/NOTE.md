# Superseded resource audit

The BPB values in this directory are valid repeated scores of the same frozen
checkpoint, but the RAM readings are NOT valid. The original monitor sampled only
the Windows venv launcher (about 4 MiB), missing its Python child that loaded torch.
The corrected benchmark.py includes all descendants. Use ../benchmark/summary.json
for the final time and memory audit. These original logs are retained for transparency.
No model, checkpoint, training choice, data or evaluator was changed in response.
