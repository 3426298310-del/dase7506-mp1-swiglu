"""Verify fixed course files against original package hashes (no torch needed)."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root/'PACKAGE_MANIFEST.json').read_text())
fixed = ['common.py','evaluate.py','model.py','configs/baseline.json','tests/test_contract.py']
fixed += [p.relative_to(root).as_posix() for p in (root/'data').iterdir() if p.is_file()]
for name in fixed:
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    expected = manifest['code/'+name]
    if actual != expected:
        raise SystemExit(f'CHANGED protected file: {name}')
    print(f'OK {name}')
print(f'All {len(fixed)} protected files match the supplied package.')
