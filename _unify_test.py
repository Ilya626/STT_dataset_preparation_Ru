import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from training.runpod_canary_launcher import find_manifest, normalize_manifest, split_manifest
base = Path('fine-tuning-dataset-20250913').resolve()
man = find_manifest(base, None)
out_dir = Path('training/_work').resolve()
out_dir.mkdir(parents=True, exist_ok=True)
normalized = out_dir / 'manifest.normalized.jsonl'
print('Manifest:', man)
print('Normalizing...')
print('Rows:', normalize_manifest(man, base, normalized))
tr = out_dir / 'train.jsonl'
va = out_dir / 'val.jsonl'
print('Split:', split_manifest(normalized, tr, va, 0.02, 42))
print('OK')
