"""Check published derived artifacts using only the Python standard library."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
records = json.loads((ROOT / 'artifacts/source_manifest.json').read_text())
failures = []
for record in records:
    path = ROOT / record['path']
    if not path.is_file():
        failures.append(f'Missing: {record["path"]}')
        continue
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record['published_sha256']:
        failures.append(f'Hash mismatch: {record["path"]}')
    if path.suffix == '.gz':
        if hashlib.sha256(gzip.decompress(data)).hexdigest() != record['uncompressed_sha256']:
            failures.append(f'Uncompressed hash mismatch: {record["path"]}')
if failures:
    raise SystemExit('\n'.join(failures))
print(f'PASS: {len(records)} artifact/source hashes checked.')
