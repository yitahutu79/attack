"""Expand supplied CSVs for original analysis scripts; never overwrite existing CSVs."""
from pathlib import Path
import gzip

ROOT = Path(__file__).resolve().parents[1]
for archive in sorted(ROOT.rglob('*.csv.gz')):
    if '.git' in archive.parts or 'verification' in archive.parts:
        continue
    target = archive.with_suffix('')
    if target.exists():
        print(f'Skipped existing: {target.relative_to(ROOT)}')
        continue
    with target.open('xb') as handle:
        handle.write(gzip.decompress(archive.read_bytes()))
    print(f'Expanded: {target.relative_to(ROOT)}')
