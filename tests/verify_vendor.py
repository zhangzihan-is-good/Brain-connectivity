"""Verify the selected, unmodified vendored IC source files offline."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def verify():
    manifest = json.loads((ROOT / '_vendor' / 'IC_PROVENANCE.json').read_text(encoding='utf8'))
    vendor = (ROOT / '_vendor' / 'IC').resolve()
    rows = []
    expected = set()
    for row in manifest['files']:
        path = (vendor / row['path']).resolve()
        if not path.is_relative_to(vendor) or path == vendor:
            raise ValueError('Manifest path escapes the selected vendor directory')
        data = path.read_bytes()
        if blob(data) != row['blob_sha']:
            raise ValueError('Selected upstream file was changed: ' + row['path'])
        expected.add(path)
        rows.append(dict(path=row['path'], role=row.get('role'), git_blob_sha1=blob(data),
                         sha256=hashlib.sha256(data).hexdigest()))
    actual = {path.resolve() for path in vendor.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts}
    if actual != expected:
        raise ValueError('Vendor files do not match the selected-file manifest')
    return dict(status='PASS', commit=manifest['commit'], selection=manifest['selection'], files=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON report; no report file is written by default')
    args = parser.parse_args()
    result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(result['status'], len(result['files']), 'selected unmodified upstream files', result['commit'])
