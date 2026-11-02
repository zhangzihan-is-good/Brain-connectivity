"""Check or explicitly refresh code/configuration integrity (standard library).

Run from Merge_code: python ABIDE/tools/integrity.py --check
After reviewed source/config edits only: python ABIDE/tools/integrity.py --refresh
Execution entry points only check; they never refresh their own hash contract.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
MERGE = BASE.parent
MANIFEST = BASE/'configs/integrity/active_manifest.json'
SHARED = ('rgic_backend.py','rgic_shared.py','rgic_profiles_abide.py',
          'io_shared.py','shared_paths.py','estimators_shared.py','_vendor/IC_PROVENANCE.json')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def inventory():
    files = [p for p in BASE.rglob('*') if p.is_file() and p != MANIFEST
             and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')]
    for path in files:
        relative = path.relative_to(BASE)
        if relative.parts[0] in ('results','figures','report','audit','archive','_packaging_work'):
            raise RuntimeError('Existing output/archive is not an active input: '+str(relative))
        if path.stem.lower() == 'readme' or path.suffix not in ('.py','.json','.txt'):
            raise RuntimeError('Only active code/configuration is allowed: '+str(relative))
    return sorted(p.relative_to(BASE).as_posix() for p in files)

def dependencies():
    return sorted((*SHARED, *(p.relative_to(MERGE).as_posix()
        for p in (MERGE/'_vendor/IC/rgic').rglob('*.py'))))

def entry(root, relative):
    path = (root/relative).resolve(); path.relative_to(root)
    return dict(path=relative, bytes=path.stat().st_size, sha256=sha(path))

def refresh():
    doc = dict(schema='abide_code_and_fixed_inputs_v1',
        files=[entry(BASE,p) for p in inventory()],
        dependencies=[entry(MERGE,p) for p in dependencies()])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return check()

def check():
    doc = read(MANIFEST); errors = []
    if doc.get('schema') != 'abide_code_and_fixed_inputs_v1': errors.append('manifest schema')
    if inventory() != sorted(r['path'] for r in doc['files']): errors.append('package file inventory')
    if dependencies() != sorted(r['path'] for r in doc['dependencies']): errors.append('shared dependency inventory')
    for root,rows in ((BASE,doc['files']),(MERGE,doc['dependencies'])):
        for row in rows:
            path = (root/row['path']).resolve(); path.relative_to(root)
            if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                errors.append(str(path.relative_to(MERGE)))
    if errors: raise RuntimeError('ABIDE active integrity failed: '+', '.join(errors))
    return dict(status='PASS_ACTIVE_CODE_CONFIG_AND_DEPENDENCIES',files=len(doc['files']),
        dependency_files=len(doc['dependencies']),scientific_computation_performed=False)

def bindings():
    """Called only after check; bind a new run to this exact active contract."""
    doc = read(MANIFEST)
    result = {str((BASE/r['path']).resolve()):r['sha256'] for r in doc['files']}
    result.update({str((MERGE/r['path']).resolve()):r['sha256'] for r in doc['dependencies']})
    result[str(MANIFEST)] = sha(MANIFEST)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true'); group.add_argument('--refresh',action='store_true')
    args = parser.parse_args()
    print(json.dumps(refresh() if args.refresh else check(),ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
