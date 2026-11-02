"""Standard-library integrity and isolated-output checks for ABIDE."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
MERGE=BASE.parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify():
    import importlib.util
    spec=importlib.util.spec_from_file_location('_abide_integrity',BASE/'tools/integrity.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    return checker.check()


def validate_output(output,data_workspace):
    out=Path(output).resolve();data=Path(data_workspace).resolve()
    if not data.is_dir():raise ValueError('Data workspace is missing')
    protected=[BASE.resolve(),data/'dataset',data/'第四部分',data/'第三部分最终版']
    if out==data or out in data.parents or any(out==p or p in out.parents or out in p.parents for p in protected):
        raise ValueError('Output must be isolated from the package and original scientific/input directories')
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output directory; existing results are never overwritten')
    return out,data
