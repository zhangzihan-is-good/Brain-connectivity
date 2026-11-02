"""Run synthetic upstream IC tests without datasets or saved audit outputs."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-root', type=Path, help='Writable parent for temporary test fixtures; defaults to system temp')
    parser.add_argument('--output', type=Path, help='Optional JSON report; no report file is written by default')
    args = parser.parse_args()
    parent = (args.scratch_root or Path(tempfile.gettempdir())).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    scratch = (parent / ('ic-check-' + uuid.uuid4().hex)).resolve()
    if scratch.parent != parent:
        raise ValueError('Unexpected scratch directory')
    scratch.mkdir(mode=0o777)
    old_tempdir, old_mkdtemp = tempfile.tempdir, tempfile.mkdtemp
    tempfile.tempdir = str(scratch)
    # Inherit the scratch parent's Windows ACL for synthetic test fixtures.
    # This changes no upstream source and does not patch the library writer.
    if os.name == 'nt':
        def fixture_mkdtemp(suffix=None, prefix=None, dir=None):
            base = Path(dir or tempfile.gettempdir()).resolve()
            path = (base / ((prefix or 'tmp') + uuid.uuid4().hex + (suffix or ''))).resolve()
            if not path.is_relative_to(scratch):
                raise ValueError('Test fixture escaped its temporary scratch directory')
            path.mkdir(mode=0o777)
            return str(path)
        tempfile.mkdtemp = fixture_mkdtemp
    vendor = ROOT / '_vendor' / 'IC'
    sys.path.insert(0, str(vendor))
    try:
        faulthandler.dump_traceback_later(30, repeat=False)
        suite = unittest.defaultTestLoader.discover(str(vendor / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                       failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                       source_modified=False, fixtures='synthetic; temporary scratch removed after run')
    finally:
        faulthandler.cancel_dump_traceback_later()
        tempfile.tempdir, tempfile.mkdtemp = old_tempdir, old_mkdtemp
        if scratch.resolve().parent != parent or not scratch.name.startswith('ic-check-'):
            raise ValueError('Refusing to remove an unexpected scratch path')
        shutil.rmtree(scratch)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    print(receipt['status'], receipt['tests'], 'synthetic upstream tests')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
