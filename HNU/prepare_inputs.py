"""Validate a supplied HNU research-input contract and its local NPY files.

The provider supplies the participant/session contract, group assignment, ROI
order and file identities. This utility checks those supplied inputs locally;
it does not construct a study cohort or transform an imaging download into the
manuscript's preprocessed inputs. Contract contents are never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys

sys.dont_write_bytecode = True
_DLL_DIRECTORY = Path(sys.prefix) / 'Library' / 'bin'
_DLL_HANDLE = (os.add_dll_directory(str(_DLL_DIRECTORY))
               if os.name == 'nt' and _DLL_DIRECTORY.is_dir() else None)
import numpy as np

SHAPE = (295, 109)
RECORDINGS = 290
PARTICIPANTS = 29
SCANS = 10
DEFAULT_DESTINATION = Path(__file__).resolve().parent / 'input_config.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_contract(path):
    """Read supplied plain UTF-8 JSON without changing any contract bytes."""
    payload = Path(path).read_bytes()
    try:
        contract = json.loads(payload.decode('utf8'))
    except (UnicodeError, ValueError) as exc:
        raise ValueError('Contract must be plain UTF-8 JSON without a BOM') from exc
    require(isinstance(contract, dict), 'Contract must be a JSON object')
    return contract, payload


def validate_contract(contract):
    """Check the supplied mapping; no participant or file identities are inferred."""
    specification = contract.get('input_contract')
    require(isinstance(specification, dict), 'Missing input_contract object')
    require(specification.get('shape') == list(SHAPE), 'Declared shape must be 295 by 109')
    require(specification.get('dtype') == 'float32', 'Declared dtype must be float32')
    require(specification.get('TR') == 2, 'Declared TR must be 2 seconds')
    order = contract.get('roi_order')
    require(isinstance(order, list) and len(order) == SHAPE[1], 'roi_order must describe all 109 columns')
    for index, item in enumerate(order):
        require(isinstance(item, dict), 'Each roi_order entry must be an object')
        column = item.get('column_index')
        require(isinstance(column, (str, int)) and not isinstance(column, bool)
                and str(column) == str(index), 'roi_order column indices must run from 0 through 108 in order')
        require(isinstance(item.get('name'), str) and bool(item['name'].strip()), 'Each ROI column needs its supplied name')
    halves = contract.get('halves')
    require(isinstance(halves, dict) and set(halves) == {'A', 'B'}, 'halves must provide the supplied A and B assignments')
    members = []
    for half in ('A', 'B'):
        group = halves[half]
        require(isinstance(group, list) and len(group) >= 3, 'Each supplied half needs at least three participants')
        require(all(isinstance(value, str) and value.strip() for value in group), 'Participant identifiers must be nonempty strings')
        members.extend(group)
    require(len(members) == PARTICIPANTS and len(set(members)) == PARTICIPANTS,
            'The supplied halves must partition 29 distinct participants')
    recordings = contract.get('recordings')
    require(isinstance(recordings, list) and len(recordings) == RECORDINGS, 'Contract must list exactly 290 recordings')
    pairs, filenames = set(), set()
    expected = {(person, scan) for person in members for scan in range(1, SCANS + 1)}
    for index, item in enumerate(recordings):
        prefix = 'Recording entry ' + str(index + 1) + ': '
        require(isinstance(item, dict), prefix + 'expected an object')
        subject, scan = item.get('subject'), item.get('order')
        require(isinstance(subject, str) and subject in members, prefix + 'participant is outside the supplied halves')
        require(type(scan) is int and 1 <= scan <= SCANS, prefix + 'order must be an integer from 1 through 10')
        require(isinstance(item.get('session'), str) and bool(item['session'].strip()), prefix + 'session must be a nonempty string')
        pair = (subject, scan)
        require(pair not in pairs, prefix + 'duplicate participant/order pair')
        pairs.add(pair)
        filename = item.get('filename')
        require(isinstance(filename, str) and bool(filename), prefix + 'filename must be a nonempty string')
        require(not PureWindowsPath(filename).drive and not Path(filename).is_absolute()
                and '/' not in filename and '\\' not in filename and filename not in ('.', '..')
                and Path(filename).suffix.lower() == '.npy', prefix + 'filename must be a local NPY basename')
        require(filename not in filenames, prefix + 'duplicate input filename')
        filenames.add(filename)
        require(isinstance(item.get('sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None,
                prefix + 'sha256 must be the supplied lowercase SHA256 value')
    require(pairs == expected, 'Every supplied participant needs exactly orders 1 through 10')
    return dict(recordings=RECORDINGS, participants=PARTICIPANTS,
                group_sizes={name: len(halves[name]) for name in ('A', 'B')},
                shape=list(SHAPE), dtype='float32', TR_seconds=2)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(contract, data_root):
    """Check exactly the files named by the supplied contract, using no pickle."""
    summary = validate_contract(contract)
    root = Path(data_root).expanduser().resolve()
    require(root.is_dir(), 'Data root must be an existing local directory')
    for index, item in enumerate(contract['recordings']):
        prefix = 'Recording entry ' + str(index + 1) + ': '
        path = (root / item['filename']).resolve()
        require(path.parent == root, prefix + 'input resolves outside the data root')
        require(path.is_file(), prefix + 'NPY file is missing')
        require(sha256_file(path) == item['sha256'], prefix + 'file SHA256 does not match the supplied contract')
        try:
            value = np.load(path, allow_pickle=False, mmap_mode='r')
        except (OSError, ValueError) as exc:
            raise ValueError(prefix + 'input is not a readable numerical NPY file') from exc
        require(isinstance(value, np.ndarray), prefix + 'expected one numerical NPY array')
        require(value.shape == SHAPE, prefix + 'expected shape 295 by 109')
        require(value.dtype == np.dtype('float32'), prefix + 'expected float32')
        require(bool(np.isfinite(value).all()), prefix + 'input contains nonfinite values')
        del value
    summary.update(status='verified', scope='supplied contract structure, file identities, axes, dtype and finite values')
    return summary


def install_contract(payload, destination):
    """Copy supplied bytes locally; never overwrite a different existing file."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return 'already_present'
        raise FileExistsError('Destination already exists with different contents; no file was changed')
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return 'created'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True, help='Existing study-input contract supplied by the author/provider')
    parser.add_argument('--data-root', type=Path, required=True, help='Directory containing the 290 preprocessed NPY files named by that contract')
    parser.add_argument('--install', action='store_true', help='After successful validation, copy the supplied contract for local use')
    parser.add_argument('--destination', type=Path, help='Local destination for --install; defaults to HNU/input_config.json')
    args = parser.parse_args(argv)
    if args.destination is not None and not args.install:
        parser.error('--destination requires --install')
    try:
        contract, payload = read_contract(args.contract)
        summary = validate_inputs(contract, args.data_root)
        summary['local_contract'] = install_contract(payload, args.destination or DEFAULT_DESTINATION) if args.install else 'not_requested'
    except ValueError as exc:
        parser.exit(2, 'Input preparation failed: ' + str(exc) + '\n')
    except OSError:
        parser.exit(2, 'Input preparation failed: local file access failed; check the supplied paths and permissions\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
