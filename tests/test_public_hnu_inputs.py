"""Synthetic-only checks for accepting a provider-supplied local HNU contract."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_public_hnu_prepare_inputs', ROOT / 'HNU' / 'prepare_inputs.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
import numpy as np


class HNUInputPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = Path(tempfile.gettempdir()).resolve()
        cls.scratch = (cls.parent / ('hnu-input-check-' + uuid.uuid4().hex)).resolve()
        assert cls.scratch.parent == cls.parent
        cls.scratch.mkdir(mode=0o777)
        cls.data = cls.scratch / 'synthetic_arrays'
        cls.data.mkdir()
        participants = ['fixture-' + str(index) for index in range(helper.PARTICIPANTS)]
        boundary = len(participants) // 2
        cls.contract = {
            'input_contract': {'shape': list(helper.SHAPE), 'dtype': 'float32', 'TR': 2},
            'roi_order': [{'column_index': str(index), 'name': 'fixture-column-' + str(index)}
                          for index in range(helper.SHAPE[1])],
            'halves': {'A': participants[:boundary], 'B': participants[boundary:]},
            'recordings': []}
        values = np.arange(np.prod(helper.SHAPE), dtype=np.float32).reshape(helper.SHAPE)
        for participant in participants:
            for order in range(1, helper.SCANS + 1):
                filename = 'fixture-record-' + str(len(cls.contract['recordings'])) + '.npy'
                path = cls.data / filename
                np.save(path, values, allow_pickle=False)
                cls.contract['recordings'].append(dict(subject=participant, order=order,
                    session='fixture-session-' + str(order), filename=filename, sha256=helper.sha256_file(path)))
        cls.payload = (json.dumps(cls.contract, ensure_ascii=False, indent=2) + '\n').encode('utf8')
        cls.source = cls.scratch / 'supplied_contract.json'
        cls.source.write_bytes(cls.payload)

    @classmethod
    def tearDownClass(cls):
        assert cls.scratch.resolve().parent == cls.parent
        assert cls.scratch.name.startswith('hnu-input-check-')
        shutil.rmtree(cls.scratch)

    def test_all_files_validate_without_changing_contract(self):
        original = copy.deepcopy(self.contract)
        summary = helper.validate_inputs(self.contract, self.data)
        self.assertEqual(summary['recordings'], len(self.contract['recordings']))
        self.assertEqual(summary['participants'], sum(map(len, self.contract['halves'].values())))
        self.assertEqual(summary['shape'], list(helper.SHAPE))
        self.assertEqual(self.contract, original)
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_explicit_install_preserves_bytes_and_prints_no_identities(self):
        destination = self.scratch / 'installed' / 'input_config.json'
        args = ['--contract', str(self.source), '--data-root', str(self.data),
                '--install', '--destination', str(destination)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(helper.main(args), 0)
        self.assertEqual(destination.read_bytes(), self.payload)
        self.assertEqual(json.loads(output.getvalue())['local_contract'], 'created')
        for row in self.contract['recordings']:
            self.assertNotIn(row['subject'], output.getvalue())
            self.assertNotIn(row['filename'], output.getvalue())
            self.assertNotIn(row['sha256'], output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(helper.main(args), 0)
        self.assertEqual(destination.read_bytes(), self.payload)

    def test_checksum_failure_never_installs_contract(self):
        contract = copy.deepcopy(self.contract)
        contract['recordings'][0]['sha256'] = hashlib.sha256(b'synthetic-mismatch').hexdigest()
        supplied = self.scratch / 'checksum_mismatch.json'
        supplied.write_text(json.dumps(contract), encoding='utf8')
        destination = self.scratch / 'must_not_be_created.json'
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as raised:
            helper.main(['--contract', str(supplied), '--data-root', str(self.data),
                         '--install', '--destination', str(destination)])
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(destination.exists())
        self.assertNotIn(contract['recordings'][0]['subject'], output.getvalue())
        self.assertNotIn(contract['recordings'][0]['sha256'], output.getvalue())

    def test_mapping_and_traversal_errors_are_rejected(self):
        broken = copy.deepcopy(self.contract)
        broken['recordings'][0]['filename'] = '../outside.npy'
        with self.assertRaises(ValueError):
            helper.validate_contract(broken)
        broken = copy.deepcopy(self.contract)
        broken['recordings'][1]['order'] = broken['recordings'][0]['order']
        with self.assertRaises(ValueError):
            helper.validate_contract(broken)
        broken = copy.deepcopy(self.contract)
        broken['halves']['B'][0] = broken['halves']['A'][0]
        with self.assertRaises(ValueError):
            helper.validate_contract(broken)

    def test_loader_shape_dtype_and_finite_boundaries(self):
        root = self.scratch / 'invalid_arrays'
        root.mkdir()
        examples = [np.zeros((helper.SHAPE[0] - 1, helper.SHAPE[1]), dtype=np.float32),
                    np.zeros(helper.SHAPE, dtype=np.float64),
                    np.full(helper.SHAPE, np.nan, dtype=np.float32)]
        for values in examples:
            with self.subTest(shape=values.shape, dtype=str(values.dtype)):
                contract = copy.deepcopy(self.contract)
                path = root / contract['recordings'][0]['filename']
                np.save(path, values, allow_pickle=False)
                contract['recordings'][0]['sha256'] = helper.sha256_file(path)
                with self.assertRaises(ValueError):
                    helper.validate_inputs(contract, root)

    def test_existing_different_destination_is_preserved(self):
        destination = self.scratch / 'existing_contract.json'
        previous = b'{"previous_local_contract": true}\n'
        destination.write_bytes(previous)
        with self.assertRaises(FileExistsError):
            helper.install_contract(self.payload, destination)
        self.assertEqual(destination.read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
