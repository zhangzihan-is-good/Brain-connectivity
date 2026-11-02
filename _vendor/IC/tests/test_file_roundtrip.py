"""Check that persisted state preserves scientific outputs and ROI identity."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rgic import RGIC
from rgic.io import load_model, load_recordings, save_model, save_results


class FileRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(23)
        self.records = []
        for length in (90, 100, 110):
            x = rng.normal(size=length + 1)
            self.records.append(np.column_stack([
                x[1:], 0.8*x[:-1] + 0.2*rng.normal(size=length),
                rng.normal(size=length)]))
        self.names = ('source', 'delayed', 'independent')

    def test_fitted_state_roundtrip_preserves_core_outputs(self):
        model = RGIC().fit(self.records, roi_names=self.names)
        path = self.root / 'model.npz'
        save_model(model, path)
        restored = load_model(path)
        a = model.transform(self.records, roi_names=self.names)
        b = restored.transform(self.records, roi_names=self.names)
        self.assertEqual(restored.roi_names_, self.names)
        for original, loaded in zip(a, b):
            for field in ('ic', 'eta', 'potential', 'reliability', 'lag', 'opposite_tie'):
                np.testing.assert_array_equal(getattr(original, field), getattr(loaded, field))
        with self.assertRaises(ValueError):
            restored.transform(self.records, roi_names=self.names[::-1])

    def test_variable_length_input_and_output_matrix_orientation(self):
        path = self.root / 'records.npz'
        np.savez(path, **{f'record_{i:03d}': x for i, x in enumerate(self.records)})
        loaded = load_recordings(path)
        self.assertEqual([x.shape[0] for x in loaded], [90, 100, 110])
        results = RGIC().fit(loaded).transform(loaded)
        output = self.root / 'output.npz'
        save_results(results, output, source_path=path)
        with np.load(output, allow_pickle=False) as data:
            self.assertEqual(data['ic'].shape, (3, 3, 3))
            np.testing.assert_allclose(data['ic'], data['ic'].transpose(0, 2, 1))
            np.testing.assert_allclose(data['lag_samples'], -data['lag_samples'].transpose(0, 2, 1))
            metadata = json.loads(data['metadata'].item())
            self.assertEqual(len(metadata['input_file_sha256']), 64)

    def test_invalid_reference_product_and_overwrite_are_rejected(self):
        path = self.root / 'model.npz'
        model = RGIC().fit(self.records)
        save_model(model, path)
        with self.assertRaises(ValueError):
            save_model(model, path)
        with np.load(path, allow_pickle=False) as data:
            altered = {k: data[k].copy() for k in data.files}
        altered['reliability'][:] = 0.123456
        corrupted = self.root / 'corrupted.npz'
        np.savez(corrupted, **altered)
        with self.assertRaises(ValueError):
            load_model(corrupted)


if __name__ == '__main__':
    unittest.main()
