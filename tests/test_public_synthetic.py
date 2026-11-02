"""Public, participant-data-free tests that run in a standalone checkout."""
import importlib.util
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from rgic_backend import RGIC,RGICConfig
import rgic_shared
spec=importlib.util.spec_from_file_location('_rgic_synthetic_demo',ROOT/'examples'/'synthetic_demo.py')
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)

class PublicSyntheticTests(unittest.TestCase):
    def test_shared_cached_adapter_retains_zero_gate_reference(self):
        r0=np.array([.2,-.3]);potential=np.array([.8,-.9]);eta=np.zeros(2)
        np.testing.assert_array_equal(rgic_shared.fuse_cached(r0,eta,potential),r0)

    def test_example_is_deterministic_and_has_expected_direction(self):
        arrays,summary=demo.compute();again,other=demo.compute()
        self.assertFalse(summary['participant_data_used'])
        self.assertEqual(summary['array_sha256'],other['array_sha256'])
        self.assertEqual(summary['first_pair_lag_samples'],[1,1,1])
        for key,value in arrays.items():np.testing.assert_array_equal(value,again[key])
        self.assertEqual(arrays['ic'].shape,(3,6))
        self.assertTrue(np.isfinite(arrays['ic']).all())
        self.assertTrue(np.all((arrays['eta']>=0)&(arrays['eta']<=1)))

    def test_target_changes_cannot_refit_reference_reliability(self):
        reference,target=demo.make_records()
        model=RGIC(RGICConfig(lags=(-1,1))).fit(reference)
        before=model.reliability_.copy()
        model.transform(target)
        altered=np.random.default_rng(7).normal(size=target.shape)
        model.transform(altered)
        np.testing.assert_array_equal(model.reliability_,before)

    def test_roi_order_mismatch_is_rejected(self):
        reference,target=demo.make_records()
        model=RGIC().fit(reference,roi_names=('a','b','c','d'))
        with self.assertRaises(ValueError):model.transform(target,roi_names=('b','a','c','d'))

if __name__=='__main__':unittest.main()
