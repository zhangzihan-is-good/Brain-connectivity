"""Independent NumPy-only checks for historical IC adapter boundaries."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
_DLL_PATH = Path(sys.prefix) / 'Library' / 'bin'
_DLL_HANDLE = os.add_dll_directory(str(_DLL_PATH)) if os.name == 'nt' and _DLL_PATH.is_dir() else None
import numpy as np

MERGE = Path(__file__).resolve().parents[1]
if str(MERGE) not in sys.path:
    sys.path.insert(0, str(MERGE))
import rgic_backend as backend
from rgic_profiles_abide import abide_potential


class BackendProfileBoundaries(unittest.TestCase):
    def test_noncontiguous_three_dimensional_evidence(self):
        raw = np.arange(1., 25.).reshape(2, 3, 4).transpose(1, 0, 2)
        self.assertFalse(raw.flags.c_contiguous)
        eta, q = backend.normalise_evidence(raw)
        expected_q = np.quantile(raw, .95, axis=-1)
        expected = np.clip(raw / (expected_q[..., None] + 1e-8), 0., 1.)
        self.assertEqual(eta.dtype, raw.dtype)
        np.testing.assert_array_equal(q, expected_q)
        np.testing.assert_array_equal(eta, expected)

    def test_integer_evidence_returns_fractional_float_gate(self):
        raw = np.array([1, 2, 3], dtype=np.int64)
        eta, q = backend.normalise_evidence(raw)
        expected_q = np.quantile(raw, .95)
        self.assertEqual(eta.dtype.kind, 'f')
        np.testing.assert_array_equal(eta, np.clip(raw / (float(expected_q) + 1e-8), 0., 1.))
        self.assertEqual(float(q), float(expected_q))
        self.assertGreater(eta[0], 0.)
        self.assertLess(eta[0], 1.)

    def test_potential_preserves_numpy_broadcasting(self):
        examples = [(np.array([.2, -.3]), 1.),
                    (np.array(.2), np.array([1., 2.])),
                    (np.array([[.2], [-.3]]), np.array([1., 2.]))]
        for rho, cost in examples:
            with self.subTest(rho_shape=np.shape(rho), cost_shape=np.shape(cost)):
                expected = np.sign(rho) * np.minimum(np.abs(rho) * np.exp(.2 * cost), .9)
                np.testing.assert_array_equal(backend.compensated_peak(rho, cost, .2), expected)

    def test_broad_float64_input_is_rounded_only_after_compensation(self):
        rng = np.random.default_rng(82)
        rho = rng.uniform(-.5, .5, 10000)
        lag = rng.choice([-2, -1, 1, 2], 10000).astype(np.int16)
        expected = (np.sign(rho) * np.minimum(np.abs(rho) * np.exp(.2 * np.abs(lag)), .9)).astype(np.float32)
        prematurely_rounded = (np.sign(rho.astype(np.float32)) * np.minimum(
            np.abs(rho.astype(np.float32)) * np.exp(.2 * np.abs(lag)), .9)).astype(np.float32)
        self.assertGreater(np.count_nonzero(expected != prematurely_rounded), 0)
        actual = abide_potential(rho, lag, .2, .9)
        self.assertEqual(actual.dtype, np.dtype('float32'))
        np.testing.assert_array_equal(actual, expected)

    def test_legacy_zero_percentile(self):
        raw = np.array([0., 1., 2.])
        for percentile_api in [False, True]:
            with self.subTest(percentile_api=percentile_api):
                eta, q = backend.normalise_evidence(raw, percentile=0., percentile_api=percentile_api)
                self.assertEqual(float(q), 0.)
                np.testing.assert_array_equal(eta, np.array([0., 1., 1.]))

    def test_simulation_clipped_exponent_uses_matching_library_cost(self):
        rho = np.array([0., 1e-310, 1e-200, -.1])
        cost = np.full(4, 1000.)
        expected = np.sign(rho) * np.minimum(np.abs(rho) * np.exp(np.minimum(cost, 700.)), .9)
        actual = backend.compensated_peak(rho, cost, 1., profile='simulation')
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(actual[1], 0.)
        self.assertLess(actual[1], .9)

    def test_parallel_profiles_do_not_modify_numpy_or_upstream_globals(self):
        quantile_before = np.quantile
        upstream_numpy = backend._ic._gate.__globals__['np']

        def task(index):
            raw = np.random.default_rng(index).uniform(0., 1., (3, 19)).astype(np.float32)
            percentile = [50., 90., 95., 100.][index % 4]
            eta, q = backend.normalise_evidence(raw, percentile, percentile_api=bool(index % 2))
            expected, quantiles = [], []
            for row in raw:
                value = np.percentile(row, percentile) if index % 2 else np.quantile(row, percentile / 100.)
                quantiles.append(value)
                expected.append(np.clip(row / (float(value) + 1e-8), 0., 1.))
            return np.array_equal(eta, np.asarray(expected)) and np.array_equal(q, np.asarray(quantiles))

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(task, range(160)))
        self.assertTrue(all(outcomes))
        self.assertIs(np.quantile, quantile_before)
        self.assertIs(backend._ic._gate.__globals__['np'], upstream_numpy)


if __name__ == '__main__':
    unittest.main()
