"""Scientific invariants for the prototype; run with the Python standard library."""
from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from rgic import RGIC, RGICConfig


def delayed_records(n=8, t=180, seed=5, delay=1, coupling=0.75, polarity=1):
    """Positive delay means the second signal follows the first."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.normal(size=t + delay)
        x = z[delay:]
        y = polarity * (coupling * z[:t] + np.sqrt(1-coupling**2)*rng.normal(size=t))
        out.append(np.column_stack([x, y]))
    return out


class ScientificBehaviorTests(unittest.TestCase):
    def test_known_positive_delay(self):
        ref = delayed_records(coupling=0.95)
        model = RGIC(RGICConfig(lags=(-2, -1, 1, 2))).fit(ref)
        r = model.transform(delayed_records(n=1, seed=90, coupling=0.95))[0]
        self.assertEqual(int(r.lag[0]), 1)
        m = r.matrix('lag')
        self.assertGreater(m[0, 1], 0)
        np.testing.assert_allclose(m, -m.T)

    def test_negative_correlation_is_not_reverse_temporal_order(self):
        model = RGIC().fit(delayed_records(polarity=-1, coupling=0.9))
        r = model.transform(delayed_records(n=1, seed=100, polarity=-1, coupling=0.9))[0]
        self.assertEqual(int(r.lag[0]), 1)
        self.assertLess(r.ic[0], 0)
        self.assertGreater(r.lag[0], 0)

    def test_swapping_regions_reverses_lag_not_strength(self):
        ref = delayed_records(coupling=0.85)
        target = delayed_records(n=1, seed=22, coupling=0.85)
        r = RGIC().fit(ref).transform(target)[0]
        s = RGIC().fit([a[:, ::-1] for a in ref]).transform([a[:, ::-1] for a in target])[0]
        np.testing.assert_allclose(r.ic, s.ic, atol=1e-12)
        np.testing.assert_allclose(r.strength, s.strength, atol=1e-12)
        np.testing.assert_array_equal(r.lag, -s.lag)

    def test_zero_reliability_retains_pearson(self):
        rng = np.random.default_rng(9)
        refs = []
        for _ in range(5):
            x = rng.normal(size=160)
            refs.append(np.column_stack([x, x]))
        model = RGIC().fit(refs)
        np.testing.assert_array_equal(model.reliability_, 0)
        r = model.transform(delayed_records(n=1, seed=91))[0]
        np.testing.assert_array_equal(r.ic, r.r0)
        np.testing.assert_array_equal(r.eta, 0)

    def test_test_records_never_update_reference_state(self):
        model = RGIC().fit(delayed_records())
        before = [getattr(model, name).copy() for name in ['F_', 'L_', 'S_', 'reliability_']]
        model.transform(delayed_records(n=2, seed=98, polarity=-1))
        for name, saved in zip(['F_', 'L_', 'S_', 'reliability_'], before):
            np.testing.assert_array_equal(getattr(model, name), saved)

    def test_time_unit_conversion_preserves_outputs(self):
        seconds = RGICConfig(lags=(-2, -1, 1, 2), lag_unit='seconds', tr=2,
                             lambda_=0.4, gamma=0.1)
        samples = replace(seconds, lag_unit='samples', tr=None, lambda_=0.8, gamma=0.2)
        refs = delayed_records(delay=2, coupling=0.6)
        target = delayed_records(n=1, seed=29, delay=2, coupling=0.6)
        a = RGIC(seconds).fit(refs).transform(target)[0]
        b = RGIC(samples).fit(refs).transform(target)[0]
        for name in ['ic', 'eta', 'potential', 'reliability']:
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=1e-12)
        np.testing.assert_allclose(a.cost, 2*b.cost)

    def test_opposite_peak_tie_is_flagged(self):
        x = np.tile([1., 0., -1., 0.], 50)
        y = np.tile([0., 1., 0., -1.], 50)
        records = [np.column_stack([x, y]) for _ in range(3)]
        r = RGIC().fit(records).transform(records)[0]
        self.assertTrue(bool(r.opposite_tie[0]))
        self.assertEqual(int(r.lag[0]), -1)

    def test_variable_record_lengths_are_supported(self):
        refs = delayed_records()
        refs[0] = refs[0][:-20]
        model = RGIC().fit(refs)
        out = model.transform([refs[0], refs[1]])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].matrix('ic').shape, (2, 2))

    def test_reference_frequency_lag_and_sign_counts(self):
        records = delayed_records(n=6, coupling=0.95)
        records += [delayed_records(n=1, seed=100, coupling=0.95)[0][:, ::-1]]
        records += [delayed_records(n=1, seed=101, coupling=0.95, polarity=-1)[0][:, ::-1]]
        rng = np.random.default_rng(42)
        for _ in range(2):
            x = rng.normal(size=180)
            records.append(np.column_stack([x, x]))
        model = RGIC().fit(records)
        np.testing.assert_allclose(model.F_, 0.8)
        np.testing.assert_allclose(model.L_, 0.75)
        np.testing.assert_allclose(model.S_, 0.75)
        np.testing.assert_allclose(model.reliability_, 0.45)

    def test_zero_target_gain_retains_pearson(self):
        model = RGIC().fit(delayed_records())
        x = np.random.default_rng(11).normal(size=180)
        result = model.transform(np.column_stack([x, x]))[0]
        np.testing.assert_array_equal(result.gain, 0)
        np.testing.assert_array_equal(result.ic, result.r0)

    def test_gate_normalization_includes_zero_edges(self):
        from rgic.core import _gate
        gain = np.r_[np.zeros(19), 0.02, 0.08]
        _, eta, q = _gate(gain, np.zeros(21), np.ones(21), RGICConfig())
        self.assertAlmostEqual(q, 0.02)
        self.assertAlmostEqual(eta[-2], 0.02/(0.02+1e-8))
        self.assertEqual(eta[-1], 1)

    def test_roi_labels_catch_order_mismatch(self):
        model = RGIC().fit(delayed_records(), roi_names=['A', 'B'])
        with self.assertRaises(ValueError):
            model.transform(delayed_records(n=1), roi_names=['B', 'A'])

    def test_invalid_inputs_and_unfitted_model_fail(self):
        with self.assertRaises((ValueError, RuntimeError)):
            RGIC().transform(delayed_records(n=1))
        with self.assertRaises(ValueError):
            RGIC().fit([np.ones((20, 2))])
        with self.assertRaises(ValueError):
            RGICConfig(lag_unit='seconds')
        with self.assertRaises(ValueError):
            RGICConfig(lags=(0, 1))
        bad = delayed_records(n=1)[0]
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            RGIC().fit([bad])

    def test_config_is_immutable(self):
        cfg = RGICConfig()
        with self.assertRaises(FrozenInstanceError):
            cfg.delta = 0.01


if __name__ == '__main__':
    unittest.main()
