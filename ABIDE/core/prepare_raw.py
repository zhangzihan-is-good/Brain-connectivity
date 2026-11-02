"""Locked ABIDE prediction and multitaper primitives for supplied ROI series."""
from __future__ import annotations
from _paths import activate
activate()
import runtime  # Keep process-local DLL handles alive before numerical imports.
import numpy as np
import estimators_shared as _estimators

SPECTRAL_BAND = (.01, .1)


def estimator_module():
    return _estimators


def spectral_required(x, tr):
    """Final-HNU multitaper estimators, ABIDE's prespecified 0.01–0.1 Hz band.

    Exact five-taper point estimates; unused leave-one-taper diagnostics are
    omitted. No temporal refiltering is performed on the downloaded series.
    """
    estimators = estimator_module()
    n, r = x.shape
    i, j = np.triu_indices(r, 1)
    taper = estimators.dpss(n, NW=3., Kmax=5, sym=False, norm=2)
    fft = np.fft.rfft((x-x.mean(axis=0))[None, :, :]*taper[:, :, None], n=n, axis=1)
    frequency = np.fft.rfftfreq(n, d=tr)
    band = (frequency >= SPECTRAL_BAND[0]) & (frequency <= SPECTRAL_BAND[1])
    if np.sum(band) < 2:
        raise ValueError("Insufficient frequency bins in 0.01–0.1 Hz")
    fourier = fft[:, band, :]
    power = np.abs(fourier)**2
    psi, coherence = np.empty(len(i)), np.empty(len(i))
    for lo in range(0, len(i), 256):
        sl = slice(lo, min(lo+256, len(i)))
        cross = fourier[:, :, i[sl]] * np.conj(fourier[:, :, j[sl]])
        cs = cross.mean(axis=0)
        pi = power[:, :, i[sl]].mean(axis=0)
        pj = power[:, :, j[sl]].mean(axis=0)
        coherency = cs/np.sqrt(pi*pj)
        psi[sl] = np.imag(np.sum(np.conj(coherency[:-1])*coherency[1:], axis=0))
        coherence[sl] = np.mean(np.abs(coherency)**2, axis=0)
    return dict(psi=psi, coherence=coherence), dict(frequency_bins=int(np.sum(band)),
                first_frequency=float(frequency[band][0]), last_frequency=float(frequency[band][-1]))


def prediction_required(x):
    """Use the locked covariance/OLS helpers, computing only used h1 / p1,p2.

    Each retained value is identical to final-HNU gaussian_prediction(x);
    unused histories h2/h3 and order p3 are not evaluated.
    """
    estimators = estimator_module()
    r = x.shape[1]
    i, j = np.triu_indices(r, 1)
    e = len(i)
    z = (x-x.mean(axis=0))/x.std(axis=0)
    covariance = {p: estimators._lag_cov(z, p) for p in (1, 2)}
    source = np.column_stack((i, j)).ravel()
    target = np.column_stack((j, i)).ravel()
    out = {}
    diagnostics = {}
    def finite_condition(a, b):
        value = float(max(np.max(a), np.max(b)))
        return value if np.isfinite(value) else None
    for d in (1, 2):
        cov, n = covariance[d]
        own = target[:, None] + r
        predictors = np.column_stack((own, source+r*d))
        restricted, _, cr, _, sr = estimators._regress_cov(cov, own, target, n)
        full, _, cf, _, sf = estimators._regress_cov(cov, predictors, target, n)
        values, tiny, bad = estimators._ratio(restricted, full, .5)
        name = "te_h1_d{}".format(d)
        out[name] = values.reshape(e, 2)
        diagnostics[name] = dict(singular=int(np.sum(sr | sf)), tiny_negative_clipped=int(np.sum(tiny)),
                                  invalid=int(np.sum(bad)), max_condition=finite_condition(cr, cf))
    for p in (1, 2):
        cov, n = covariance[p]
        own = target[:, None] + r*np.arange(1, p+1)[None, :]
        predictors = np.column_stack((own, source[:, None]+r*np.arange(1, p+1)[None, :]))
        restricted, _, cr, _, sr = estimators._regress_cov(cov, own, target, n)
        full, _, cf, _, sf = estimators._regress_cov(cov, predictors, target, n)
        values, tiny, bad = estimators._ratio(restricted, full, 1.)
        name = "var_p{}".format(p)
        out[name] = values.reshape(e, 2)
        diagnostics[name] = dict(singular=int(np.sum(sr | sf)), tiny_negative_clipped=int(np.sum(tiny)),
                                  invalid=int(np.sum(bad)), max_condition=finite_condition(cr, cf))
    return out, diagnostics

