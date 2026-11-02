"""Native baseline estimators used in Fig. 1; natural logarithms for information measures."""

import functools
import numpy as np
from scipy import signal, special, stats

ATOM_MATRIX = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=float)

ATOM_INVERSE = np.linalg.inv(ATOM_MATRIX)

@functools.lru_cache(maxsize=8)
def _tapers(n):
    return signal.windows.dpss(n, NW=3.0, Kmax=5, sym=False, norm=2)

def _batch_cov(v):
    centered = v - v.mean(axis=1, keepdims=True)
    return np.einsum('nti,ntj->nij', centered, centered, optimize=True) / (v.shape[1] - 1)

def _batch_cmi(a, b, z):
    cov = _batch_cov(np.stack([a, b, z], axis=-1))
    scale = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-30))
    corr = cov / scale[:, :, None] / scale[:, None, :]
    den = np.sqrt(np.maximum((1 - corr[:, 0, 2] ** 2) * (1 - corr[:, 1, 2] ** 2), 1e-15))
    r = np.clip((corr[:, 0, 1] - corr[:, 0, 2] * corr[:, 1, 2]) / den, -1 + 1e-12, 1 - 1e-12)
    return -0.5 * np.log1p(-r * r)

def _batch_gc(src, dst, p=2):
    n = src.shape[1]
    allvars = np.stack([dst[:, p:]] + [dst[:, p - k:n - k] for k in range(1, p + 1)] + [src[:, p - k:n - k] for k in range(1, p + 1)], axis=-1)
    cov = _batch_cov(allvars)

    def residual(cols):
        sub = cov[:, cols, :][:, :, cols]
        if np.any(np.linalg.cond(sub) > 1e12):
            raise ValueError('Ill-conditioned VAR design covariance')
        cross = cov[:, cols, 0]
        try:
            beta = np.linalg.solve(sub, cross[..., None])[..., 0]
        except np.linalg.LinAlgError:
            beta = np.einsum('nij,nj->ni', np.linalg.pinv(sub), cross, optimize=True)
        return np.maximum(cov[:, 0, 0] - np.sum(cross * beta, axis=1), 1e-30)
    er = residual(list(range(1, p + 1)))
    ef = residual(list(range(1, 2 * p + 1)))
    return np.maximum(0, np.log(er / ef))

def phiid_gaussian_mmi_batch(x, y, tau=1):
    v = np.stack([x[:, :-tau], y[:, :-tau], x[:, tau:], y[:, tau:]], axis=-1)
    cov = _batch_cov(v)
    scale = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-30))
    cov = cov / scale[:, :, None] / scale[:, None, :]
    if np.any(np.linalg.cond(cov) > 1e12):
        raise ValueError('Ill-conditioned Gaussian information covariance')
    cache = {}

    def h(ids):
        key = tuple(sorted(ids))
        if key not in cache:
            sub = cov[:, key, :][:, :, key]
            sign, logdet = np.linalg.slogdet(sub)
            invalid = sign <= 0
            if np.any(invalid):
                logdet[invalid] = np.log(np.maximum(np.linalg.eigvalsh(sub[invalid]), 1e-12)).sum(axis=1)
            cache[key] = 0.5 * logdet
        return cache[key]

    def mi(a, b):
        return h(a) + h(b) - h(a + b)
    ixta, ixtb, iyta, iytb = [mi([a], [b]) for a, b in [(0, 2), (0, 3), (1, 2), (1, 3)]]
    ixytab = mi([0, 1], [2, 3])
    ixyta, ixytb = (mi([0, 1], [2]), mi([0, 1], [3]))
    ixtab, iytab = (mi([0], [2, 3]), mi([1], [2, 3]))
    knowns = np.stack([np.minimum.reduce([ixta, ixtb, iyta, iytb]), np.minimum(ixta, iyta), np.minimum(ixtb, iytb), np.minimum(ixtab, iytab), np.minimum(ixta, ixtb), np.minimum(iyta, iytb), np.minimum(ixyta, ixytb), ixta, ixtb, iyta, iytb, ixyta, ixytb, ixtab, iytab, ixytab], axis=1)
    return knowns @ ATOM_INVERSE.T

def compute_baselines_batch(x, y, tr, max_delay=2):
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.ndim != 2 or y.shape != x.shape or x.shape[1] < 32:
        raise ValueError('Expected equal [pair,time] arrays with at least 32 time points')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (not np.isfinite(tr)) or (tr <= 0):
        raise ValueError('Finite inputs and positive TR are required')
    if np.any(np.std(x, axis=1) <= 0) or np.any(np.std(y, axis=1) <= 0):
        raise ValueError('Constant recordings are not supported')
    pairs, n = x.shape
    if not isinstance(max_delay, int) or max_delay < 1 or 3 * max_delay >= n:
        raise ValueError('Invalid source-delay window')
    start = max_delay
    fwd = np.stack([_batch_cmi(x[:, start - d:n - d], y[:, start:], y[:, start - 1:n - 1]) for d in range(1, max_delay + 1)], axis=1)
    rev = np.stack([_batch_cmi(y[:, start - d:n - d], x[:, start:], x[:, start - 1:n - 1]) for d in range(1, max_delay + 1)], axis=1)
    tf, trv = (fwd.max(axis=1), rev.max(axis=1))
    te_lag = np.where(tf >= trv, fwd.argmax(axis=1) + 1, -rev.argmax(axis=1) - 1).astype(float)
    gf, gr = (_batch_gc(x, y, max_delay), _batch_gc(y, x, max_delay))
    tapers = _tapers(n)
    xf = np.fft.rfft((x - x.mean(axis=1, keepdims=True))[:, None, :] * tapers[None, :, :], axis=-1)
    yf = np.fft.rfft((y - y.mean(axis=1, keepdims=True))[:, None, :] * tapers[None, :, :], axis=-1)
    sxx, syy = (np.mean(np.abs(xf) ** 2, axis=1), np.mean(np.abs(yf) ** 2, axis=1))
    sxy = np.mean(xf * np.conj(yf), axis=1)
    cohy = sxy / np.sqrt(np.maximum(sxx * syy, 1e-30))
    freqs = np.fft.rfftfreq(n, d=tr)
    mask = (freqs >= 0.009) & (freqs <= min(0.08, 0.5 / tr))
    c = cohy[:, mask]
    psi = np.imag(np.sum(np.conj(c[:, :-1]) * c[:, 1:], axis=1)) if c.shape[1] >= 2 else np.zeros(pairs)
    coh = np.mean(np.abs(c) ** 2, axis=1) if c.shape[1] else np.zeros(pairs)
    gx = special.ndtri(stats.rankdata(x, method='average', axis=1) / (n + 1.0))
    gy = special.ndtri(stats.rankdata(y, method='average', axis=1) / (n + 1.0))
    gcov = _batch_cov(np.stack([gx, gy], axis=-1))
    r = np.clip(gcov[:, 0, 1] / np.sqrt(gcov[:, 0, 0] * gcov[:, 1, 1]), -1 + 1e-12, 1 - 1e-12)
    correction = 0.5 * (special.digamma((n - 1) / 2.0) - special.digamma((n - 2) / 2.0))
    gcmi = np.maximum(0, -0.5 * np.log1p(-r * r) - correction)
    atoms = np.stack([phiid_gaussian_mmi_batch(x, y, tau) for tau in range(1, max_delay + 1)], axis=1)
    uf, ur = (atoms[:, :, 6], atoms[:, :, 9])
    best = np.argmax(np.maximum(np.abs(uf), np.abs(ur)), axis=1)
    idx = np.arange(pairs)
    diff = uf[idx, best] - ur[idx, best]
    zeros = np.zeros(pairs)
    nans = np.full(pairs, np.nan)
    return {'Gaussian_TE': {'strength': np.maximum(tf, trv), 'direction': tf - trv, 'lag': te_lag}, 'VAR_Granger': {'strength': np.maximum(gf, gr), 'direction': gf - gr, 'lag': nans.copy()}, 'PSI': {'strength': np.abs(psi), 'direction': psi, 'lag': nans.copy()}, 'Coherence': {'strength': coh, 'direction': zeros.copy(), 'lag': nans.copy()}, 'GCMI': {'strength': gcmi, 'direction': zeros.copy(), 'lag': nans.copy()}, 'PhiID_UniqueTransfer': {'strength': np.maximum(np.abs(uf[idx, best]), np.abs(ur[idx, best])), 'direction': diff, 'lag': nans.copy(), 'native_direction_valid': False}}
