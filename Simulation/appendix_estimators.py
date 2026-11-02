"""Additional estimators for the simulation appendix; shared baselines come from estimators.py."""
import numpy as np
from scipy import special
from scipy.spatial import cKDTree
from estimators import compute_baselines_batch, _batch_cov, _batch_gc, _tapers

RIDGE = 1e-8

def _validate(x, y, tr):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 32:
        raise ValueError('Expected two equal one-dimensional recordings with at least 32 samples')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (not np.isfinite(tr)) or (tr <= 0):
        raise ValueError('Inputs must be finite and TR positive')
    if np.std(x) <= 0 or np.std(y) <= 0:
        raise ValueError('Constant recordings are not supported')
    return (x, y)


def _standardize(z):
    z = np.asarray(z, dtype=float)
    return (z - np.mean(z, axis=0)) / np.maximum(np.std(z, axis=0, ddof=1), 1e-15)


def _cmi_knn(a, b, z, k=4, theiler=5):
    joint = _standardize(np.column_stack([a, b, z]))
    n = len(joint)
    tree = cKDTree(joint)
    neighbors = min(n, k + 2 * theiler + 1)
    dist, idx = tree.query(joint, k=neighbors, p=np.inf)
    eligible = np.abs(idx - np.arange(n)[:, None]) > theiler
    cumulative = np.cumsum(eligible, axis=1)
    if np.any(cumulative[:, -1] < k):
        raise ValueError('Insufficient neighbors after temporal exclusion')
    kth = np.argmax(cumulative >= k, axis=1)
    eps = np.nextafter(dist[np.arange(n), kth], 0.0)
    counts = []
    for columns in [(0, 2), (1, 2), (2,)]:
        sub = joint[:, columns]
        sub_tree = cKDTree(sub)
        groups = sub_tree.query_ball_point(sub, eps, p=np.inf)
        c = np.array([sum((abs(j - i) > theiler for j in group)) for i, group in enumerate(groups)])
        counts.append(c)
    nxz, nyz, nz = counts
    return float(special.digamma(k) + np.mean(special.digamma(nz + 1) - special.digamma(nxz + 1) - special.digamma(nyz + 1)))


def compute_knn_te_secondary(x, y, tr, max_delay=2):
    x, y = _validate(x, y, tr)
    if not isinstance(max_delay, int) or max_delay < 1 or max_delay >= len(x) - 2:
        raise ValueError('Invalid source-delay window')
    n = len(x)
    start = max_delay
    fwd = np.array([_cmi_knn(x[start - d:n - d], y[start:], y[start - 1:n - 1]) for d in range(1, max_delay + 1)])
    rev = np.array([_cmi_knn(y[start - d:n - d], x[start:], x[start - 1:n - 1]) for d in range(1, max_delay + 1)])
    f, r = (max(0.0, float(fwd.max())), max(0.0, float(rev.max())))
    lag = int(fwd.argmax()) + 1 if f >= r else -(int(rev.argmax()) + 1)
    return {'KNN_TE': {'strength': max(f, r), 'direction': f - r, 'lag': float(lag)}}


def cmi_history(a, b, z, ridge=RIDGE):
    v = np.concatenate([a[:, :, None], b[:, :, None], z], axis=-1)
    cov = _batch_cov(v)
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-30))
    cov = cov / sd[:, :, None] / sd[:, None, :]
    cov += np.eye(cov.shape[1])[None, :, :] * ridge
    precision = np.linalg.inv(cov)
    r = np.clip(-precision[:, 0, 1] / np.sqrt(precision[:, 0, 0] * precision[:, 1, 1]), -1 + 1e-12, 1 - 1e-12)
    return -0.5 * np.log1p(-r * r)


def te_history(x, y, max_delay, history):
    n = x.shape[1]
    start = max(max_delay, history)
    yp = np.stack([y[:, start - j:n - j] for j in range(1, history + 1)], axis=-1)
    xp = np.stack([x[:, start - j:n - j] for j in range(1, history + 1)], axis=-1)
    fwd = np.stack([cmi_history(x[:, start - d:n - d], y[:, start:], yp) for d in range(1, max_delay + 1)], axis=1)
    rev = np.stack([cmi_history(y[:, start - d:n - d], x[:, start:], xp) for d in range(1, max_delay + 1)], axis=1)
    f, r = (fwd.max(axis=1), rev.max(axis=1))
    lag = np.where(f >= r, fwd.argmax(axis=1) + 1, -rev.argmax(axis=1) - 1).astype(float)
    return dict(strength=np.maximum(f, r), direction=f - r, lag=lag)


def psi_jackknife(x, y, tr):
    n = x.shape[1]
    tapers = _tapers(n)
    xf = np.fft.rfft((x - x.mean(axis=1, keepdims=True))[:, None, :] * tapers[None, :, :], axis=-1)
    yf = np.fft.rfft((y - y.mean(axis=1, keepdims=True))[:, None, :] * tapers[None, :, :], axis=-1)
    f = np.fft.rfftfreq(n, d=tr)
    mask = (f >= 0.009) & (f <= 0.08)
    xpower, ypower, cross = (np.abs(xf[:, :, mask]) ** 2, np.abs(yf[:, :, mask]) ** 2, xf[:, :, mask] * np.conj(yf[:, :, mask]))

    def score(px, py, xy):
        c = xy / np.sqrt(np.maximum(px * py, 1e-30))
        return np.imag(np.sum(np.conj(c[:, :-1]) * c[:, 1:], axis=1))
    sx, sy, sc = (xpower.sum(axis=1), ypower.sum(axis=1), cross.sum(axis=1))
    raw = score(sx, sy, sc)
    deleted = np.stack([score(sx - xpower[:, k], sy - ypower[:, k], sc - cross[:, k]) for k in range(5)], axis=1)
    se = np.sqrt(4 / 5 * np.sum((deleted - deleted.mean(axis=1, keepdims=True)) ** 2, axis=1))
    standardized = raw / np.maximum(se, 1e-12)
    nans = np.full(len(x), np.nan)
    return (dict(strength=np.abs(raw), direction=raw, lag=nans.copy()), dict(strength=np.abs(standardized), direction=standardized, lag=nans.copy(), jackknife_se=se))

