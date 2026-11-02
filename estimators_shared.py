"""Frozen HNU1 primitive estimators, float64, native information units = nats.

No reference data enter this module. x is (time, ROI); returned edges use
np.triu_indices(R, 1). Directional columns are [i -> j, j -> i].

Author-source validation (accessed 2026-10-19):
GCMI: https://github.com/robince/gcmi, python/src/gcmi/_core.py
  copnorm, mi_gg, _bias_sequence; official returns bits, here nats.
PhiID: https://github.com/Imperial-MIND-lab/integrated-info-decomp
  phyid/{calculate,measures,utils}.py; Gaussian branch uses natural logs.
  Mean local MMI atoms equal the covariance calculation below: local Gaussian
  quadratic terms cancel on averaging, redundancy chooses minimum *mean* MI,
  and the lattice inversion is linear. We do not minimize local MI pointwise.
  Atom names also checked against pmediano/PhiID/PhiIDFull.m: xty is unique X
  to unique Y, ytx the reverse. All 16 signed atoms are retained.

Array contract: r0(E), rho(4,E) at [-2,-1,1,2]; W{1,2}_{rstar,lag,tie}(E);
te_h{1,2,3}_d{1,2}(E,2); var_p{1,2,3}(E,2); psi(E), psi_loo(5,E),
psi_se(E), psi_z(E), coherence(E), gcmi(E); phiid_d{1,2}(E,2) signed atoms,
phiid_atoms_d{1,2}(E,16). Information strength is pooled by the caller.
Diagnostics contain conditions, ranks, residual variances and VAR stability.
No ridge, random jitter, filtering, frame removal or reference estimation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri, digamma
from scipy.signal.windows import dpss

VERSION = "hnu-primitives-1.0.0"
SOURCE_COMMITS = {
    "gcmi": "cd5426bb4c6b7094858f37c3710a4ae9367f8ef1",
    "phyid": "6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44",
}
OFFSETS = np.array([-2, -1, 1, 2], dtype=np.int8)
PHIID_NAMES = tuple(a + "t" + b for a in "rxys" for b in "rxys")


def _validate(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 12 or x.shape[1] < 2:
        raise ValueError("Expected (T>=12, R>=2) real time series")
    if not np.isfinite(x).all() or np.any(np.std(x, axis=0) == 0):
        raise ValueError("Nonfinite/constant ROI must be resolved by common input QC")
    return x


def _crosscorr(a, b):
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)
    den = np.sqrt(np.sum(a*a, axis=0)[:, None] * np.sum(b*b, axis=0)[None, :])
    return np.clip((a.T @ b) / den, -1, 1)


def lag_peaks(rho, offsets=OFFSETS, candidates=(-1, 1), tolerance=1e-12):
    """Return representative signed r, lag, any-tie, opposite-direction-tie.

    Within tolerance of the absolute maximum, choose min |lag|, then signed
    lag ascending. Only opposite-direction ties suppress direction readouts.
    """
    offsets = np.asarray(offsets)
    indices = sorted([int(np.flatnonzero(offsets == d)[0]) for d in candidates],
                     key=lambda k: (abs(int(offsets[k])), int(offsets[k])))
    rr = np.asarray(rho)[indices]
    dd = offsets[indices]
    equal = np.max(np.abs(rr), axis=0)[None, :] - np.abs(rr) <= tolerance
    pick = np.argmax(equal, axis=0)
    r = rr[pick, np.arange(rr.shape[1])]
    lag = dd[pick]
    any_tie = np.sum(equal, axis=0) > 1
    opposite = np.any(equal & (dd[:, None] > 0), axis=0) & np.any(equal & (dd[:, None] < 0), axis=0)
    return r, lag, any_tie, opposite


def correlation_primitives(x):
    i, j = np.triu_indices(x.shape[1], 1)
    out = {"r0": _crosscorr(x, x)[i, j]}
    rho = []
    for d in OFFSETS:
        d = int(d)
        a, b = (x[:-d], x[d:]) if d > 0 else (x[-d:], x[:d])
        rho.append(_crosscorr(a, b)[i, j])
    out["rho"] = np.asarray(rho)
    for name, delays in (("W1", (-1, 1)), ("W2", (-2, -1, 1, 2))):
        r, lag, tie_any, tie = lag_peaks(out["rho"], candidates=delays)
        out.update({name+"_rstar": r, name+"_lag": lag,
                    name+"_tie": tie, name+"_tie_any": tie_any})
    return out


def _lag_cov(x, maxlag):
    n, r = x.shape
    y = np.concatenate([x[maxlag-l:n-l] for l in range(maxlag+1)], axis=1)
    y = y - y.mean(axis=0)
    return y.T @ y / len(y), len(y)


def _regress_cov(cov, predictors, response, n):
    """Batched centered OLS with intercept, without degrees-of-freedom scaling.

    Singular designs use Moore-Penrose least squares and are flagged; no ridge.
    Condition/rank describe the actual centered design including intercept.
    """
    gram = cov[predictors[:, :, None], predictors[:, None, :]]
    cross = cov[predictors, response[:, None]]
    eig = np.linalg.eigvalsh(gram)
    largest = np.maximum(eig[:, -1], 1.0)
    smallest = np.minimum(eig[:, 0], 1.0)
    tol = np.finfo(float).eps * max(n, predictors.shape[1]+1)
    rank = 1 + np.sum(eig > (tol*tol*largest[:, None]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = np.sqrt(largest / np.maximum(smallest, 0))
    singular = (eig[:, 0] <= 0) | (condition > 1/tol)
    beta = np.empty_like(cross)
    good = ~singular
    if np.any(good):
        beta[good] = np.linalg.solve(gram[good], cross[good])
    if np.any(singular):
        beta[singular] = np.einsum("eij,ej->ei", np.linalg.pinv(gram[singular]), cross[singular])
    residual = cov[response, response] - np.sum(beta*cross, axis=1)
    return residual, beta, condition, rank, singular


def _ratio(restricted, full, scale):
    with np.errstate(divide="ignore", invalid="ignore"):
        val = scale * np.log(restricted/full)
    tiny_negative = (val < 0) & (val >= -1e-12)
    bad = (val < -1e-12) | (restricted <= 0) | (full <= 0)
    val[tiny_negative] = 0
    val[bad] = np.nan
    return val, tiny_negative, bad


def gaussian_prediction(x):
    """Gaussian TE histories 1..3/delays 1..2 and bivariate VAR 1..3."""
    r = x.shape[1]
    i, j = np.triu_indices(r, 1)
    e = len(i)
    # A shared affine scaling changes no residual ratio; source data unmodified.
    z = (x-x.mean(axis=0))/x.std(axis=0)
    covs = {p: _lag_cov(z, p) for p in (1, 2, 3)}
    out = {}
    source = np.column_stack((i, j)).ravel()
    target = np.column_stack((j, i)).ravel()
    for h in (1, 2, 3):
        own = target[:, None] + r*np.arange(1, h+1)[None, :]
        for d in (1, 2):
            c, n = covs[max(h, d)]
            pred = np.column_stack((own, source+r*d))
            rr, _, cr, qr, sr = _regress_cov(c, own, target, n)
            rf, _, cf, qf, sf = _regress_cov(c, pred, target, n)
            val, tiny, bad = _ratio(rr, rf, .5)
            name = f"te_h{h}_d{d}"
            out[name] = val.reshape(e, 2)
            out[name+"_condition"] = np.column_stack((cr, cf)).reshape(e, 2, 2)
            out[name+"_rank"] = np.column_stack((qr, qf)).reshape(e, 2, 2)
            out[name+"_residual"] = np.column_stack((rr, rf)).reshape(e, 2, 2)
            out[name+"_flags"] = np.column_stack((sr | sf, tiny, bad)).reshape(e, 2, 3)
    for p in (1, 2, 3):
        c, n = covs[p]
        own = target[:, None] + r*np.arange(1, p+1)[None, :]
        pred = np.column_stack((own, source[:, None]+r*np.arange(1, p+1)[None, :]))
        rr, _, cr, qr, sr = _regress_cov(c, own, target, n)
        rf, beta, cf, qf, sf = _regress_cov(c, pred, target, n)
        val, tiny, bad = _ratio(rr, rf, 1.)
        name = f"var_p{p}"
        out[name] = val.reshape(e, 2)
        out[name+"_condition"] = np.column_stack((cr, cf)).reshape(e, 2, 2)
        out[name+"_rank"] = np.column_stack((qr, qf)).reshape(e, 2, 2)
        out[name+"_residual"] = np.column_stack((rr, rf)).reshape(e, 2, 2)
        out[name+"_flags"] = np.column_stack((sr | sf, tiny, bad)).reshape(e, 2, 3)
        b = beta.reshape(e, 2, 2*p)
        # direction0 response j: [j own history, i source history]
        # direction1 response i: [i own history, j source history]
        companion = np.zeros((e, 2*p, 2*p), dtype=float)
        for l in range(p):
            companion[:, 0, 2*l] = b[:, 1, l]
            companion[:, 0, 2*l+1] = b[:, 1, p+l]
            companion[:, 1, 2*l] = b[:, 0, p+l]
            companion[:, 1, 2*l+1] = b[:, 0, l]
        if p > 1:
            companion[:, 2:, :-2] = np.eye(2*(p-1))[None, :, :]
        radius = np.max(np.abs(np.linalg.eigvals(companion)), axis=1)
        out[name+"_spectral_radius"] = radius
        out[name+"_unstable"] = radius >= 1
    return out


def spectral_primitives(x, tr=2.0, chunk_size=256):
    """DPSS NW=3 K=5 sym=False norm=2; X_i conj(X_j) sign convention."""
    n, r = x.shape
    i, j = np.triu_indices(r, 1)
    taper = dpss(n, NW=3., Kmax=5, sym=False, norm=2)
    fft = np.fft.rfft((x-x.mean(axis=0))[None, :, :]*taper[:, :, None], n=n, axis=1)
    freq = np.fft.rfftfreq(n, d=tr)
    band = (freq >= .009) & (freq <= .08)
    bins = np.flatnonzero(band)
    if len(bins) < 2:
        raise ValueError("Insufficient bins in frozen spectral band")
    f = fft[:, band, :]
    power = np.abs(f)**2
    e = len(i)
    psi, coherence = np.empty(e), np.empty(e)
    loo = np.empty((5, e))
    for lo in range(0, e, chunk_size):
        sl = slice(lo, min(lo+chunk_size, e))
        cross = f[:, :, i[sl]] * np.conj(f[:, :, j[sl]])
        def estimates(keep):
            cs = cross[keep].mean(axis=0)
            pi = power[keep][:, :, i[sl]].mean(axis=0)
            pj = power[keep][:, :, j[sl]].mean(axis=0)
            coherency = cs/np.sqrt(pi*pj)
            ps = np.imag(np.sum(np.conj(coherency[:-1])*coherency[1:], axis=0))
            co = np.mean(np.abs(coherency)**2, axis=0)
            return ps, co
        psi[sl], coherence[sl] = estimates(np.ones(5, dtype=bool))
        for k in range(5):
            loo[k, sl] = estimates(np.arange(5) != k)[0]
    se = np.sqrt(4/5*np.sum((loo-loo.mean(axis=0))**2, axis=0))
    z = np.full_like(psi, np.nan)
    np.divide(psi, se, out=z, where=se > 0)
    return {"psi": psi, "psi_loo": loo, "psi_se": se, "psi_z": z,
            "coherence": coherence, "spectral_frequencies": freq[band]}


def gcmi_primitives(x):
    """Author copnorm ranks/(N+1), analytic Gaussian bias, native nats.

    Stable ordinal ranks reproduce the scalar author implementation including
    its deterministic ties. Ties are audited; no jitter is added.
    """
    n, r = x.shape
    order = np.argsort(x, axis=0, kind="mergesort")
    ranks = np.argsort(order, axis=0, kind="mergesort") + 1
    z = ndtri(ranks/(n+1.))
    z -= z.mean(axis=0)
    c = z.T @ z/(n-1.)
    i, j = np.triu_indices(r, 1)
    corr = c[i, j]/np.sqrt(c[i, i]*c[j, j])
    dterm = (np.log(2.)-np.log(n-1.))/2
    bias1 = dterm+.5*digamma((n-1.)/2)
    bias2 = 2*dterm+.5*(digamma((n-1.)/2)+digamma((n-2.)/2))
    val = -.5*np.log1p(-corr*corr)-2*bias1+bias2
    ties = np.array([n-len(np.unique(x[:, k])) for k in range(r)], dtype=np.int32)
    return {"gcmi": val, "gcmi_ties_per_roi": ties,
            "gcmi_bias_nats": np.asarray(2*bias1-bias2)}


def _phiid_covariance_atoms(c):
    """Vectorized Gaussian/MMI product-lattice inversion with semantic labels.

    Returns (edge,16) signed atoms, total past-future MI, covariance condition,
    failure mask. No hidden diagonal loading or singular determinant fallback.
    """
    eig = np.linalg.eigvalsh(c)
    cond = np.full(len(c), np.inf)
    positive = eig[:, 0] > 0
    cond[positive] = eig[positive, -1]/eig[positive, 0]
    bad = ~positive | (cond > 1/np.finfo(float).eps)
    ent = {}
    for mask in range(1, 16):
        idx = [k for k in range(4) if mask & (1 << k)]
        sub = c[:, idx][:, :, idx]
        sign, logdet = np.linalg.slogdet(sub)
        ent[mask] = .5*logdet
        bad |= sign <= 0
    def mi(a, b):
        # Invalid singular covariance is explicitly returned in ``bad`` below.
        with np.errstate(invalid="ignore"):
            return ent[a]+ent[b]-ent[a | b]
    xx, xy, yx, yy = mi(1, 4), mi(1, 8), mi(2, 4), mi(2, 8)
    joint_x, joint_y = mi(3, 4), mi(3, 8)
    x_joint, y_joint = mi(1, 12), mi(2, 12)
    total = mi(3, 12)
    # Each axis order: redundancy, unique X, unique Y, synergy. Entries are
    # cumulative redundancies; Möbius inversion uses the PID zeta matrix.
    cumulative = np.empty((len(c), 4, 4), dtype=float)
    cumulative[:, 0, 0] = np.minimum.reduce([xx, xy, yx, yy])
    cumulative[:, 0, 1] = np.minimum(xx, yx)
    cumulative[:, 0, 2] = np.minimum(xy, yy)
    cumulative[:, 0, 3] = np.minimum(x_joint, y_joint)
    cumulative[:, 1, 0] = np.minimum(xx, xy)
    cumulative[:, 2, 0] = np.minimum(yx, yy)
    cumulative[:, 3, 0] = np.minimum(joint_x, joint_y)
    cumulative[:, 1, 1], cumulative[:, 1, 2] = xx, xy
    cumulative[:, 2, 1], cumulative[:, 2, 2] = yx, yy
    cumulative[:, 3, 1], cumulative[:, 3, 2] = joint_x, joint_y
    cumulative[:, 1, 3], cumulative[:, 2, 3] = x_joint, y_joint
    cumulative[:, 3, 3] = total
    mobius = np.array([[1, 0, 0, 0], [-1, 1, 0, 0],
                       [-1, 0, 1, 0], [1, -1, -1, 1]], dtype=float)
    atoms = np.einsum("ab,ebc,dc->ead", mobius, cumulative, mobius).reshape(len(c), 16)
    atoms[bad] = np.nan
    total[bad] = np.nan
    return atoms, total, cond, bad


def phiid_primitives(x):
    i, j = np.triu_indices(x.shape[1], 1)
    out = {}
    # Affine normalization is solely numerical, as in official Gaussian API.
    z = (x-x.mean(axis=0))/x.std(axis=0, ddof=1)
    for d in (1, 2):
        a, b = z[:-d], z[d:]
        a = a-a.mean(axis=0)
        b = b-b.mean(axis=0)
        n = len(a)-1
        aa, ab, bb = a.T@a/n, a.T@b/n, b.T@b/n
        c = np.empty((len(i), 4, 4))
        # Variables [X_past,Y_past,X_future,Y_future].
        ids = [i, j, i, j]
        for u in range(4):
            for v in range(4):
                block = aa if u < 2 and v < 2 else bb if u >= 2 and v >= 2 else ab if u < 2 else ab.T
                c[:, u, v] = block[ids[u], ids[v]]
        atoms, total, cond, bad = _phiid_covariance_atoms(c)
        out[f"phiid_atoms_d{d}"] = atoms
        out[f"phiid_d{d}"] = atoms[:, [PHIID_NAMES.index("xty"), PHIID_NAMES.index("ytx")]]
        out[f"phiid_total_mi_d{d}"] = total
        out[f"phiid_condition_d{d}"] = cond
        out[f"phiid_failed_d{d}"] = bad
        out[f"phiid_reconstruction_error_d{d}"] = atoms.sum(axis=1)-total
    return out


def compute_primitives(x, tr=2.0):
    """Compute all W1/W2 primitives and prespecified estimator sensitivities."""
    x = _validate(x)
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("TR must be finite and positive")
    out = correlation_primitives(x)
    out.update(gaussian_prediction(x))
    out.update(spectral_primitives(x, tr))
    out.update(gcmi_primitives(x))
    out.update(phiid_primitives(x))
    out["offsets"] = OFFSETS.copy()
    out["phiid_atom_names"] = np.asarray(PHIID_NAMES)
    return out
