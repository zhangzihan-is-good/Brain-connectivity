import numpy as np
import _shared
from rgic_shared import fls_components,simulation_estimate
FIELDS = {'3T': {'tr': 0.72, 'n': 1200, 'tsnr': 30.2, 'tsnr_sd': 3.8}, '7T': {'tr': 1.0, 'n': 900, 'tsnr': 28.7, 'tsnr_sd': 4.8}}
FAMILIES = ['attenuation', 'dispersion', 'control']
RELATIONS = ['sync', 'stable', 'lag_variable', 'sign_variable', 'sporadic', 'independent']
BASE_SEED = 20261004
N_PAIRS = 24
N_REFERENCE = 16
N_TARGET = 3
N_DEV = 4
N_TEST = 10
BAND = (0.009, 0.08)
KNEE = 0.035
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 6.0, 8.0, 12.0]
GAMMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0, 1.5, 2.0]

def cells():
    return [dict(id=i, field=f, family=m, relation=r) for i, (f, m, r) in enumerate(((f, m, r) for f in FIELDS for m in FAMILIES for r in RELATIONS))]

def rng_for(cell, split, repeat, variant=0):
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, cell['id'], split, repeat, variant]))

def unit(z):
    z = z - z.mean(axis=-1, keepdims=True)
    return z / np.maximum(z.std(axis=-1, keepdims=True), 1e-12)

def spectrum_process(rng, shape, tr, colored=True, knee=KNEE):
    n = shape[-1]
    f = np.fft.rfftfreq(n, tr)
    weight = ((f >= BAND[0]) & (f <= BAND[1])).astype(float)
    if colored:
        weight = weight / np.sqrt(1 + (f / knee) ** 2)
    z = np.fft.irfft(np.fft.rfft(rng.standard_normal(shape), axis=-1) * weight, n=n, axis=-1)
    return (unit(z), f)

def generate(cell, split, repeat, noise_ratio=1.0, knee=KNEE, n_pairs=N_PAIRS, variant=0):
    if cell.get('field') not in FIELDS or cell.get('family') not in FAMILIES or cell.get('relation') not in RELATIONS:
        raise ValueError('Unknown simulation condition')
    if not np.isfinite(noise_ratio) or noise_ratio < 0 or not np.isfinite(knee) or knee <= 0 or not isinstance(n_pairs, int) or n_pairs < 1:
        raise ValueError('Invalid noise power, spectral knee or pair count')
    cfg = FIELDS[cell['field']]
    rng = rng_for(cell, split, repeat, variant)
    tr, n = (cfg['tr'], cfg['n'])
    pad = 128
    total = n + 2 * pad
    nr = N_REFERENCE + N_TARGET
    shape = (nr, n_pairs, total)
    kappa = rng.uniform(0.35, 0.8, n_pairs)
    template_d = rng.choice([-2.0, -1.0, 1.0, 2.0], n_pairs)
    template_s = rng.choice([-1.0, 1.0], n_pairs)
    tsnr = np.clip(rng.normal(cfg['tsnr'], cfg['tsnr_sd'], n_pairs), cfg['tsnr'] - 2 * cfg['tsnr_sd'], cfg['tsnr'] + 2 * cfg['tsnr_sd'])
    d = np.broadcast_to(template_d, (nr, n_pairs)).copy()
    signs = np.broadcast_to(template_s, (nr, n_pairs)).copy()
    present = np.ones((nr, n_pairs), dtype=bool)
    relation = cell['relation']
    if relation == 'sync':
        template_d[:] = 0
        d[:] = 0
    if relation == 'lag_variable':
        d = rng.choice([-2.0, -1.0, 1.0, 2.0], (nr, n_pairs))
    if relation == 'sign_variable':
        signs = rng.choice([-1.0, 1.0], (nr, n_pairs))
    if relation == 'sporadic':
        present = rng.random((nr, n_pairs)) < 0.3
    if relation == 'independent':
        present[:] = False
    cost = np.abs(d)
    if cell['family'] == 'attenuation':
        attenuation = np.exp(-0.18 * cost)
        width = np.zeros_like(cost)
        jitter = rng.normal(0, 0.08 * cost)
    elif cell['family'] == 'dispersion':
        attenuation = np.exp(-0.06 * cost)
        width = 0.3 * cost
        jitter = rng.normal(0, 0.12 * cost)
    else:
        attenuation = np.ones_like(cost)
        width = np.zeros_like(cost)
        jitter = np.zeros_like(cost)
    actual_d = d + jitter
    u, f = spectrum_process(rng, shape, tr, True, knee)
    v, _ = spectrum_process(rng, shape, tr, True, knee)
    ex, _ = spectrum_process(rng, shape, tr, False, knee)
    ey, _ = spectrum_process(rng, shape, tr, False, knee)
    phase = np.exp(-2j * np.pi * actual_d[..., None] * f)
    kernel = np.exp(-2 * np.pi ** 2 * width[..., None] ** 2 * f ** 2)
    driven = np.fft.irfft(np.fft.rfft(u, axis=-1) * phase * kernel, n=total, axis=-1)
    driven_variance = np.mean(driven ** 2, axis=-1)
    a = kappa[None, :] * attenuation * present
    latent_y = signs[..., None] * a[..., None] * driven + np.sqrt(np.maximum(1 - a ** 2 * driven_variance, 0))[..., None] * v
    sx = (u + np.sqrt(noise_ratio) * ex) / np.sqrt(1 + noise_ratio)
    sy = (latent_y + np.sqrt(noise_ratio) * ey) / np.sqrt(1 + noise_ratio)
    sx = unit(sx[..., pad:pad + n])
    sy = unit(sy[..., pad:pad + n])
    scale = 100 / tsnr
    x = (100 + sx * scale[None, :, None]).astype(np.float32)
    y = (100 + sy * scale[None, :, None]).astype(np.float32)
    return dict(x=x, y=y, kappa=kappa, template_d=template_d, template_s=template_s, actual_d=actual_d, nominal_d=d, actual_s=signs, present=present, tsnr=tsnr, noise_ratio=np.array(noise_ratio), knee=np.array(knee), cell_id=np.array(cell['id']), split=np.array(split), repeat=np.array(repeat))

def pearson(x, y):
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.shape != y.shape or x.ndim < 1 or x.shape[-1] < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Expected finite, equal arrays with at least two samples')
    a, b = (x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True))
    den = np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))
    if np.any(den <= 0):
        raise ValueError('Constant recordings are not supported')
    return np.sum(a * b, axis=-1) / den

def candidates(data, tr, window='physical'):
    if window not in ('physical', 'W2') or not np.isfinite(tr) or tr <= 0:
        raise ValueError('Expected physical or W2 window and positive finite TR')
    x, y = (data['x'], data['y'])
    limit = int(np.floor(2.2 / tr + 1e-10)) if window == 'physical' else 2
    if limit < 1 or limit > x.shape[-1] - 2:
        raise ValueError('Window must retain at least two overlapping samples')
    lags = np.array(list(range(-limit, 0)) + list(range(1, limit + 1)))
    ccf = np.stack([pearson(x[..., :-l], y[..., l:]) if l > 0 else pearson(x[..., -l:], y[..., :l]) for l in lags], axis=-1)
    ix = np.argmax(np.abs(ccf), axis=-1)
    peak = np.take_along_axis(ccf, ix[..., None], axis=-1)[..., 0]
    tau = lags[ix]
    r0 = pearson(x, y)
    return dict(r0=r0, peak=peak, tau=tau, G=np.maximum(0, np.abs(peak) - np.abs(r0)), C=np.abs(tau), lags=lags, ccf=ccf)

def reliability_components(stats):
    """FLS from positive-gain references; target records never fit the template."""
    tau, peak, gain = (np.asarray(stats[k]) for k in ('tau', 'peak', 'G'))
    if tau.shape != peak.shape or tau.ndim != 2 or len(tau) < N_REFERENCE or not np.isfinite(peak).all() or not np.isin(tau, stats['lags']).all():
        raise ValueError('Expected record-by-edge lag peaks and at least 16 references')
    if gain.shape != tau.shape or not np.isfinite(gain).all() or np.any(gain < 0):
        raise ValueError('Gain must be finite, nonnegative and match lag-peak arrays')
    result=fls_components(tau[:N_REFERENCE],peak[:N_REFERENCE],gain[:N_REFERENCE],stats['lags'])
    return {key:result[key] for key in ('F','L','S','R')}

def reliability(stats):
    R = reliability_components(stats)['R']
    return np.broadcast_to(R, np.asarray(stats['peak']).shape).copy()

def estimate(stats, R, method, tr, lam=1.0, gamma=0.2, seconds_parameters=True):
    if method not in ('Pearson', 'Lag peak', 'RGIC', 'No reliability', 'No gain', 'No cost', 'No compensation'):
        raise ValueError('Unknown estimator')
    if not np.isfinite([tr, lam, gamma]).all() or tr <= 0 or lam < 0 or gamma < 0:
        raise ValueError('Expected positive finite TR and nonnegative finite parameters')
    R = np.asarray(R, dtype=float)
    if R.shape != stats['r0'].shape or not np.isfinite(R).all() or np.any((R < 0) | (R > 1)):
        raise ValueError('Reliability must match edge-array shape and lie in [0, 1]')
    if any(np.asarray(stats[k]).shape != R.shape or not np.isfinite(stats[k]).all() for k in ('r0', 'peak', 'tau', 'G', 'C')):
        raise ValueError('Lag statistics must have matching finite shapes')
    r0, peak, tau, G, C = [stats[k] for k in ['r0', 'peak', 'tau', 'G', 'C']]
    if np.any(G < 0) or np.any(C < 0):
        raise ValueError('Gain and temporal cost must be nonnegative')
    return simulation_estimate(stats,R,method,tr,lam,gamma,seconds_parameters)
