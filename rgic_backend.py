"""Pinned IC primitives and explicitly selected historical numerical profiles.

The published package accepts raw recordings, not cached sufficient statistics.
Its private gate/potential primitives are pinned by commit and called here.
Experiment adapters retain the original lag search, reference set, aggregation,
and floating-point operation order. No experiment is silently standardized.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import numpy as np

IC_COMMIT = '58625fc4a3afd69d451f27b5c279a3c2a3cae90f'
IC_VERSION = '0.1.0a3'
_PACKAGE_NAME = '_merge_pinned_ic'
_ROOT = Path(__file__).resolve().parent

def _load_library():
    if _PACKAGE_NAME not in sys.modules:
        directory = _ROOT/'_vendor'/'IC'/'rgic'
        spec = importlib.util.spec_from_file_location(
            _PACKAGE_NAME, directory/'__init__.py', submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE_NAME] = module
        spec.loader.exec_module(module)
    return sys.modules[_PACKAGE_NAME+'.core']

_ic = _load_library()
RGIC, RGICConfig, RGICResult = _ic.RGIC, _ic.RGICConfig, _ic.RGICResult

class _NumpyGateProfile:
    """Per-call proxy: preserve percentile spelling and support NumPy 1.21.

    Only the globals of a cloned upstream function see this proxy. Neither
    NumPy nor the upstream module is monkey-patched, so concurrent calls are safe.
    """
    def __init__(self, percentile=None):
        self.percentile = percentile
    def __getattr__(self, name):
        return getattr(np, name)
    def quantile(self, values, q, method='linear'):
        if self.percentile is not None:
            return np.percentile(values, self.percentile)
        if np.lib.NumpyVersion(np.__version__) < '1.22.0':
            return np.quantile(values, q, interpolation=method)
        return np.quantile(values, q, method=method)

def normalise_evidence(raw, percentile=95., epsilon=1e-8, percentile_api=False):
    """Normalize already-discounted historical evidence with IC's actual gate.

    Pre-discounting preserves differing multiplication orders in the four
    experiments. Unit reliability/zero cost leave that evidence unchanged.
    Returns eta with the input shape and q over its final (edge) dimension.
    """
    raw = np.asarray(raw)
    if raw.ndim < 1:
        raise ValueError('Evidence must have an edge dimension')
    # The private primitive only reads these fields. A lightweight config also
    # preserves the old helpers' accepted 0th-percentile boundary, which the
    # public raw-recording RGICConfig deliberately disallows.
    cfg = types.SimpleNamespace(lambda_=0.,quantile=float(percentile)/100.,epsilon=epsilon)
    scope = dict(_ic._gate.__globals__)
    scope['np'] = _NumpyGateProfile(float(percentile) if percentile_api else None)
    gate = types.FunctionType(_ic._gate.__code__, scope, _ic._gate.__name__, _ic._gate.__defaults__)
    rows = raw.reshape((-1, raw.shape[-1]))
    # Ufunc output layout depends on both operands, including the broadcast
    # quantile array. Preserve that layout rather than assuming raw's order.
    with np.errstate(all='ignore'):
        layout=np.divide(raw,np.ones(raw.shape[:-1]+(1,),dtype=float))
    eta=np.empty_like(layout,dtype=raw.dtype if raw.dtype.kind=='f' else np.float64,order='K')
    q = np.empty(raw.shape[:-1], dtype=float)
    for index,(leading,row) in enumerate(zip(np.ndindex(raw.shape[:-1]),rows)):
        _, result, quantile = gate(row, np.zeros_like(row), np.ones_like(row), cfg)
        # Index the allocation, not a possibly copied reshape view.
        eta[leading]=result
        q.flat[index] = quantile
    return eta, q

def compensated_peak(rstar, cost, gamma, cap=.9, profile='direct64'):
    """Call IC potential and preserve the experiment's pre-existing rounding.

    IC uses log-space compensation; historical profiles used exp then multiply.
    In addition to last-bit differences, legacy overflow may deliberately yield
    NaN and a warning. The reference calculation must not replace those accepted
    outputs with a new exception. Only stable64 returns the library value.
    """
    original = np.asarray(rstar)
    cost = np.asarray(cost)
    # Suppress reference-only warnings; the selected historical expression below
    # remains responsible for exactly the warnings/results its old caller saw.
    with np.errstate(all='ignore'):
        cfg=types.SimpleNamespace(gamma=float(gamma),cap=float(cap))
        library_rho,library_cost=np.broadcast_arrays(np.asarray(original,dtype=np.float64),
                                                    np.asarray(cost,dtype=np.float64))
        if profile=='simulation':
            library_cost=np.minimum(float(gamma)*library_cost,700)
            cfg=types.SimpleNamespace(gamma=1.,cap=float(cap))
        library = _ic._potential(library_rho,library_cost,cfg)
    if profile == 'stable64':
        return library
    if profile == 'matrix_scalar':
        rho, c = float(original), float(cost)
        result = float(np.sign(rho)*min(abs(rho)*float(np.exp(float(gamma)*abs(c))),float(cap)))
    elif profile == 'float32':
        rho = np.asarray(original)
        result = np.asarray(np.sign(rho)*np.minimum(np.abs(rho)*np.exp(float(gamma)*np.abs(cost)),float(cap)),dtype=np.float32)
    elif profile in ('direct64','simulation'):
        rho = np.asarray(original) if profile=='simulation' else np.asarray(original,dtype=np.float64)
        exponent = gamma*cost if profile=='simulation' else float(gamma)*cost
        if profile=='simulation':exponent=np.minimum(exponent,700)
        result = np.sign(rho)*np.minimum(np.abs(rho)*np.exp(exponent),float(cap))
    else:
        raise ValueError('Unknown numerical potential profile: '+str(profile))
    # Returning a declared legacy profile is intentional compatibility, not an
    # exception fallback. Validate equivalence against the old implementation;
    # a log-space reference is not an oracle for its overflow or dtype behavior.
    return result

def fuse_cached(r0, eta, potential):
    """Cached-feature adapter: IC has no public standalone fusion API."""
    return (1-eta)*r0+eta*potential

def backend_info():
    return dict(repository='https://github.com/zhangzihan-is-good/IC',commit=IC_COMMIT,
                version=IC_VERSION,source=str(_ROOT/'_vendor'/'IC'),
                library_functions=['rgic.core._gate','rgic.core._potential'],
                compatibility='historical sufficient-statistic, dtype, ordering and rounding adapters')
