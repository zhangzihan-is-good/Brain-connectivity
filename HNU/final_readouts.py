"""Paired repeated-scan and frozen-person-template readouts, without IO.

``values_train`` and ``values_query`` are person, scan, upper-triangle edge.
Retest consumes only query scans (or train if query is None); temporal never
uses a query in its template. Strengths are nonnegative native magnitudes.
Direction inputs are canonical signed native fluxes: magnitude selects edges,
then scores contain only fractional membership times a unit arrow.
"""
import numpy as np
import temporal_readouts as tr
import statistics_hnu as rs
from readouts import percentile, membership
from stable_sum import fsum_rows

DENSITIES = tr.DENSITIES
_PATTERN_KEYS = ('matrix', 'S', 'D', 'meanS', 'betweenD', 'gap', 'gap_score',
                 'subject_ID', 'ID')


def _inputs(train, query, mode):
    if mode not in ('temporal', 'retest'):
        raise ValueError('mode must be temporal or retest')
    if mode == 'retest':
        b = np.asarray(train if query is None else query, np.float64)
        if b.ndim != 3 or b.shape[0] < 2 or b.shape[1] < 2 or b.shape[2] < 2:
            raise ValueError('Retest requires (n>=2,k>=2,E>=2)')
        return b, b
    return tr._inputs(train, query)


def _invalid(n, reason):
    out = tr._pattern(np.full((n, n), np.nan))
    out['reason'] = reason
    return out


def _retest_pattern(z, details):
    """All ordered a!=b pairs, including the same pair rule for D and ID."""
    n, k, f = z.shape
    if details:
        sims = np.stack([z[:, a] @ z[:, b].T for a in range(k)
                         for b in range(k) if a != b], axis=1)
        result = tr._pattern(sims.mean(axis=1), sims)
        result['query_similarity'] = sims
    else:
        total = z.sum(axis=1)
        matrix = total @ total.T
        for a in range(k):
            matrix -= z[:, a] @ z[:, a].T
        result = tr._pattern(matrix / (k * (k - 1)))
    return result


def _strength(a, b, mode, kind, r0a, r0b, mask, details):
    n, k, E = b.shape
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.any(a < 0) or np.any(b < 0):
        return _invalid(n, 'nonfinite_or_negative_strength')
    if kind in ('highgain', 'gain_fls'):
        if mask is None:
            raise ValueError('High-gain readouts require a common reference-only edge mask')
        index = np.asarray(mask)
        a, b = a[..., index], b[..., index]
        if a.shape[-1] < 2:
            return _invalid(n, 'insufficient_fixed_mask')
    if kind == 'polarity':
        if mode == 'retest' and r0b is None:
            r0b = r0a
        if r0b is None or (mode == 'temporal' and r0a is None):
            raise ValueError('Polarity requires matching zero-lag correlations')
        rb = np.asarray(r0b, np.float64)
        ra = rb if mode == 'retest' else np.asarray(r0a, np.float64)
        if ra.shape != a.shape or rb.shape != b.shape or not (np.isfinite(ra).all() and np.isfinite(rb).all()):
            raise ValueError('r0 must be finite and match each strength input')
        # This is a common zero-lag polarity, never a native direction arrow.
        a, b = np.sign(ra) * a, np.sign(rb) * b
    elif mode == 'temporal':
        return tr.strength(a, b, local=(kind == 'local'), details=details)
    local = kind == 'local'
    if local:
        incident = tr._incident(E)
        a, b = a[..., incident], b[..., incident]
    za, va = tr._normalize(percentile(a))
    zb, vb = tr._normalize(percentile(b))
    result = dict(n_subjects=n, n_train=a.shape[1], n_query=k,
                  valid_train=va, valid_query=vb)
    if mode == 'temporal':
        template, vt, norm = tr._normalize(za.mean(axis=1), return_norm=True)
        result.update(valid_template=vt, template_pre_norm=norm)
        if not (va.all() and vb.all() and vt.all()):
            result.update(_invalid(n, 'constant_required_profile'))
            return result
        matrix = zb.mean(1) @ template.T
        sims = zb @ template.T if details else None
        result.update(tr._pattern(matrix, sims))
        if details:
            result.update(template=template, query_profiles=zb, query_similarity=sims)
    else:
        if not vb.all():
            result.update(_invalid(n, 'constant_required_profile'))
            return result
        if local:
            R = zb.shape[2]
            if details:
                total = zb.sum(axis=1)
                result['ROI_S'] = ((total * total).sum(-1) - (zb * zb).sum((1, -1))) / (k * (k - 1))
            z = zb.reshape(n, k, -1) / np.sqrt(float(R))
        else:
            z = zb
        result.update(_retest_pattern(z, details))
        if details:
            result['query_profiles'] = zb
    result['reason'] = 'ok'
    return result


def _counts(E):
    counts = np.floor(DENSITIES * E + .5).astype(int)
    if np.any(counts < 1) or np.any(counts > E):
        raise ValueError('Every density must round to a positive feasible quota')
    return counts


def _bundle(patterns, counts, reason='ok'):
    density = tr._stack_patterns(patterns)
    result = tr._auc(density, DENSITIES)
    for key in _PATTERN_KEYS:
        result['density_' + key] = density[key]
    result.update(densities=DENSITIES.copy(), counts=np.asarray(counts).copy(),
                  density_eligible=np.array([p['eligible'] for p in patterns]), reason=reason)
    result['eligible'] = bool(all(p['eligible'] for p in patterns))
    return result


def _invalid_direction(n, counts, reason):
    return {key: _bundle([_invalid(n, reason) for _ in counts], counts, reason)
            for key in ('signed', 'coverage', 'conditional')}


def _conditional(matrix, coverage, sims=None, coverage_sims=None):
    ratio = np.divide(matrix, coverage, out=np.full_like(matrix, np.nan), where=coverage > 0)
    qs = (np.divide(sims, coverage_sims, out=np.full_like(sims, np.nan), where=coverage_sims > 0)
          if sims is not None else None)
    return tr._pattern(ratio, qs)


def _pair_pattern(ua, ub, wa, wb, count, mode, details):
    n, k, E = ua.shape
    q = ub.shape[1]
    if mode == 'temporal':
        matrix = ub.mean(1) @ ua.mean(1).T / count
        coverage = wb.mean(1) @ wa.mean(1).T / count
        pairs = [(b, a) for b in range(q) for a in range(k)]
    else:
        total, ctotal = ub.sum(1), wb.sum(1)
        matrix, coverage = total @ total.T, ctotal @ ctotal.T
        for s in range(q):
            matrix -= ub[:, s] @ ub[:, s].T
            coverage -= wb[:, s] @ wb[:, s].T
        denominator = q * (q - 1) * count
        matrix, coverage = matrix / denominator, coverage / denominator
        pairs = [(b, a) for b in range(q) for a in range(q) if a != b]
    sims = (np.stack([ub[:, b] @ ua[:, a].T / count for b, a in pairs], axis=1)
            if details else None)
    csims = (np.stack([wb[:, b] @ wa[:, a].T / count for b, a in pairs], axis=1)
             if details else None)
    if details:
        # Nonnegative explicit overlap avoids total-minus-same cancellation at
        # mathematically zero overlap; ID and the point use the same pairs.
        matrix, coverage = sims.mean(axis=1), csims.mean(axis=1)
    return (tr._pattern(matrix, sims), tr._pattern(coverage, csims),
            _conditional(matrix, coverage, sims, csims), sims, csims)


def _direction(a, b, mode, kind, details):
    n, k, E = a.shape
    q = b.shape[1]
    counts = _counts(E)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return _invalid_direction(n, counts, 'nonfinite_flux')
    if min((a != 0).sum(-1).min(), (b != 0).sum(-1).min()) < counts.max():
        return _invalid_direction(n, counts, 'insufficient_input_quota')
    if kind == 'consensus' and mode == 'temporal':
        original = tr.direction(a, b, details=details)
        out = {}
        for key, source in [('signed', original), ('coverage', original['coverage']),
                            ('conditional', original['conditional_orientation'])]:
            ps = []
            for d in range(len(counts)):
                p = {name: source[name][d] for name in _PATTERN_KEYS}
                p['eligible'] = bool(np.isfinite(p['matrix']).all())
                ps.append(p)
            out[key] = _bundle(ps, counts, original['reason'])
        if details:
            for name in ('template', 'template_weights', 'train_weights', 'query_weights',
                         'query_profiles', 'query_similarity', 'coverage_query_similarity'):
                out['signed'][name] = original.get(name)
        return out
    signed, coverages, conditionals = [], [], []
    saved = {key: [] for key in ('train_weights', 'query_weights', 'template',
                                'template_weights', 'query_similarity', 'coverage_query_similarity')}
    consensus_active = []
    for count in counts:
        wa, va = membership(np.abs(a), count)
        wb, vb = membership(np.abs(b), count)
        ua, ub = wa * np.sign(a), wb * np.sign(b)
        if kind == 'pairwise':
            p, c, cond, sims, csims = _pair_pattern(ua, ub, wa, wb, count, mode, details)
        else:
            totals = np.stack([fsum_rows(np.moveaxis(ub[:, np.arange(q) != s], 1, -1))
                               for s in range(q)], axis=1)
            wt, vt = membership(np.abs(totals), count)
            consensus_active.append((totals != 0).sum(-1))
            if not vt.all():
                out = _invalid_direction(n, counts, 'insufficient_consensus_quota')
                out['signed']['consensus_active'] = np.asarray(consensus_active)
                return out
            template = wt * np.sign(totals)
            # Gallery identity j always omits the same numbered scan s as i.
            matrix = np.einsum('isf,jsf->ij', ub, template) / (q * count)
            coverage = np.einsum('isf,jsf->ij', wb, wt) / (q * count)
            sims = (np.einsum('isf,jsf->isj', ub, template) / count if details else None)
            csims = (np.einsum('isf,jsf->isj', wb, wt) / count if details else None)
            p, c = tr._pattern(matrix, sims), tr._pattern(coverage, csims)
            cond = _conditional(matrix, coverage, sims, csims)
            if details:
                saved['template'].append(template)
                saved['template_weights'].append(wt)
        signed.append(p); coverages.append(c); conditionals.append(cond)
        if details:
            saved['train_weights'].append(wa); saved['query_weights'].append(wb)
            saved['query_similarity'].append(sims); saved['coverage_query_similarity'].append(csims)
    out = dict(signed=_bundle(signed, counts), coverage=_bundle(coverages, counts),
               conditional=_bundle(conditionals, counts))
    out['signed'].update(train_active=(a != 0).sum(-1), query_active=(b != 0).sum(-1))
    if consensus_active:
        out['signed']['consensus_active'] = np.asarray(consensus_active)
    if details:
        for key, value in saved.items():
            if value:
                out['signed'][key] = np.asarray(value)
    return out


def _cosine(x):
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    valid = np.isfinite(x).all(-1) & (norm[..., 0] > 0)
    z = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    return z, valid, norm[..., 0]


def _role_pattern(a, b, mode, details):
    n = len(b)
    za, va, _ = _cosine(a)
    zb, vb, _ = _cosine(b)
    if mode == 'retest':
        if not vb.all():
            return _invalid(n, 'zero_role_profile')
        out = _retest_pattern(zb, details)
    else:
        template, vt, norm = _cosine(za.mean(1))
        if not (va.all() and vb.all() and vt.all()):
            return _invalid(n, 'zero_role_profile_or_template')
        sims = zb @ template.T if details else None
        out = tr._pattern(zb.mean(1) @ template.T, sims)
        out['template_pre_norm'] = norm
        if details:
            out.update(template=template, query_profiles=zb, query_similarity=sims)
    return out


def _roles(a, b, mode, details):
    # Use the frozen canonical incidence and exact fractional summation rule.
    from reference_masks import roles
    n, k, E = a.shape
    q = b.shape[1]
    incident = tr._incident(E)
    nodes = len(incident)
    counts = _counts(E)
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or min((a != 0).sum(-1).min(), (b != 0).sum(-1).min()) < counts.max():
        return {key: _bundle([_invalid(n, 'insufficient_input_quota') for _ in counts], counts,
                             'insufficient_input_quota') for key in ('net_cosine', 'degree_cosine')}
    patterns = dict(net_cosine=[], degree_cosine=[])
    ys = dict(net_cosine=[], degree_cosine=[])
    for count in counts:
        wa, _ = membership(np.abs(a), count)
        wb, _ = membership(np.abs(b), count)
        na, da = roles((wa * np.sign(a)).reshape(-1, E), wa.reshape(-1, E), nodes)
        nb, db = roles((wb * np.sign(b)).reshape(-1, E), wb.reshape(-1, E), nodes)
        for key, x, y in [('net_cosine', na, nb), ('degree_cosine', da, db)]:
            x, y = x.reshape(n, k, nodes), y.reshape(n, q, nodes)
            patterns[key].append(_role_pattern(x, y, mode, details))
            # A two-block ICC and a five-scan ICC are distinct readouts.
            ys[key].append(y if mode == 'retest' else np.stack([x.mean(1), y.mean(1)], axis=1))
    out = {}
    for key in patterns:
        out[key] = _bundle(patterns[key], counts)
        trace_y = np.asarray(ys[key])
        traces = np.array([rs.trace_statistics(y)['point'] for y in trace_y])
        out[key].update(trace_y=trace_y, density_traceICC=traces,
                        traceICC_auc=float(tr._integrate(traces, DENSITIES)),
                        trace_kind=('retest_scan_traceICC' if mode == 'retest' else 'temporal_blockmean_traceICC'))
    return out


def analyze(values_train, values_query, mode, kind, r0_train=None, r0_query=None,
            mask=None, details=True):
    """Return representation -> full-gallery pattern, with no silent deletion.

    Strength keys are the requested kind. Direction keys are signed, coverage,
    conditional; each is an actual-density-normalized AUC and retains all four
    density_* patterns. ID is retrieval per single query/comparison before any
    scan or density averaging. Conditional orientation is the signed/unsigned
    overlap ratio (zero overlap NA), not another primary success score.

    highgain/gain_fls mask is a single reference-frozen edge subset, never a
    target-derived subset. Polarity uses sign(r0)*strength before ranks and is
    not a direction readout. Roles use L2 cosine without centering, plus an
    explicitly named raw-count trace ICC; isolated coordinates remain present.
    """
    a, b = _inputs(values_train, values_query, mode)
    if kind in ('global', 'local', 'highgain', 'gain_fls', 'polarity'):
        return {kind: _strength(a, b, mode, kind, r0_train, r0_query, mask, details)}
    if kind in ('pairwise', 'consensus'):
        return _direction(a, b, mode, kind, details)
    if kind == 'roles':
        return _roles(a, b, mode, details)
    raise ValueError('Unknown kind: ' + str(kind))
