"""Frozen-person temporal templates, with explicit disjoint train/query inputs.

No file IO, fitting across people, hidden session split, subject deletion, or
query-dependent template is performed here.  ``details=False`` skips retrieval
ID and large saved intermediates, while retaining the identical gap score.
Strength inputs are finite nonnegative scalar strengths, not signed fluxes.
Direction inputs are native signed fluxes in a fixed upper-triangle edge order.
"""
from functools import lru_cache
import numpy as np
from readouts import percentile, membership
from stable_sum import fsum_rows

DENSITIES = np.array([.005, .01, .025, .05], dtype=np.float64)
ID_TOLERANCE = 1e-12


def _inputs(train, query):
    a, b = np.asarray(train, np.float64), np.asarray(query, np.float64)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
        raise ValueError('Expected train (n,k,E) and query (n,q,E) with the same n,E')
    if a.shape[0] < 2 or min(a.shape[1], b.shape[1]) < 1 or a.shape[2] < 2:
        raise ValueError('At least two people/coordinates and one train/query scan required')
    return a, b


def _normalize(x, return_norm=False):
    centered = x - x.mean(axis=-1, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=-1, keepdims=True))
    valid = np.isfinite(x).all(axis=-1) & (norm[..., 0] > 0)
    z = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)
    return (z, valid, norm[..., 0]) if return_norm else (z, valid)


@lru_cache(maxsize=16)
def _incident(edges):
    nodes = int((1 + np.sqrt(1 + 8 * edges)) / 2)
    if nodes < 3 or nodes * (nodes - 1) // 2 != edges:
        raise ValueError('Local profiles require a complete graph with at least three nodes')
    ii, jj = np.triu_indices(nodes, 1)
    result = np.array([np.flatnonzero((ii == r) | (jj == r)) for r in range(nodes)])
    result.flags.writeable = False
    return result


def _retrieval(similarity):
    """Retrieve separately for every query scan; retain the complete gallery."""
    n, q, gallery = similarity.shape
    if gallery != n:
        raise ValueError('The query and template galleries must contain the same identities')
    valid = np.isfinite(similarity).all(axis=-1)
    safe = np.where(np.isfinite(similarity), similarity, -np.inf)
    maximum = np.where(valid[..., None], safe.max(axis=-1, keepdims=True), 0.)
    tied = (np.abs(safe - maximum) <= ID_TOLERANCE) & valid[..., None]
    count = tied.sum(axis=-1)
    own = tied[np.arange(n), :, np.arange(n)]
    per_query = np.divide(own, count, out=np.full((n, q), np.nan), where=valid & (count > 0))
    return per_query.mean(axis=1)


def _pattern(matrix, query_similarity=None):
    n = matrix.shape[0]
    S = np.diag(matrix).copy()
    D = matrix.copy()
    np.fill_diagonal(D, np.nan)
    off = ~np.eye(n, dtype=bool)
    meanS = float(S.mean()) if np.isfinite(S).all() else np.nan
    between = float(D[off].mean()) if np.isfinite(D[off]).all() else np.nan
    subject_ID = _retrieval(query_similarity) if query_similarity is not None else np.full(n, np.nan)
    return dict(matrix=matrix, S=S, D=D, meanS=meanS, betweenD=between,
                gap=meanS - between, gap_score=meanS - between,
                subject_ID=subject_ID, ID=float(subject_ID.mean()),
                eligible=bool(np.isfinite(matrix).all()))


def strength(train, query, local=False, details=True):
    """Rank-based train-person templates evaluated on later query scans.

    ``train`` is (n,k,E), ``query`` (n,q,E).  Each scan is independently average-
    tie percentile ranked and centered/L2-normalized.  The mean of a person's
    k normalized training profiles is centered/L2-normalized once more.  For
    local=True each ROI's incident-edge profile follows this rule independently;
    all ROIs receive the same weight.  Any constant required profile makes the
    whole result ineligible, with NA scores: no ROI/person is dropped.

    matrix[i,j] averages person i's q query similarities to person j's template.
    S is its diagonal, D its off-diagonal (diagonal NA).  ID averages per-query
    retrieval with 1e-12 ties split uniformly; ID is NA in the fast path.
    """
    a, b = _inputs(train, query)
    n, k, E = a.shape
    q = b.shape[1]
    result = dict(n_subjects=n, n_train=k, n_query=q, n_edges=E, local=bool(local),
                  template=None, query_profiles=None, query_similarity=None)
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.any(a < 0) or np.any(b < 0):
        result.update(_pattern(np.full((n, n), np.nan)), reason='nonfinite_or_negative_strength')
        return result
    if local:
        incident = _incident(E)
        a, b = a[..., incident], b[..., incident]
        result['n_nodes'] = len(incident)
    z_train, train_valid = _normalize(percentile(a))
    z_query, query_valid = _normalize(percentile(b))
    template, template_valid, template_norm = _normalize(z_train.mean(axis=1), return_norm=True)
    valid = bool(train_valid.all() and query_valid.all() and template_valid.all())
    result.update(valid_train=train_valid, valid_query=query_valid, valid_template=template_valid,
                  template_pre_norm=template_norm)
    if details:
        result.update(template=template, query_profiles=z_query)
    if not valid:
        result.update(_pattern(np.full((n, n), np.nan)), reason='constant_required_profile')
        return result
    if local:
        # Each of the R complete ROI profiles has equal weight, never pooled L2.
        matrix = np.einsum('ire,jre->ij', z_query.mean(axis=1), template) / template.shape[1]
        sims = np.einsum('iqre,jre->iqj', z_query, template) / template.shape[1] if details else None
        if details:
            result['ROI_S'] = np.einsum('iqre,ire->ir', z_query, template) / q
    else:
        matrix = z_query.mean(axis=1) @ template.T
        sims = np.matmul(z_query, template.T) if details else None
    result.update(_pattern(matrix, sims), reason='ok')
    if details:
        result['query_similarity'] = sims
    return result


def _integrate(values, densities):
    return np.trapz(values, densities, axis=0) / (densities[-1] - densities[0])


def _stack_patterns(patterns):
    keys = ('matrix', 'S', 'D', 'meanS', 'betweenD', 'gap', 'gap_score', 'subject_ID', 'ID')
    return {key: np.array([p[key] for p in patterns]) for key in keys}


def _auc(patterns, densities):
    result = _pattern(_integrate(patterns['matrix'], densities))
    # ID is the AUC of per-density, per-query retrieval; it is not retrieval
    # performed after averaging queries or after averaging density matrices.
    result['subject_ID'] = _integrate(patterns['subject_ID'], densities)
    result['ID'] = float(result['subject_ID'].mean())
    return result


def direction(trainflux, queryflux, densities=DENSITIES, details=True):
    """Quota-matched, magnitude-free arrow templates at fixed densities.

    For every scan, abs(flux) selects m=floor(density*E+.5) positive-confidence
    edges, splitting exact cutoff ties fractionally.  Magnitudes are then
    discarded: selected arrows are weight*sign(flux).  For each density the
    k training arrows are correctly rounded by math.fsum-equivalent summation
    per edge, then the same quota selects a frozen signed person template.
    Query scans independently select the same quota.  matrix[d,i,j] is their
    mean signed dot product / m, always retaining the full E-edge denominator.

    Coverage is the analogous unsigned weight dot/m; conditional orientation
    is signed_matrix/coverage_matrix (zero overlap -> NA).  That companion is
    overlap-weighted across queries, never a replacement for the primary yield.
    Any insufficient train/query scan OR consensus at ANY quota invalidates the
    entire density family, including its AUC.  No smaller quota/subject subset
    is substituted.  The AUC uses the actual density widths, normalized by the
    entire fixed density range.  Shapes lead with density; auc has no such axis.
    """
    a, b = _inputs(trainflux, queryflux)
    n, k, E = a.shape
    q = b.shape[1]
    d = np.asarray(densities, np.float64)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all() or np.any(np.diff(d) <= 0) or d[0] <= 0 or d[-1] > 1:
        raise ValueError('At least two strictly increasing positive densities <=1 are required')
    counts = np.floor(d * E + .5).astype(int)
    if np.any(counts < 1) or np.any(counts > E):
        raise ValueError('Every density must round to a positive feasible edge quota')
    result = dict(n_subjects=n, n_train=k, n_query=q, n_edges=E, densities=d.copy(), counts=counts,
                  template=None, template_weights=None, train_weights=None, query_weights=None,
                  weights=None, query_profiles=None, query_similarity=None, coverage_query_similarity=None)

    def invalid(reason):
        patterns = _stack_patterns([_pattern(np.full((n, n), np.nan)) for _ in d])
        result.update(patterns, eligible=False, reason=reason)
        result['coverage'] = dict(patterns)
        result['conditional_orientation'] = dict(patterns)
        result['auc'] = _auc(patterns, d)
        result['auc'].update(eligible=False, coverage=_auc(patterns, d), conditional_orientation=_auc(patterns, d))
        result['coverage_matrix'] = patterns['matrix'].copy()
        result['conditional_orientation_matrix'] = patterns['matrix'].copy()
        return result

    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return invalid('nonfinite_flux')
    result['train_active'] = (a != 0).sum(axis=-1)
    result['query_active'] = (b != 0).sum(axis=-1)
    if min(result['train_active'].min(), result['query_active'].min()) < counts.max():
        return invalid('insufficient_input_quota')
    signed_patterns, coverage_patterns, conditional_patterns = [], [], []
    templates, template_weights, train_weights, query_weights, query_profiles = [], [], [], [], []
    similarities, coverage_similarities, consensus_support = [], [], []
    for m in counts:
        wa, va = membership(np.abs(a), m)
        wb, vb = membership(np.abs(b), m)
        arrows = wa * np.sign(a)
        # The final axis contains k training terms only; no query is accessed.
        total = fsum_rows(np.moveaxis(arrows, 1, -1))
        wt, vt = membership(np.abs(total), m)
        consensus_support.append((total != 0).sum(axis=-1))
        if not (va.all() and vb.all() and vt.all()):
            result['consensus_active'] = np.array(consensus_support)
            return invalid('insufficient_consensus_quota')
        template = wt * np.sign(total)
        zq = wb * np.sign(b)
        matrix = zq.mean(axis=1) @ template.T / m
        coverage = wb.mean(axis=1) @ wt.T / m
        conditional = np.divide(matrix, coverage, out=np.full_like(matrix, np.nan), where=coverage > 0)
        sims = np.matmul(zq, template.T) / m if details else None
        csims = np.matmul(wb, wt.T) / m if details else None
        conditional_sims = (np.divide(sims, csims, out=np.full_like(sims, np.nan), where=csims > 0)
                            if details else None)
        signed_patterns.append(_pattern(matrix, sims))
        coverage_patterns.append(_pattern(coverage, csims))
        conditional_patterns.append(_pattern(conditional, conditional_sims))
        if details:
            templates.append(template); template_weights.append(wt)
            train_weights.append(wa); query_weights.append(wb); query_profiles.append(zq)
            similarities.append(sims); coverage_similarities.append(csims)
    signed = _stack_patterns(signed_patterns)
    unsigned = _stack_patterns(coverage_patterns)
    conditional = _stack_patterns(conditional_patterns)
    result.update(signed, eligible=True, reason='ok', coverage=unsigned,
                  conditional_orientation=conditional, coverage_matrix=unsigned['matrix'],
                  conditional_orientation_matrix=conditional['matrix'],
                  consensus_active=np.array(consensus_support))
    result['auc'] = _auc(signed, d)
    result['auc'].update(coverage=_auc(unsigned, d), conditional_orientation=_auc(conditional, d))
    if details:
        result.update(template=np.array(templates), template_weights=np.array(template_weights),
                      train_weights=np.array(train_weights), query_weights=np.array(query_weights),
                      query_profiles=np.array(query_profiles), query_similarity=np.array(similarities),
                      coverage_query_similarity=np.array(coverage_similarities))
        result['weights'] = dict(train=result['train_weights'], query=result['query_weights'],
                                 template=result['template_weights'])
    return result


def bootstrap_summary(pattern, counts):
    """Whole-person summaries of strength output or direction output['auc'].

    gap uses the existing query+gallery delete-one identity jackknife.  meanS
    resamples people; betweenD is the corresponding meanS-minus-gap bootstrap.
    ID resamples query people conditional on the original unique gallery, a
    distinct uncertainty convention.  This function performs no selection.
    """
    import statistics_hnu as rs
    S, D = np.asarray(pattern['S'], float), np.asarray(pattern['D'], float)
    c = np.asarray(counts, float)
    if S.ndim != 1 or len(S) < 3 or D.shape != (len(S), len(S)):
        raise ValueError('A single pattern with at least three people is required')
    n = len(S)
    if c.ndim != 2 or c.shape[1] != n or not np.isfinite(c).all() or np.any(c < 0) or np.any(c != np.floor(c)) or np.any(c.sum(axis=1) != n):
        raise ValueError('counts must be nonnegative whole-person bootstrap counts summing to n')
    gap = rs.gallery_gap(S, D, c)
    own = c @ S / n
    identity = c @ np.asarray(pattern['subject_ID'], float) / n
    return dict(meanS=dict(point=pattern['meanS'], bootstrap=own),
                betweenD=dict(point=pattern['betweenD'], bootstrap=own - gap['bootstrap']),
                gap=gap, ID=dict(point=pattern['ID'], bootstrap=identity))
