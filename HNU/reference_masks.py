"""Reference-only masks, complete-density integration and exact node roles."""
import numpy as np
from scipy.sparse import csr_matrix
import rgic
from stable_sum import fsum_rows
PRIORITY=np.random.default_rng(20261024).permutation(5886)
DENSITIES=np.array([.005,.01,.025,.05],dtype=np.float64)

def masks(p,ref,w):
    """All strata fixed by independent reference; delta is always0 for masks."""
    gain=np.mean(np.maximum(np.abs(p[w+'_rstar'][ref])-np.abs(p['r0'][ref]),0.),axis=0)
    R=rgic.ReferenceTable(p,ref,w).evaluate(0.)
    scores={'laggain':gain,'FLS':R,'laggain_FLS':gain*R}
    result={}
    for name,value in scores.items():
        order=np.lexsort((PRIORITY,-value))
        result[name]=[np.array(part,dtype=np.int32) for part in np.array_split(order,5)]
    return result,scores



def integrate(curves, densities=DENSITIES):
    """Normalized complete-range trapezoid; any missing quota stays missing."""
    return np.trapz(np.asarray(curves, float), densities, axis=0) / (densities[-1] - densities[0])



def incidence(n_nodes):
    ii, jj = np.triu_indices(n_nodes, 1)
    rows = np.repeat(np.arange(len(ii)), 2)
    cols = np.column_stack([ii, jj]).ravel()
    signed = csr_matrix((np.tile([1., -1.], len(ii)), (rows, cols)), shape=(len(ii), n_nodes))
    unsigned = csr_matrix((np.ones(2 * len(ii)), (rows, cols)), shape=(len(ii), n_nodes))
    return signed, unsigned



def roles(weighted_sign, weights, n_nodes):
    signed, unsigned = incidence(n_nodes)
    net = np.asarray(signed.T.dot(weighted_sign.T).T)
    degree = np.asarray(unsigned.T.dot(weights.T).T)
    # Binary rows sum exactly in binary64. Fractional rows use the same strict
    # math.fsum-equivalent rule for all methods, avoiding fake nearzero roles.
    fraction = np.any((weights != 0) & (weights != 1), axis=-1)
    if fraction.any():
        ii, jj = np.triu_indices(n_nodes, 1)
        incident = np.array([np.flatnonzero((ii == node) | (jj == node)) for node in range(n_nodes)])
        outward = np.where(ii[incident] == np.arange(n_nodes)[:, None], 1., -1.)
        net[fraction] = fsum_rows(weighted_sign[fraction][:, incident] * outward)
        degree[fraction] = fsum_rows(weights[fraction][:, incident])
    return net, degree

