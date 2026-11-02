"""ABIDE numerical profiles for the shared, pinned-IC backend.

The IC library supplies potential and gate normalization. This profile retains
the ABIDE study's float32 reduction order, training-group gate, alpha mixing,
cached-potential branch, and original scalar-matrix rounding conventions.
"""
from __future__ import annotations
import numpy as np
from rgic_backend import normalise_evidence, compensated_peak, fuse_cached


def abide_gate_components(gain, cost, lag, rho):
    gain,cost,rho=[np.asarray(a,dtype=np.float32) for a in (gain,cost,rho)]
    lag=np.asarray(lag)
    valid=gain>0
    counts=valid.sum(0)
    frequency=valid.mean(0).astype(np.float32)
    max_count=np.zeros(gain.shape[1],dtype=np.int64)
    for value in np.unique(lag):
        max_count=np.maximum(max_count,((lag==value)&valid).sum(0))
    consistency=np.divide(max_count,counts,out=np.zeros(len(counts),dtype=np.float64),where=counts>0).astype(np.float32)
    sign_sum=(np.sign(rho)*valid).sum(0,dtype=np.float32)
    signs=(np.abs(sign_sum)/(counts.astype(np.float64)+1e-8)).astype(np.float32)
    mean_gain=gain.mean(0).astype(np.float32)
    mean_cost=((cost*valid.astype(np.float32)).sum(0)/(counts.astype(np.float32)+1e-8)).astype(np.float32)
    reliability=(frequency*consistency*signs).astype(np.float32)
    return dict(mean_gain=mean_gain,mean_cost=mean_cost,reliability=reliability,
                frequency=frequency,lag_consistency=consistency,sign_consistency=signs)


def abide_eta(components,lambda_cost,percentile,eps=1e-8):
    evidence=(components['mean_gain']*np.exp(-float(lambda_cost)*components['mean_cost'])).astype(np.float32)
    raw=(components['reliability']*evidence).astype(np.float32)
    eta,_=normalise_evidence(raw,percentile=percentile,epsilon=eps,percentile_api=True)
    return np.asarray(eta,dtype=np.float32)


def abide_potential(rho,lag,gamma,cap,profile='float32'):
    # Original broad helper does not eagerly cast rho; the profile's 'float32'
    # matches the historical NumPy expression for the stored float32 primitives.
    return np.asarray(compensated_peak(rho,np.abs(lag),gamma,cap,profile=profile),dtype=np.float32)


def abide_values(r0,rho,lag,eta,core,cached_potential=None):
    r0=np.asarray(r0,dtype=np.float32)
    if core['alpha']==0:return r0,np.zeros_like(r0)
    if core['gamma']==.2 and core['cap']==.9 and cached_potential is not None:
        potential=np.asarray(cached_potential,dtype=np.float32)
    else:
        profile='float32' if core['gamma']<=32 else 'stable64'
        potential=abide_potential(np.asarray(rho,dtype=np.float32),np.asarray(lag),core['gamma'],core['cap'],profile=profile)
    eta=np.asarray(eta,dtype=np.float32)
    if core['alpha']==1:
        ic=np.asarray(fuse_cached(r0,eta,potential),dtype=np.float32)
    else:
        ic=(r0+np.float32(core['alpha'])*eta*(potential-r0)).astype(np.float32)
    direction=(np.sign(lag)*np.abs(ic-r0)).astype(np.float32)
    return ic,direction


def abide_group_eta(gain,cost,lag,rho,lambda_cost=1.,percentile=95.,eps=1e-8):
    """Original rgc_ic/report_data loop profile, including float32 sign sums."""
    gain,cost,lag,rho=[np.asarray(a,dtype=np.float32) for a in (gain,cost,lag,rho)]
    if gain.ndim!=2 or not(gain.shape==cost.shape==lag.shape==rho.shape):raise ValueError('Bad group eta input')
    valid=gain>0
    frequency=valid.mean(axis=0).astype(np.float32)
    count=valid.sum(axis=0)
    lag_consistency=np.zeros(gain.shape[1],dtype=np.float32)
    sign_consistency=np.zeros(gain.shape[1],dtype=np.float32)
    for edge in range(gain.shape[1]):
        mask=valid[:,edge]
        if not np.any(mask):continue
        _,counts=np.unique(lag[mask,edge],return_counts=True)
        lag_consistency[edge]=float(counts.max()/mask.sum())
        signs=np.sign(rho[mask,edge])
        sign_consistency[edge]=abs(float(signs.sum())/(float(mask.sum())+float(eps)))
    mean_gain=gain.mean(axis=0).astype(np.float32)
    valid_counts=count.astype(np.float32)
    mean_cost=((cost*valid.astype(np.float32)).sum(axis=0)/(valid_counts+float(eps))).astype(np.float32)
    reliability=(frequency*lag_consistency*sign_consistency).astype(np.float32)
    evidence=(mean_gain*np.exp(-float(lambda_cost)*mean_cost)).astype(np.float32)
    raw_eta=(reliability*evidence).astype(np.float32)
    eta,q_eta=normalise_evidence(raw_eta,percentile=percentile,epsilon=eps,percentile_api=True)
    return dict(eta=np.asarray(eta,dtype=np.float32),raw_eta=raw_eta,reliability=reliability,
                frequency=frequency,lag_consistency=lag_consistency,sign_consistency=sign_consistency,
                mean_gain=mean_gain,mean_cost=mean_cost,q_eta=float(q_eta))


def abide_matrix_potential(rho,lag,gamma,cap):
    """Float64 scalar compensation before the original float32 matrix storage."""
    return float(compensated_peak(rho,abs(lag),gamma,cap,profile='matrix_scalar'))


def abide_matrix_blend(pearson,potential,eta):
    return np.asarray(fuse_cached(pearson,eta,potential),dtype=np.float32)


def abide_evidence(components,lambda_cost):
    """One float32 expression shared by ablation diagnostics and normal gates."""
    return (components['mean_gain']*np.exp(-float(lambda_cost)*components['mean_cost'])).astype(np.float32)


def abide_direction(lag,delta):
    return (np.sign(lag)*np.abs(delta)).astype(np.float32)
