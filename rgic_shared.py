"""One RGIC interface for all four article experiments.

The pinned IC backend and numerical profiles are described in README.md.
Profiles deliberately retain each experiment's existing scientific behavior.
"""
import numpy as np
from rgic_backend import (RGIC,RGICConfig,RGICResult,normalise_evidence,
                          compensated_peak,fuse_cached,backend_info)

def fls_components(tau,peak,gain,offsets):
    valid=gain>0
    count=valid.sum(axis=0)
    counts=np.stack([((tau==lag)&valid).sum(axis=0) for lag in offsets])
    signs=(np.sign(peak)*valid).sum(axis=0)
    F=count/len(tau)
    L=np.divide(counts.max(axis=0),count,out=np.zeros_like(F),where=count>0)
    S=np.divide(np.abs(signs),count,out=np.zeros_like(F),where=count>0)
    return dict(F=F,L=L,S=S,R=F*L*S,m_positive=count,signed_lag_counts=counts,sign_sum=signs)

def simulation_estimate(stats,R,method,tr,lam=1.,gamma=.2,seconds_parameters=True):
    """Simulation's prevalidated paired-edge profile, including ablations."""
    r0,peak,tau,G,C=[stats[k] for k in ('r0','peak','tau','G','C')]
    cost=C*tr if seconds_parameters else C
    P=compensated_peak(peak,cost,gamma,.9,profile='simulation')
    A=np.zeros_like(r0);Q=np.zeros(r0.shape[:-1]+(1,))
    if method=='Pearson':
        IC,eta=r0.copy(),np.zeros_like(r0)
    elif method=='Lag peak':
        IC,eta=peak.copy(),np.ones_like(r0)
    else:
        rr=np.ones_like(R) if method=='No reliability' else R
        gg=np.ones_like(G) if method=='No gain' else G
        discount=np.ones_like(C) if method=='No cost' else np.exp(-lam*cost)
        A=rr*gg*discount
        eta,q=normalise_evidence(A)
        Q=q[...,None]
        if method=='No compensation':
            P=np.sign(peak)*np.minimum(np.abs(peak),.9)
        IC=fuse_cached(r0,eta,P)
    D=np.sign(tau)*np.abs(IC-r0)
    return dict(strength=np.abs(IC),signed_strength=IC,direction=D,lag=tau*tr,eta=eta,A=A,
                Q=np.broadcast_to(Q,r0.shape),R=R,G=G,peak=peak,r0=r0,adjustment=np.abs(IC-r0))

def hnu_fls(r0,rstar,lag,indices,offsets,delta=0):
    r0=np.asarray(r0,dtype=np.float64);rstar=np.asarray(rstar,dtype=np.float64)
    lag=np.asarray(lag);indices=np.asarray(indices,dtype=np.int64)
    if indices.ndim!=1 or not len(indices): raise ValueError('FLS requires at least one reference index')
    if r0.ndim!=2 or r0.shape!=rstar.shape or r0.shape!=lag.shape: raise ValueError('Expected equal (reference records, edges) arrays')
    if not np.isfinite(delta) or delta<0: raise ValueError('delta must be finite and nonnegative')
    if np.any(indices<0) or np.any(indices>=len(r0)) or len(np.unique(indices))!=len(indices): raise ValueError('Invalid or duplicate reference indices')
    offsets=tuple(offsets)
    if not offsets or len(set(offsets))!=len(offsets): raise ValueError('Distinct signed lag offsets are required')
    selected=lag[indices]
    gain=np.maximum(0.,np.abs(rstar[indices])-np.abs(r0[indices])-delta)
    if np.any((gain>0)&~np.isin(selected,offsets)): raise ValueError('Positive-gain reference lag lies outside supplied offsets')
    result=fls_components(selected,rstar[indices],gain,offsets)
    result.update(N_ref=np.array(len(indices)),reference_indices=indices.copy())
    return result

def hnu_gate(r0,rs,lag,R,cfg):
    r0=np.asarray(r0,dtype=np.float64);rs=np.asarray(rs,dtype=np.float64)
    C=np.abs(np.asarray(lag,dtype=np.float64))
    G=np.maximum(0.,np.abs(rs)-np.abs(r0)-float(cfg.get('delta',0.)))
    A=G*np.exp(-float(cfg.get('lambda',1.))*C)*np.asarray(R,dtype=np.float64)
    # Keep the old scalar/array return type and the old NumPy deprecation
    # behavior (including warnings-as-errors). The library still computes eta.
    q=np.quantile(A,.95,axis=-1,interpolation='linear')
    eta=np.clip(A/(q[...,None]+1e-8),0.,1.)
    with np.errstate(all='ignore'):
        library_eta,_=normalise_evidence(A)
    np.copyto(eta,library_eta)
    return dict(G=G,A=A,Q95=q,eta=eta)

def hnu_potential(rs,lag,cfg):
    return compensated_peak(np.asarray(rs,dtype=np.float64),np.abs(np.asarray(lag,dtype=np.float64)),
                            float(cfg.get('gamma',.2)),float(cfg.get('cap',.9)))

def hnu_derive(r0,rs,lag,tie,R,cfg):
    r0=np.asarray(r0,dtype=np.float64)
    result=hnu_gate(r0,rs,lag,R,cfg);eta=result['eta'];P=hnu_potential(rs,lag,cfg)
    IC=fuse_cached(r0,eta,P)
    direction=np.where(np.asarray(tie,dtype=bool),0,np.sign(lag))
    D=direction*np.abs(IC-r0)
    H=np.stack([np.where(direction>0,eta*np.abs(P),0),np.where(direction<0,eta*np.abs(P),0)],axis=-1)
    return dict(IC=IC,D=D,H=H,P=P,**result)

class HNUReferenceTable:
    """Historical cumulative reference cache, retaining exact operation order."""
    def __init__(self,p,ref,window):
        gain=np.abs(p[window+'_rstar'][ref])-np.abs(p['r0'][ref])
        order=np.argsort(-gain,axis=0,kind='stable')
        self.gain=np.take_along_axis(gain,order,axis=0)
        lags=np.take_along_axis(p[window+'_lag'][ref],order,axis=0)
        signs=np.take_along_axis(np.sign(p[window+'_rstar'][ref]),order,axis=0)
        n,e=gain.shape;m=np.arange(1,n+1)[:,None]
        offsets=[-1,1] if window=='W1' else [-2,-1,1,2]
        maxcounts=np.maximum.reduce([np.cumsum(lags==offset,axis=0) for offset in offsets])
        signsum=np.cumsum(signs,axis=0)
        F=np.broadcast_to(m/n,gain.shape)
        L=maxcounts/m;S=np.abs(signsum)/m
        self.R=np.concatenate([np.zeros((1,e)),F*L*S],axis=0)
        self.columns=np.arange(e)
        self.gain.setflags(write=False);self.R.setflags(write=False)
    def evaluate(self,delta):
        m=np.sum(self.gain>float(delta),axis=0)
        return self.R[m,self.columns]

# ABIDE's float32/group gate is a separate explicit historical profile.
from rgic_profiles_abide import (abide_gate_components,abide_eta,abide_potential,
                                abide_values,abide_group_eta,abide_matrix_potential,
                                abide_matrix_blend,abide_evidence,abide_direction)
