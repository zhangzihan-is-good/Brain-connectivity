"""HNU adapters to the shared IC implementation and matched comparators.

Public signatures and return layouts retain the fixed HNU experiment contract.
The shared module owns all RGIC equations and the optimized reference table.
"""
from pathlib import Path
import sys
import numpy as np

_SHARED_ROOT = Path(__file__).resolve().parents[1]
if str(_SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARED_ROOT))
from rgic_shared import (hnu_fls, hnu_gate, hnu_potential, hnu_derive,
                         HNUReferenceTable)

EPSILON = 1e-8


def fls(r0, rstar, lag, indices, offsets, delta=0):
    return hnu_fls(r0, rstar, lag, indices, offsets, delta)


def gate(r0, rs, lag, R, cfg):
    return hnu_gate(r0, rs, lag, R, cfg)


def potential(rs, lag, cfg):
    return hnu_potential(rs, lag, cfg)


def derive(r0, rs, lag, tie, R, cfg):
    return hnu_derive(r0, rs, lag, tie, R, cfg)


ReferenceTable = HNUReferenceTable

METHODS=['RGIC','Lag_peak','Gaussian_TE','VAR_Granger','PSI','PhiID_unique_transfer','Pearson','PureLag_including_zero','Coherence','GCMI']
def reference(p,ref,window,delta):return ReferenceTable(p,np.asarray(ref),window).evaluate(delta)
def matched(rgic,direction=False):
    w=rgic['parameters']['window'];two=w=='W2'
    rows=[rgic,dict(method='Lag_peak',config_id='lag_'+w,parameters=dict(window=w)),
        dict(method='Gaussian_TE',config_id='te_h1_'+('max12' if two else 'd1'),parameters=dict(history=1,delay_mode='max12' if two else 'd1')),
        dict(method='VAR_Granger',config_id='var_p'+('2' if two else '1'),parameters=dict(order=2 if two else 1)),
        dict(method='PSI',config_id='psi_fixed',parameters={}),
        dict(method='PhiID_unique_transfer',config_id='phiid_'+('max12' if two else 'd1'),parameters=dict(delay_mode='max12' if two else 'd1')),
        dict(method='Pearson',config_id='pearson_fixed',parameters={}),
        dict(method='PureLag_including_zero',config_id='purelag_'+w,parameters=dict(window=w)),
        dict(method='Coherence',config_id='coherence_fixed',parameters={}),
        dict(method='GCMI',config_id='gcmi_clip0',parameters=dict(strength_policy='clip0'))]
    if direction:
        rows=rows[:6]
        for c in rows[2:]:
            if c['method']!='PSI':c['parameters']['direction_policy']='signed_native';c['config_id']+='_signed_native'
    return rows

def scalar(p,indices,config,R_reference=None):
    """Return nonnegative(records,5886) scalar strength, not directional H."""
    method=config['method'];cfg=config['parameters'];ix=np.asarray(indices)
    if method=='RGIC':
        if R_reference is None:raise ValueError('Reference-only common R is required')
        w=cfg['window']
        return np.abs(derive(p['r0'][ix],p[w+'_rstar'][ix],p[w+'_lag'][ix],p[w+'_tie'][ix],R_reference,cfg)['IC'])
    if method=='Pearson':return np.abs(p['r0'][ix])
    if method=='Lag_peak':return np.abs(p[cfg['window']+'_rstar'][ix])
    if method=='PureLag_including_zero':return np.maximum(np.abs(p['r0'][ix]),np.abs(p[cfg['window']+'_rstar'][ix]))
    if method=='Gaussian_TE':
        prefix=f'te_h{cfg["history"]}_'
        h=p[prefix+cfg['delay_mode']][ix] if cfg['delay_mode']!='max12' else np.maximum(p[prefix+'d1'][ix],p[prefix+'d2'][ix])
        return np.abs(h).max(-1)
    if method=='VAR_Granger':return np.abs(p[f'var_p{cfg["order"]}'][ix]).max(-1)
    if method=='PhiID_unique_transfer':
        choice=cfg['delay_mode']
        h=np.abs(p['phiid_'+choice][ix]) if choice!='max12' else np.maximum(np.abs(p['phiid_d1'][ix]),np.abs(p['phiid_d2'][ix]))
        return h.max(-1)
    if method in ['PSI','Coherence','GCMI']:
        return np.abs(p[{'PSI':'psi','Coherence':'coherence','GCMI':'gcmi'}[method]][ix])
    raise ValueError(method)

def strength(p,ix,c,R=None):
    if c["method"]=="GCMI":return np.maximum(p["gcmi"][np.asarray(ix)],0.)
    return scalar(p,ix,c,R)
def evidence(p,ix,c):
    """Two cached ordered directional scores, before contrast policy."""
    cfg=c['parameters'];method=c['method'];ix=np.asarray(ix)
    if method=='VAR_Granger':return p[f'var_p{cfg["order"]}'][ix]
    if method=='Gaussian_TE':
        prefix=f'te_h{cfg["history"]}_';choice=cfg['delay_mode']
        if choice!='max12':return p[prefix+choice][ix]
        a,b=p[prefix+'d1'][ix],p[prefix+'d2'][ix]
    elif method=='PhiID_unique_transfer':
        choice=cfg['delay_mode']
        if choice!='max12':return p['phiid_'+choice][ix]
        a,b=p['phiid_d1'][ix],p['phiid_d2'][ix]
    else:raise ValueError(method)
    # Preserve the historical magnitude maxima. For signed PhiID, select the
    # original signed atom of greatest absolute magnitude, per channel;
    # exact delay ties choose d1. Taking max(raw negative atoms) is different.
    if cfg.get('direction_policy','signed_native')=='legacy_abs':
        return np.maximum(np.abs(a),np.abs(b))
    if method=='PhiID_unique_transfer':return np.where(np.abs(a)>=np.abs(b),a,b)
    return np.maximum(a,b)


def flux(p,ix,c,R=None,prepared=None):
    """Signed canonical i<j evidence, no zero-lag strength fallback."""
    cfg=c['parameters'];method=c['method'];ix=np.asarray(ix)
    if method in ['RGIC','Lag_peak']:
        w=cfg['window'];direction=np.where(p[w+'_tie'][ix],0,np.sign(p[w+'_lag'][ix]))
        if method=='Lag_peak':return direction*np.abs(p[w+'_rstar'][ix])
        if prepared is None:
            if R is None:raise ValueError('RGIC needs a reference-only R or prepared (R,eta)')
            eta=gate(p['r0'][ix],p[w+'_rstar'][ix],p[w+'_lag'][ix],R,cfg)['eta']
        else:R,eta=prepared
        return direction*eta*np.abs(potential(p[w+'_rstar'][ix],p[w+'_lag'][ix],cfg))
    if method=='PSI':return p['psi'][ix]
    value=evidence(p,ix,c)
    if cfg.get('direction_policy','signed_native')=='legacy_abs':value=np.abs(value)
    return value[...,0]-value[...,1]
