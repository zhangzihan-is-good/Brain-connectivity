"""Original whole-person/JK/bootstrap arithmetic; no file IO."""
import numpy as np
def bootstrap_indices(n,B=5000,seed=20265024):
    ix=np.random.default_rng(seed).integers(0,n,(B,n));counts=np.zeros((B,n),np.uint16)
    np.add.at(counts,(np.arange(B)[:,None],ix),1)
    return ix,counts


def trace_statistics(y):
    y=np.asarray(y,np.float64);n,k,f=y.shape
    assert n>=2 and k>=2 and np.isfinite(y).all()
    means=y.mean(1);centered=means-means.mean(0)
    gram=centered@centered.T
    within=np.sum((y-means[:,None,:])**2,axis=(1,2))
    # Exact repeated profiles can acquire tiny residuals from floating means.
    within[np.all(np.ptp(y,axis=1)==0,axis=1)]=0.
    B=k*np.trace(gram)/(n-1);W=within.sum()/(n*(k-1));den=B+(k-1)*W
    varied=np.any(np.ptp(y,axis=(0,1))>0)
    rho=(B-W)/den if varied and den>0 else np.nan
    # Trace-estimator support guard: distinguish constant outputs from1.
    packed=np.ascontiguousarray(means).view(np.dtype((np.void,means.dtype.itemsize*f))).ravel()
    _,classes=np.unique(packed,return_inverse=True)
    indicator=np.eye(classes.max()+1)[classes]
    return dict(point=float(rho),between=float(B),within=float(W),mean_gram=gram,
                within_ss=within,mean_class_indicator=indicator,n_subjects=n,n_sessions=k,n_features=f)


def trace_bootstrap(y,counts):
    s=trace_statistics(y);n=s['n_subjects'];k=s['n_sessions'];c=np.asarray(counts,float)
    assert c.shape[1]==n and np.all(c.sum(1)==n)
    gram=s['mean_gram'];first=c@np.diag(gram)
    second=np.sum((c@gram)*c,axis=1)/n
    between=k*(first-second)/(n-1)
    numerical_tolerance=256*np.finfo(float).eps*np.maximum(first,np.finfo(float).tiny)
    assert np.all(between>=-numerical_tolerance)
    between=np.maximum(between,0)
    within=c@s['within_ss']/(n*(k-1));den=between+(k-1)*within
    nclasses=np.sum((c@s['mean_class_indicator'])>0,axis=1)
    eligible=((within>0)|(nclasses>1))&(den>0)&np.isfinite(den)
    rho=np.divide(between-within,den,out=np.full(len(c),np.nan),where=eligible)
    result=dict(s)
    result.update(bootstrap=rho,bootstrap_between=between,bootstrap_within=within,eligible=eligible)
    return result


def gallery_gap(S,D,counts):
    S=np.asarray(S,float);D=np.asarray(D,float);n=len(S)
    assert D.shape==(n,n)
    if not np.isfinite(S).all() or not np.isfinite(D[~np.eye(n,dtype=bool)]).all():
        return dict(point=np.nan,bootstrap=np.full(len(counts),np.nan),delete_one=np.full(n,np.nan),pseudovalues=np.full(n,np.nan))
    point=S.mean()-D[~np.eye(n,dtype=bool)].mean();loo=[]
    for i in range(n):
        keep=np.arange(n)!=i;d=D[np.ix_(keep,keep)]
        loo.append(S[keep].mean()-d[~np.eye(n-1,dtype=bool)].mean())
    loo=np.asarray(loo);pv=n*point-(n-1)*loo
    assert abs(pv.mean()-point)<1e-11
    return dict(point=float(point),bootstrap=np.asarray(counts,float)@pv/n,delete_one=loo,pseudovalues=pv)


def interval(x):
    x=np.asarray(x,float);v=x[np.isfinite(x)]
    lo,hi=np.percentile(v,[2.5,97.5]) if len(v)>=.95*len(x) else (np.nan,np.nan)
    return dict(ci_low=float(lo),ci_high=float(hi),valid_bootstrap=int(len(v)),n_bootstrap=int(len(x)),valid_fraction=float(len(v)/len(x)))


def rank(values):
    v=np.asarray(values,float);best=np.full(v.shape,np.nan);worst=best.copy();one=best.copy();two=best.copy()
    if not np.isfinite(v).all():return best,worst,one,two
    remaining=list(np.argsort(-v,kind='stable'));pos=1
    while remaining:
        block=[i for i in remaining if v[remaining[0]]-v[i]<=1e-12];end=pos+len(block)-1
        best[block]=pos;worst[block]=end;one[block]=1/len(block) if pos==1 else 0
        two[block]=min(max(3-pos,0),len(block))/len(block)
        remaining=[i for i in remaining if i not in block];pos=end+1
    return best,worst,one,two


def rank_frequency(points,bootstrap):
    m=np.asarray(bootstrap,float);valid=np.isfinite(m).all(1);n=len(m);families={}
    for family,size in [('six',6),('eight',8),('ten',10)]:
        b=m[:,:size];ok=np.isfinite(b).all(1);rb=np.full_like(b,np.nan);rw=rb.copy();one=rb.copy();two=rb.copy()
        for i in np.flatnonzero(ok):rb[i],rw[i],one[i],two[i]=rank(b[i])
        pb,pw,_,_=rank(np.asarray(points)[:size]);enough=ok.sum()>=.95*n
        frequency={k:np.full(size,np.nan) for k in ['rank1_frequency','top2_frequency','rank1_possible','rank1_certain','top2_possible','top2_certain']}
        if enough:
            frequency=dict(rank1_frequency=one[ok].mean(0),top2_frequency=two[ok].mean(0),rank1_possible=(rb[ok]<=1).mean(0),
                rank1_certain=(rw[ok]<=1).mean(0),top2_possible=(rb[ok]<=2).mean(0),top2_certain=(rw[ok]<=2).mean(0))
        families[family]=dict(point_best=pb,point_worst=pw,valid=ok,bootstrap_best=rb,bootstrap_worst=rw,**frequency)
    return families
