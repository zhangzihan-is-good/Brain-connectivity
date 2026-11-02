"""Exact rank, quota and simultaneous-interval helpers used by personal templates."""
import numpy as np

def percentile(x):
    """Exact average-tie ranks, vectorized along the final axis."""
    x=np.asarray(x,np.float64)
    d=x.shape[-1]
    if d<2:raise ValueError('At least two coordinates required')
    order=np.argsort(x,axis=-1,kind='quicksort')
    sorted_x=np.take_along_axis(x,order,axis=-1)
    changes=sorted_x[...,1:]!=sorted_x[...,:-1]
    starts=np.concatenate([np.ones(x.shape[:-1]+(1,),bool),changes],axis=-1)
    ends=np.concatenate([changes,np.ones(x.shape[:-1]+(1,),bool)],axis=-1)
    positions=np.arange(d)
    lo=np.maximum.accumulate(np.where(starts,positions,0),axis=-1)
    hi=np.minimum.accumulate(np.where(ends,positions,d-1)[...,::-1],axis=-1)[...,::-1]
    # Match scipy's (average 1-based rank - 1)/(d-1), including ties.
    sorted_rank=((lo+hi)*.5)/(d-1)
    result=np.empty_like(x)
    np.put_along_axis(result,order,sorted_rank,axis=-1)
    return result

def membership(confidence, count):
    """Exactly count expected positive entries, uniformly splitting cutoff ties.

    An insufficient row is represented by all-NaN weights, never by selecting
    zero-confidence entries or silently reducing the requested quota.
    """
    x = np.asarray(confidence, np.float64)
    m = int(count)
    if x.ndim < 2 or not 0 < m <= x.shape[-1] or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Finite nonnegative confidence and valid positive quota required')
    valid = (x > 0).sum(-1) >= m
    cutoff = np.partition(x, x.shape[-1] - m, axis=-1)[..., x.shape[-1] - m]
    above = x > cutoff[..., None]
    tied = x == cutoff[..., None]
    fraction = (m - above.sum(-1)) / tied.sum(-1)
    out = above.astype(float) + tied * fraction[..., None]
    out[~valid] = np.nan
    return out, valid

def simultaneous(points,draws):
    """One complete prespecified max-standardized family; NA never omitted."""
    point=np.asarray(points,float);draw=np.asarray(draws,float)
    ok=np.isfinite(draw).all(1);lo=np.full(point.shape,np.nan);hi=lo.copy()
    if np.isfinite(point).all() and ok.sum()>=.95*len(draw):
        sd=np.std(draw[ok],axis=0,ddof=1);active=sd>0
        critical=np.quantile(np.max(np.abs((draw[ok][:,active]-point[active])/sd[active]),axis=1),.95) if active.any() else 0.
        lo=point-critical*sd;hi=point+critical*sd
    return lo,hi,int(ok.sum())
