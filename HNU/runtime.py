"""Fresh raw-input computation and output helpers for fixed-model reproduction."""
import os
import sys
sys.dont_write_bytecode=True
for _name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='1'
from pathlib import Path
_dll_path=Path(sys.prefix)/'Library/bin'
_dll_handle=os.add_dll_directory(str(_dll_path)) if os.name=='nt' and _dll_path.is_dir() else None
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.signal import butter,sosfiltfilt
from scipy.signal.windows import dpss
import estimators
_SHARED_ROOT=Path(__file__).resolve().parents[1]
if str(_SHARED_ROOT) not in sys.path:
    sys.path.insert(0,str(_SHARED_ROOT))
from io_shared import sha256_file,clean_json,write_json,write_csv
KEYS=['r0','psi','coherence','gcmi','var_p1','var_p2','var_p3','phiid_d1','phiid_d2']
KEYS += [f'te_h{h}_d{d}' for h in [1,2,3] for d in [1,2]]
KEYS += [w+'_'+k for w in ['W1','W2'] for k in ['rstar','lag','tie']]

def sha(path):
    return sha256_file(path)



def clean(value):
    return clean_json(value)



def write(path,value):
    return write_json(path,value)



def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,path)



def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    return write_csv(path,rows,fields=fields,clean=True)



def filter_butter(x):
    """Original float64 demean + Butterworth4 SOS default odd-pad27, no rezscore."""
    x=np.ascontiguousarray(x,dtype=np.float64)
    sos=butter(4,[.01,.1],btype='bandpass',fs=.5,output='sos')
    return np.ascontiguousarray(sosfiltfilt(sos,x-x.mean(axis=0),axis=0))



def matched_spectral(x):
    """Original DPSS operation order, fixed shared .01-.1Hz internal band."""
    T=len(x);tapers=dpss(T,NW=3.,Kmax=5,sym=False,norm=2)
    fft=np.fft.rfft((x-x.mean(axis=0))[None,:,:]*tapers[:,:,None],n=T,axis=1)
    freq=np.fft.rfftfreq(T,d=2.);keep=(freq>=.01)&(freq<=.1)
    f=fft[:,keep,:];power=np.abs(f)**2;ii,jj=np.triu_indices(109,1)
    psi=np.empty(5886);coherence=np.empty(5886)
    for lo in range(0,5886,256):
        sl=slice(lo,min(lo+256,5886));cross=f[:,:,ii[sl]]*np.conj(f[:,:,jj[sl]])
        k=np.ones(5,dtype=bool)
        cs=cross[k].mean(axis=0)
        pi=power[k][:,:,ii[sl]].mean(axis=0);pj=power[k][:,:,jj[sl]].mean(axis=0)
        coherency=cs/np.sqrt(pi*pj)
        psi[sl]=np.imag(np.sum(np.conj(coherency[:-1])*coherency[1:],axis=0))
        coherence[sl]=np.mean(np.abs(coherency)**2,axis=0)
    return dict(psi=psi,coherence=coherence)



def primitive_arrays(raw):
    if raw.shape!=(295,109) or raw.dtype!=np.float32 or not np.isfinite(raw).all():
        raise ValueError('Expected finite historical 295x109 float32 ROI input')
    x=filter_butter(raw);out=estimators.compute_primitives(x,2.)
    out.update(matched_spectral(x))
    diagnostic={}
    for key,value in out.items():
        if any(word in key for word in ['condition','rank','residual','unstable','failed']):
            a=np.asarray(value);finite=a[np.isfinite(a)]
            diagnostic[key]=dict(shape=list(a.shape),nonfinite=int((~np.isfinite(a)).sum()),
                minimum=float(finite.min()) if finite.size else None,maximum=float(finite.max()) if finite.size else None,
                true_count=int(a.sum()) if a.dtype==bool else None)
    return {k:out[k] for k in KEYS},x,diagnostic



def load_primitives(data_root,config,workers=3):
    """Compute every recording directly from its input NPY; never read an NPZ."""
    data_root=Path(data_root).expanduser().resolve()
    rows=sorted(config['recordings'],key=lambda r:(r['subject'],r['order']))
    def one(row):
        path=data_root/row['filename']
        if sha(path)!=row['sha256']:
            raise ValueError('Input SHA mismatch: '+str(path))
        raw=np.load(path,allow_pickle=False)
        arrays,_,_=primitive_arrays(raw)
        print('Computed',row['subject'],row['session'],flush=True)
        return arrays
    with ThreadPoolExecutor(max_workers=workers) as pool:
        values=list(pool.map(one,rows))
    p={key:np.stack([value[key] for value in values]) for key in KEYS}
    for value in p.values():
        value.setflags(write=False)
    return p,rows
