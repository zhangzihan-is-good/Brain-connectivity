"""Dataset-free demonstration of the pinned RGIC raw-recording API.

This synthetic example demonstrates reference fitting and target transformation;
it is not an HCP/HNU/ABIDE experiment or a replacement for the paper benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from rgic_backend import RGIC,RGICConfig,IC_COMMIT

def make_records(seed=20261103,n_reference=8,n_targets=3,n_timepoints=256):
    rng=np.random.default_rng(seed)
    records=[]
    for _ in range(n_reference+n_targets):
        driver=rng.standard_normal(n_timepoints+1)
        source=driver[1:]
        delayed=.85*driver[:-1]+.15*rng.standard_normal(n_timepoints)
        independent=rng.standard_normal(n_timepoints)
        negative=-.7*source+.3*rng.standard_normal(n_timepoints)
        records.append(np.column_stack((source,delayed,independent,negative)))
    return np.stack(records[:n_reference]),np.stack(records[n_reference:])

def compute(seed=20261103):
    reference,target=make_records(seed)
    names=('source','delayed','independent','negative_synchronous')
    model=RGIC(RGICConfig(lags=(-1,1))).fit(reference,roi_names=names)
    fitted=model.reliability_.copy()
    results=model.transform(target,roi_names=names)
    if not np.array_equal(fitted,model.reliability_):raise AssertionError('Targets changed fitted references')
    arrays=dict(reference=reference,target=target,pairs=results[0].pairs,
                reliability=fitted,ic=np.stack([r.ic for r in results]),
                eta=np.stack([r.eta for r in results]),lag_samples=np.stack([r.lag for r in results]))
    digest=hashlib.sha256()
    for key in sorted(arrays):
        value=np.ascontiguousarray(arrays[key]);digest.update(key.encode());digest.update(value.dtype.str.encode());digest.update(value.tobytes())
    summary=dict(example='synthetic_reference_target_demo',participant_data_used=False,seed=seed,
                 n_reference=len(reference),n_targets=len(target),timepoints=reference.shape[1],regions=4,edges=6,
                 first_pair_lag_samples=arrays['lag_samples'][:,0].tolist(),
                 reference_unchanged=True,finite_outputs=bool(all(np.isfinite(a).all() for a in arrays.values())),
                 ic_commit=IC_COMMIT,numpy=np.__version__,array_sha256=digest.hexdigest())
    return arrays,summary

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--output',type=Path,help='Optional new or empty output directory')
    args=parser.parse_args(argv)
    if args.output and args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('--output must be a new or empty directory')
    started=time.perf_counter();arrays,summary=compute(args.seed)
    summary['elapsed_seconds']=time.perf_counter()-started
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output/'synthetic_results.npz',**arrays)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary
if __name__=='__main__':main()
