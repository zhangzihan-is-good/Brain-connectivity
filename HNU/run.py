"""Recompute fixed-model HNU article analyses from the 290 original ROI NPY files."""
import common as c
import numpy as np
import argparse
import json
import rgic
import reference_masks as old
import temporal_readouts as tr
import final_readouts as fr
import statistics_hnu as rs
from readouts import simultaneous
OUT=c.HERE/'results'
ENDPOINTS=['meanS','betweenD','gap','ID']

def primary(kind):return 'signed' if kind in ['pairwise','consensus'] else kind



def family(kind):return 'direction' if kind in ['pairwise','consensus','roles'] else 'strength'



def flatten_patterns(patterns,counts):
    names=[];points=[];boots=[];arrays={}
    for rep,pat in patterns.items():
        stats=tr.bootstrap_summary(pat,counts)
        for endpoint in ENDPOINTS:
            names.append(rep+'__'+endpoint);points.append(stats[endpoint]['point']);boots.append(stats[endpoint]['bootstrap'])
        for key in ['S','D','matrix','subject_ID','density_S','density_matrix','densities','counts','trace_y','density_traceICC']:
            if key in pat:arrays[rep+'__'+key]=pat[key]
        if 'trace_y' in pat:
            ys=pat['trace_y']
            trace=[rs.trace_bootstrap(y,counts) for y in ys]
            pts=np.array([r['point'] for r in trace]);bs=np.array([r['bootstrap'] for r in trace])
            names.append(rep+'__traceICC_auc');points.append(old.integrate(pts));boots.append(old.integrate(bs))
        elif rep in ['net_cosine','degree_cosine']:
            names.append(rep+'__traceICC_auc');points.append(np.nan);boots.append(np.full(len(counts),np.nan))
    return dict(names=np.array(names),point=np.array(points),bootstrap=np.stack(boots,axis=1),**arrays)



def combined(a,b):
    assert a['names']==b['names']
    return dict(names=a['names'],point=(a['point']+b['point'])/2,bootstrap=(a['bootstrap']+b['bootstrap'])/2,
                models=[dict(method=m['method'],config_id='two_transfer_models',parameters={}) for m in a['models']],
                job_paths=[x+' | '+y for x,y in zip(a['job_paths'],b['job_paths'])],n_subjects=29,subjects=[],raw=None)



def export(contexts):
    if any(key[0]!='shared' for key in contexts):
        raise ValueError('Only fixed shared models may be exported')
    summaries=[];paired=[];individual=[];density=[];lookup={}
    for key,v in contexts.items():
        strategy,scope,mode,kind=key
        for ei,name in enumerate(v['names']):
            rep,endpoint=name.split('__');points=v['point'][:,ei];draw=v['bootstrap'][:,:,ei]
            rb,rw,_,_=rs.rank(points);delta=points[0]-points[1:];db=draw[:,[0]]-draw[:,1:]
            lo,hi,ok=simultaneous(delta,db)
            for mi,m in enumerate(v['models']):
                row=dict(strategy=strategy,scope=scope,mode=mode,kind=kind,representation=rep,endpoint=endpoint,
                         main=(kind in c.KINDS and rep==primary(kind) and endpoint=='meanS'),method=m['method'],config_id=m['config_id'],
                         parameters_json=json.dumps(m['parameters'],sort_keys=True),estimate=points[mi],rank_best=rb[mi],rank_worst=rw[mi],
                         full_method_count=len(points),n_subjects=v['n_subjects'],**rs.interval(draw[:,mi]),job_path=v['job_paths'][mi])
                summaries.append(row)
                if mi:
                    d=dict(strategy=strategy,scope=scope,mode=mode,kind=kind,representation=rep,endpoint=endpoint,comparator=m['method'],
                           main=row['main'],estimate=delta[mi-1],**rs.interval(db[:,mi-1]),endpoint_family_size=len(delta),
                           endpoint_simultaneous_low=lo[mi-1],endpoint_simultaneous_high=hi[mi-1],complete_draws=ok)
                    paired.append(d);lookup[key+(name,m['method'])]=d
            if v['raw'] is not None:
                for mi,r in enumerate(v['raw']):
                    if endpoint=='meanS':
                        for pi,person in enumerate(v['subjects']):
                            individual.append(dict(strategy=strategy,scope=scope,mode=mode,kind=kind,representation=rep,method=v['models'][mi]['method'],subject=person,S=r[rep+'__S'][pi]))
                        if rep+'__density_S' in r:
                            for di,d in enumerate(r[rep+'__densities']):
                                density.append(dict(strategy=strategy,scope=scope,mode=mode,kind=kind,representation=rep,method=v['models'][mi]['method'],density=d,count=r[rep+'__counts'][di],meanS=r[rep+'__density_S'][di].mean()))
    for strategy in sorted({k[0] for k in contexts}):
        scopes=sorted({k[1] for k in contexts if k[0]==strategy})
        for scopeset,label in [([s],'main74') for s in scopes]+[(['A_selected_all29','B_selected_all29'],'both_all29_main148')]:
            if not all((strategy,s,mode,kind) in contexts for s in scopeset for mode in c.MODES for kind in c.KINDS):continue
            diffs=[];draws=[];keys=[]
            for s in scopeset:
                for mode in c.MODES:
                    for kind in c.KINDS:
                        v=contexts[strategy,s,mode,kind];name=primary(kind)+'__meanS';ei=v['names'].index(name)
                        diffs.extend(v['point'][0,ei]-v['point'][1:,ei]);draws.append(v['bootstrap'][:,[0],ei]-v['bootstrap'][:,1:,ei])
                        keys.extend([(strategy,s,mode,kind,name,m['method']) for m in v['models'][1:]])
            lo,hi,ok=simultaneous(np.array(diffs),np.concatenate(draws,axis=1))
            assert len(keys)==74*len(scopeset)
            for key,l,h in zip(keys,lo,hi):lookup[key].update({label+'_low':l,label+'_high':h,label+'_family_size':len(keys),label+'_complete_draws':ok})
    c.rep.csv_write(OUT/'summary.csv',summaries);c.rep.csv_write(OUT/'paired.csv',paired)
    c.rep.csv_write(OUT/'individual_S.csv',individual);c.rep.csv_write(OUT/'direction_density.csv',density)



def run_context(p,rows,scope,dev,target,mode,kind):
    strategy='shared'
    people=sorted(c.config()['halves'][target]) if target else sorted(c.config()['halves']['A']+c.config()['halves']['B'])
    n=len(people);ref=c.indices(rows,c.config()['halves'][dev],[1,2,3,4,5])
    ti=c.indices(rows,people,[1,2,3,4,5]);qi=c.indices(rows,people,[6,7,8,9,10])
    _,counts=rs.bootstrap_indices(n,c.B,c.SEED+n)
    pathcounts=OUT/'counts'/f'{scope}.npz'
    if not pathcounts.exists():c.rep.save(pathcounts,counts=counts,subjects=np.array(people))
    basekind={'polarity':'global','gain_fls':'highgain','roles':'consensus'}.get(kind,kind)
    model=c.model(dev,basekind)
    models=rgic.matched(model,family(kind)=='direction')
    results=[];paths=[]
    mask_by_window={w:old.masks(p,ref,w)[0] for w in ['W1','W2']}
    w=model['parameters']['window']
    mask=(mask_by_window[w]['laggain'][0] if kind=='highgain' else
          mask_by_window[w]['laggain_FLS'][0] if kind=='gain_fls' else None)
    if mask is not None:c.rep.save(OUT/'masks'/f'{strategy}_{scope}_{mode}_{kind}.npz',indices=mask,reference_indices=ref)
    for m in models:
        name=f"{strategy}_{scope}_{mode}_{kind}_{m['method']}"
        path=OUT/'jobs'/(name+'.npz')
        R=rgic.reference(p,ref,m['parameters']['window'],m['parameters']['delta']) if m['method']=='RGIC' else None
        ix=np.concatenate([ti,qi])
        values=(rgic.strength(p,ix,m,R) if family(kind)=='strength' else rgic.flux(p,ix,m,R=R))
        a=values[:n*5].reshape(n,5,5886);b=values[n*5:].reshape(n,5,5886)
        pat=fr.analyze(a,b,mode,kind,mask=mask,details=True,
                       r0_train=p['r0'][ti].reshape(n,5,5886),r0_query=p['r0'][qi].reshape(n,5,5886))
        if m['method']=='RGIC':
            c.rep.save(OUT/'values'/(name+'.npz'),train=a,query=b,reference_R=R)
        res=flatten_patterns(pat,counts)
        c.rep.save(path,**res)
        results.append(res);paths.append(str(path))
    value=dict(names=list(results[0]['names']),point=np.stack([r['point'] for r in results]),
               bootstrap=np.stack([r['bootstrap'] for r in results],axis=1),models=models,job_paths=paths,n_subjects=n,
               subjects=people,raw=results)
    print(strategy,scope,mode,kind,'done',flush=True)
    return value


def run(args):
    global OUT
    OUT=args.output.expanduser().resolve()
    if args.workers<1:
        raise ValueError('--workers must be positive')
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError('Choose a new/empty --output directory; existing results are never reused or overwritten')
    OUT.mkdir(parents=True,exist_ok=True)
    p,rows=c.rep.load_primitives(args.data_root,c.config(),args.workers)
    contexts={}
    for scope,dev,target in c.SCOPES:
        for kind in c.KINDS:
            for mode in c.MODES:
                contexts['shared',scope,mode,kind]=run_context(p,rows,scope,dev,target,mode,kind)
    for kind in c.KINDS:
        for mode in c.MODES:
            contexts['shared','equal_direction_mean',mode,kind]=combined(contexts['shared','A_to_B',mode,kind],contexts['shared','B_to_A',mode,kind])
    for scope,dev,target in c.SCOPES:
        for kind in ['polarity','gain_fls','roles']:
            for mode in c.MODES:
                contexts['shared',scope,mode,kind]=run_context(p,rows,scope,dev,target,mode,kind)
    for kind in ['polarity','gain_fls','roles']:
        for mode in c.MODES:
            contexts['shared','equal_direction_mean',mode,kind]=combined(contexts['shared','A_to_B',mode,kind],contexts['shared','B_to_A',mode,kind])
    export(contexts)
    print('Complete:',OUT,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=c.Path,default=c.DATA,help='Original 290 preprocessed ROI NPY files')
    ap.add_argument('--output',type=c.Path,default=c.HERE/'results',help='New/empty results directory')
    ap.add_argument('--workers',type=int,default=3)
    run(ap.parse_args())
