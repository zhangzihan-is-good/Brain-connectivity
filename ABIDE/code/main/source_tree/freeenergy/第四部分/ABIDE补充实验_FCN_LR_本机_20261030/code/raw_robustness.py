"""Local-only true-ROI-series robustness inputs for the AAL W1 pilot.

No classifier fitting or remote operations. Existing scientific functions are
read and hash-bound, not rewritten. build is an explicit separate CLI action;
import, select and validate-one never launch the full 200-subject construction.
"""
from __future__ import annotations

# Preserve caller-specific encodings/block sizes; serializers stay local.
import sys as _io_sys
from pathlib import Path as _IOPath
_IO_MERGE_ROOT = next(p for p in _IOPath(__file__).resolve().parents if (p / 'io_shared.py').is_file())
if str(_IO_MERGE_ROOT) not in _io_sys.path:
    _io_sys.path.insert(0, str(_IO_MERGE_ROOT))
import io_shared as _io_shared

# Shared RGIC execution: IC commit/profile provenance is verified at entry.
import sys as _rgic_sys
from pathlib import Path as _RGICPath
_RGIC_MERGE_ROOT = next(p for p in _RGICPath(__file__).resolve().parents if (p / 'rgic_shared.py').is_file())
if str(_RGIC_MERGE_ROOT) not in _rgic_sys.path:
    _rgic_sys.path.insert(0, str(_RGIC_MERGE_ROOT))
import rgic_shared as _rgic_shared

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import types

sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
_dll_handles=[]
if os.name=='nt' and hasattr(os,'add_dll_directory'):
    for _path in (r'D:\miniconda\Library\bin',r'D:\miniconda\DLLs'):
        if Path(_path).is_dir():_dll_handles.append(os.add_dll_directory(_path))
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[1]
CACHE=ROOT/'raw_cache'
RAW=WORKSPACE/'第四部分/inputs/raw_timeseries'
COHORT=WORKSPACE/'第四部分/ABIDE后续分类器队列_20261027/inputs/original_cohort_and_folds.json'
PHENOTYPES=WORKSPACE/'dataset/abide_data/01_subject_metadata/Phenotypic_V1_0b_preprocessed1.csv'
BASE_H5=WORKSPACE/'dataset/abide_data/04_rgc_intermediate_data/rgc_base.hdf5'
SOURCES={
    'archived_report_rgc_ic':WORKSPACE/'第四部分/按论文报告复现_20261027/sources/original_archive/abide_report_code_cleaned/01_main_experiment/01_matrix_construction/rgc_ic.py',
    'native_estimators':WORKSPACE/'第三部分最终版/code/estimators.py',
    'ABIDE_preparation':WORKSPACE/'第四部分/code/prepare_raw.py',
    'original_report_reader':WORKSPACE/'第四部分/按论文报告复现_20261027/code/report_data.py',
    'original_matched_reader':WORKSPACE/'第四部分/论文报告原设置_全标签_等权_四人口学_20261027/code/report_data_matched.py',
}
SEED=20261030
CONDITIONS={
    'prefix_025':dict(fraction=.25,sigma=0.),
    'prefix_050':dict(fraction=.50,sigma=0.),
    'prefix_075':dict(fraction=.75,sigma=0.),
    'prefix_100':dict(fraction=1.,sigma=0.),
    'noise_010':dict(fraction=1.,sigma=.1),
    'noise_030':dict(fraction=1.,sigma=.3),
}
BASE_KEYS=('r0','W1_gain','W1_cost','W1_peak_lag','W1_peak_rho','W1_potential')
NATIVE_KEYS=('r0','W1_rstar','W1_lag','W1_tie','te_h1_d1','var_p1','psi','coherence','gcmi','phiid_d1')
PAIRS={'te_h1_d1','var_p1','phiid_d1'}
_MODULES=None


def need(condition,message):
    if not condition:raise ValueError(message)
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def identity(value):return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()
def sha(path):
    return _io_shared.sha256_file(path, block_size=8*1024*1024)
def read(path):
    return _io_shared.read_json(path, encoding="utf-8-sig")
def output(path):
    path=Path(path).resolve();path.relative_to(CACHE.resolve());return path
def atomic(path,value):
    path=output(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.%d.partial'%os.getpid())
    with tmp.open('x',encoding='utf-8') as f:f.write(canonical(value)+'\n')
    os.replace(tmp,path)
def source_hashes():return {name:dict(path=str(path),sha256=sha(path)) for name,path in SOURCES.items()}
def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def modules():
    global _MODULES
    if _MODULES is not None:return _MODULES
    # Exact AST extraction avoids executing the archived CLI or importing its
    # unrelated utils module. The original function bodies/line numbers remain.
    path=SOURCES['archived_report_rgc_ic'];tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    names={'SubjectRGCBase','standardize_rows','vector_pairs','corr_at_lag','compute_subject_rgc_base'}
    nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    need({node.name for node in nodes}==names,'Archived RGIC function set changed')
    module=types.ModuleType('_local_raw_archived_rgc');sys.modules[module.__name__]=module
    module.__dict__.update(np=np,dataclass=dataclass,DEFAULT_GAMMA=.2,DEFAULT_DELTA=0.,DEFAULT_C_MAX=.9,_rgic_shared=_rgic_shared)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    extracted=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
    exec(compile(extracted,str(path),'exec'),module.__dict__)
    preparation=_load('_local_raw_ABIDE_preparation',SOURCES['ABIDE_preparation'])
    estimators=preparation.estimator_module()
    original=_load('_local_raw_original_report',SOURCES['original_report_reader'])
    matched=_load('_local_raw_matched_report',SOURCES['original_matched_reader'])
    _MODULES=dict(rgc=module,preparation=preparation,estimators=estimators,original=original,matched=matched)
    return _MODULES


def select_ids():
    """Fixed proportional site x diagnosis Hamilton allocation, no score input."""
    import h5py
    cohort=read(COHORT);groups={};attributes={}
    with h5py.File(BASE_H5,'r') as handle:
        for sid in cohort['ids']:
            attrs=handle['patients'][sid].attrs
            value=dict(site=str(attrs['site']),y=int(attrs['y']),sex=float(attrs['sex']),age=float(attrs['age']))
            attributes[sid]=value;groups.setdefault((value['site'],value['y']),[]).append(sid)
    need(len(attributes)==1035 and len({key[0] for key in groups})==17,'Expected original 1035-person/17-site cohort')
    quotas={key:200*len(ids)//1035 for key,ids in groups.items()}
    ordered=sorted(groups,key=lambda key:(-(200*len(groups[key])%1035),key))
    for key in ordered[:200-sum(quotas.values())]:quotas[key]+=1
    rng=random.Random(SEED);chosen=[]
    for key in sorted(groups):chosen.extend(rng.sample(sorted(groups[key]),quotas[key]))
    chosen=sorted(chosen);need(len(chosen)==len(set(chosen))==200,'Bad stratified sample')
    value=dict(version='raw_AAL_W1_pilot200_v1',seed=SEED,n=200,atlas='aal',windows=[1],
        sampler='Hamilton proportional allocation over original merged-HDF5-site x ASD0/TC1; remainders descending then tuple tie; sorted-stratum random.Random(seed) sampling; sorted final identities',
        selection_ids=chosen,selection_ids_sha256=identity(chosen),
        strata=[dict(site=key[0],y=key[1],n_available=len(groups[key]),n_selected=quotas[key]) for key in sorted(groups)],
        attributes={sid:attributes[sid] for sid in chosen},original_cohort_path=str(COHORT),original_cohort_sha256=sha(COHORT),
        phenotype_path=str(PHENOTYPES),phenotype_sha256=sha(PHENOTYPES),source_sha256=source_hashes(),
        source_manifest_sha256=sha(RAW/'aal_source_manifest.json'),TR_manifest_sha256=sha(RAW/'header_tr_manifest.json'),
        no_classifier_scores_or_predictions_used=True)
    target=CACHE/'selection_ids.json'
    if target.exists():need(read(target)==value,'Existing prespecified sample differs; do not silently resample')
    else:atomic(target,value)
    return value


def selection():
    value=read(CACHE/'selection_ids.json')
    need(value['n']==200 and value['seed']==SEED and value['atlas']=='aal','Wrong fixed pilot sample')
    need(identity(value['selection_ids'])==value['selection_ids_sha256'],'Pilot identities changed')
    need(value['source_sha256']==source_hashes(),'Original scientific source changed after sample declaration')
    need(value['source_manifest_sha256']==sha(RAW/'aal_source_manifest.json') and value['TR_manifest_sha256']==sha(RAW/'header_tr_manifest.json'),'Raw provenance manifest changed')
    return value


def load_series(subject_id):
    """Sorted numeric ROI-ID order, matching archived report load function."""
    path=RAW/'aal'/(subject_id+'_rois_aal.1D')
    with path.open(encoding='ascii') as f:tokens=f.readline().split()
    order=np.argsort(np.asarray([int(token[1:]) for token in tokens]))
    values=np.loadtxt(path,dtype=np.float64)[:,order]
    need(values.ndim==2 and values.shape[1]==116 and len(values)>=12 and np.isfinite(values).all(),'Invalid original ROI series')
    record=next(row for row in read(RAW/'aal_source_manifest.json')['files'] if row['subject_id']==subject_id)
    need(sha(path)==record['sha256'],'Raw ROI source changed')
    return values,dict(path=str(path),sha256=record['sha256'],roi_ids_sorted=[tokens[int(i)] for i in order],tr_seconds=float(record['tr_seconds']))


def condition_series(values,condition,subject_id):
    need(condition in CONDITIONS,'Unknown prespecified raw condition')
    cfg=CONDITIONS[condition];n=max(12,int(np.floor(len(values)*cfg['fraction'])))
    x=np.array(values[:n],copy=True);noise_seed=None
    if cfg['sigma']:
        std=values.std(axis=0,ddof=0);std=np.where(np.ptp(values,axis=0)==0,0.,std)
        noise_seed=int(identity([SEED,'ROI_Gaussian_noise','aal',subject_id,condition])[:16],16)
        rng=np.random.Generator(np.random.PCG64(noise_seed))
        x+=cfg['sigma']*std[None,:]*rng.standard_normal(x.shape)
    return x,dict(condition=condition,original_n_timepoints=len(values),n_timepoints=len(x),
        prefix_fraction=cfg['fraction'],sigma_original_full_ROI_std=cfg['sigma'],noise_seed=noise_seed,
        noise_order='Add independent condition/subject/time/ROI Gaussian draws in original raw ROI units using full original-series population std (ddof0); originally constant ROI amplitude0. Then archived RGIC loader standardization once and compute_subject standardization again; native estimators receive perturbed raw values as in original 1035 native path.',
        temporal_refiltering=False,preprocessing='Original public CPAC/filt_global already applied before truncation/noise')


def compute_subject(subject_id,condition):
    started=time.perf_counter();m=modules();raw,source=load_series(subject_id);x,details=condition_series(raw,condition,subject_id)
    active=np.flatnonzero(np.ptp(x,axis=0)!=0);need(len(active)>=2,'Fewer than two nonconstant ROIs after condition')
    y=x[:,active];m['estimators']._validate(y)
    with threadpool_limits(limits=1):
        # Literal historical double standardization, not a guessed equivalent.
        standardized=m['rgc'].standardize_rows(y.T)
        report=m['rgc'].compute_subject_rgc_base(standardized,windows=(1,),gamma=.2,delta=0.,c_max=.9)
        out=m['estimators'].correlation_primitives(y)
        prediction,pred_diag=m['preparation'].prediction_required(y);out.update(prediction)
        spectral,spectral_diag=m['preparation'].spectral_required(y,source['tr_seconds']);out.update(spectral)
        out.update(m['estimators'].gcmi_primitives(y));phi=m['estimators'].phiid_primitives(y);out.update(phi)
    n_rois=116;e=n_rois*(n_rois-1)//2
    base_source=dict(r0=report.pearson,W1_gain=report.gain[1],W1_cost=report.cost[1],
                     W1_peak_lag=report.peak_lag[1],W1_peak_rho=report.peak_rho[1],W1_potential=report.potential[1])
    hi,lo=np.tril_indices(len(active),-1);base_index=active[hi]*(active[hi]-1)//2+active[lo]
    i,j=np.triu_indices(len(active),1);native_index=active[j]*(active[j]-1)//2+active[i]
    base={};native={}
    for key in BASE_KEYS:
        values=np.asarray(base_source[key]);need(np.isfinite(values).all(),'Nonfinite RGIC primitive')
        dest=np.zeros(e,dtype=np.int16 if key=='W1_peak_lag' else np.float32);dest[base_index]=values;base[key]=dest
    for key in NATIVE_KEYS:
        values=np.asarray(out[key]);need(np.isfinite(values).all(),'Nonfinite native primitive: '+key)
        dtype=np.int8 if key=='W1_lag' else np.bool_ if key=='W1_tie' else np.float32
        dest=np.zeros((e,2) if key in PAIRS else e,dtype=dtype);dest[native_index]=values;native[key]=dest
    # These are the exact raw gate inputs. Group eta is fitted later using only
    # that condition's training identities, never one per-subject eta fallback.
    details.update(subject_id=subject_id,n_rois=n_rois,n_edges=e,active_rois=active.tolist(),
        constant_roi_indices=np.setdiff1d(np.arange(n_rois),active).tolist(),tr_seconds=source['tr_seconds'],
        input_source=source,prediction_diagnostics=pred_diag,spectral_diagnostics=spectral_diag,
        phiid_failed_d1=int(np.sum(phi['phiid_failed_d1'])),
        edge_order='np.tril_indices(R,-1); native channels [low ROI -> high ROI, reverse] preserved without sign reversal',
        constant_policy='Compute on nonconstant ROIs then zero-scatter all fields for incident constant-ROI edges exactly as original 1035 native implementation',
        elapsed_seconds=time.perf_counter()-started)
    return base,native,details


def _checkpoint(subject_id,condition,receipt):
    folder=CACHE/condition/'subject_checkpoints';folder.mkdir(parents=True,exist_ok=True)
    npz=folder/(subject_id+'.npz');json_path=folder/(subject_id+'.json')
    recipe=dict(selection_ids_sha256=receipt['selection_ids_sha256'],condition=CONDITIONS[condition],
                condition_name=condition,seed=SEED,scientific_source_sha256=receipt['source_sha256'],
                raw_source_manifest_sha256=receipt['source_manifest_sha256'],code_sha256=sha(__file__))
    if npz.exists() and json_path.exists():
        note=read(json_path);need(note['recipe']==recipe and sha(npz)==note['npz_sha256'],'Existing raw checkpoint changed')
        with np.load(npz,allow_pickle=False) as h:base={k:h['b__'+k] for k in BASE_KEYS};native={k:h['n__'+k] for k in NATIVE_KEYS}
        return base,native,note['details']
    need(not npz.exists() and not json_path.exists(),'Incomplete subject publication requires explicit review, no overwrite')
    base,native,details=compute_subject(subject_id,condition)
    tmp=npz.with_name(npz.name+'.%d.partial'%os.getpid())
    with tmp.open('xb') as f:np.savez(f,**{'b__'+k:v for k,v in base.items()},**{'n__'+k:v for k,v in native.items()})
    os.replace(tmp,npz);atomic(json_path,dict(recipe=recipe,npz_sha256=sha(npz),details=details))
    return base,native,details


def build(conditions=tuple(CONDITIONS)):
    receipt=selection();ids=receipt['selection_ids'];need(set(conditions).issubset(CONDITIONS),'Unknown condition')
    claim=CACHE/'build.claim';claim.mkdir();atomic(claim/'owner.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat()))
    succeeded=False
    try:
        for condition in conditions:
            done=CACHE/condition/'manifest.json'
            if done.exists():
                reader=RawReader(CACHE,condition);reader.base_arrays('aal');reader.native_arrays('aal');continue
            handles={};descriptions={};details=[]
            for row,sid in enumerate(ids):
                base,native,note=_checkpoint(sid,condition,receipt);details.append(note)
                for family,arrays in (('report_base',base),('native_1035',native)):
                    for key,value in arrays.items():
                        handle_key=(family,key)
                        if handle_key not in handles:
                            folder=CACHE/condition/family/'aal';folder.mkdir(parents=True,exist_ok=True)
                            path=folder/(key+'.npy.partial');need(not path.exists(),'Partial aggregate requires explicit recovery')
                            handles[handle_key]=np.lib.format.open_memmap(path,mode='w+',dtype=value.dtype,shape=(len(ids),)+value.shape)
                        handles[handle_key][row]=value
                atomic(CACHE/'build_state.json',dict(status='building',condition=condition,completed_subjects=row+1,n_subjects=len(ids),last_subject=sid))
                print(canonical(dict(condition=condition,n_complete=row+1,n_subjects=len(ids),subject=sid,elapsed_seconds=note['elapsed_seconds'])),flush=True)
            for (family,key),handle in handles.items():
                handle.flush();descriptions[(family,key)]=dict(shape=list(handle.shape),dtype=handle.dtype.str)
                handle._mmap.close()  # Close our newly written mapping before Windows rename.
            handles.clear()
            for family,keys in (('report_base',BASE_KEYS),('native_1035',NATIVE_KEYS)):
                arrays={};folder=CACHE/condition/family/'aal'
                for key in keys:
                    old=folder/(key+'.npy.partial');new=folder/(key+'.npy');need(not new.exists(),'Existing aggregate array must not be replaced')
                    os.replace(old,new);arrays[key]=dict(descriptions[(family,key)],sha256=sha(new),bytes=new.stat().st_size)
                atomic(folder/'manifest.json',dict(status='complete',atlas='aal',window=1,n_subjects=len(ids),arrays=arrays,
                    condition=condition,condition_parameters=CONDITIONS[condition],selection_ids_sha256=receipt['selection_ids_sha256'],
                    scientific_source_sha256=receipt['source_sha256'],code_sha256=sha(__file__)))
            atomic(done,dict(status='complete',condition=condition,n_subjects=len(ids),selection_ids_sha256=receipt['selection_ids_sha256'],
                family_manifest_sha256={family:sha(CACHE/condition/family/'aal/manifest.json') for family in ('report_base','native_1035')},
                scientific_source_sha256=receipt['source_sha256'],code_sha256=sha(__file__),subject_details=details))
        atomic(CACHE/'build_state.json',dict(status='requested_conditions_complete',conditions=list(conditions),n_subjects=200));succeeded=True
    finally:
        # Failed/abrupt builders retain ownership evidence; never PID-reclaim.
        if succeeded:(claim/'owner.json').unlink();claim.rmdir()


class RawReader:
    """Read-only W1 API compatible with original ReportData.candidates.

    metadata splits are the original atlas roles restricted to these 200 people.
    A caller must use an explicit subset-capable new FeatureEngine: the old
    FeatureEngine.prepare correctly refuses a cohort other than 1035.
    """
    def __init__(self,cache_root=CACHE,condition='prefix_100'):
        self.cache=Path(cache_root).resolve();self.cache.relative_to(CACHE.resolve())
        self.condition=condition;need(condition in CONDITIONS,'Unknown raw condition')
        self.inputs=self.cache/condition;self.receipt=selection();self._base={};self._native={};self._gates={};self._meta=None
        self.done=read(self.inputs/'manifest.json');need(self.done['status']=='complete' and self.done['selection_ids_sha256']==self.receipt['selection_ids_sha256'],'Raw condition not complete')
    def metadata(self):
        if self._meta is not None:return self._meta
        import csv
        ids=self.receipt['selection_ids'];index={sid:i for i,sid in enumerate(ids)};old=read(COHORT);original_ids=old['ids']
        with PHENOTYPES.open(encoding='utf-8-sig',newline='') as f:pheno={row['FILE_ID']:row for row in csv.DictReader(f) if row['FILE_ID'] in index}
        matched=modules()['matched'];demos=[]
        for sid in ids:
            row=pheno[sid];demos.append([matched.numeric(row['SEX']),matched.numeric(row['AGE_AT_SCAN']),matched.HAND_MAP.get(row.get('HANDEDNESS_CATEGORY','').strip().lower(),np.nan),matched.numeric(row.get('FIQ'))])
        splits={}
        for atlas,folds in old['folds'].items():
            splits[atlas]={str(fold):{role:np.asarray([index[sid] for sid in subjects if sid in index],dtype=np.int64) for role,subjects in roles.items()} for fold,roles in folds.items()}
        self._meta=dict(ids=np.asarray(ids),y=np.asarray([self.receipt['attributes'][sid]['y'] for sid in ids],dtype=np.int64),
            sites=np.asarray([self.receipt['attributes'][sid]['site'] for sid in ids]),sex=np.asarray([self.receipt['attributes'][sid]['sex'] for sid in ids]),
            age=np.asarray([self.receipt['attributes'][sid]['age'] for sid in ids]),demographics_raw=np.asarray(demos,dtype=np.float32),
            splits=splits,n=len(ids),source_sha256={str(self.inputs/'manifest.json'):sha(self.inputs/'manifest.json')},
            original_subject_indices=np.asarray([original_ids.index(sid) for sid in ids]),
            demographics_columns=['SEX','AGE_AT_SCAN','HANDEDNESS_CATEGORY','FIQ'],raw_condition=self.condition)
        return self._meta
    def _arrays(self,atlas,family,cache):
        need(atlas=='aal','This prespecified pilot supports only AAL')
        if atlas in cache:return cache[atlas]
        path=self.inputs/family/atlas/'manifest.json';need(sha(path)==self.done['family_manifest_sha256'][family],'Family manifest changed')
        manifest=read(path);arrays={}
        for key,desc in manifest['arrays'].items():
            file=path.parent/(key+'.npy');need(sha(file)==desc['sha256'],'Raw cached array changed')
            value=np.load(file,mmap_mode='r',allow_pickle=False);need(list(value.shape)==desc['shape'],'Array shape changed');arrays[key]=value
        cache[atlas]=arrays;return arrays
    def base_arrays(self,atlas):return self._arrays(atlas,'report_base',self._base)
    def native_arrays(self,atlas):return self._arrays(atlas,'native_1035',self._native)
    def _gate(self,atlas,window,fold):
        need(int(window)==1,'Only W1 is predeclared')
        return modules()['original'].ReportData._gate(self,atlas,window,fold)
    def candidates(self,atlas,method,window,fold):
        need(int(window)==1,'Only W1 is predeclared')
        return modules()['original'].ReportData.candidates(self,atlas,method,window,str(fold))


def validate_one(subject_id=None):
    receipt=selection();sid=subject_id or receipt['selection_ids'][0];need(sid in receipt['selection_ids'],'Validation subject must be preselected')
    notes=[]
    for condition in CONDITIONS:
        base,native,details=_checkpoint(sid,condition,receipt)
        notes.append(details)
        need(all(v.shape[0]==6670 and np.isfinite(v).all() for v in list(base.values())+list(native.values())),'Unexpected primitive shape')
        if condition=='prefix_100':
            row=read(COHORT)['ids'].index(sid);checks={}
            old=WORKSPACE/'第四部分/按论文报告复现_20261027/inputs'
            for family,arrays in (('report_base',base),('native_1035',native)):
                for key,value in arrays.items():
                    reference=np.load(old/family/'aal'/(key+'.npy'),mmap_mode='r',allow_pickle=False)[row]
                    difference=float(np.max(np.abs(value.astype(np.float64)-reference.astype(np.float64))))
                    checks[family+'/'+key]=difference
                    # Constant-edge lag/cost placeholders in historical RGIC
                    # base differ harmlessly from the explicit new zero scatter.
                    if details['constant_roi_indices'] and key in ('W1_peak_lag','W1_cost'):continue
                    need(difference<=2e-6,'Full-series scientific bridge differs: '+family+'/'+key)
            details['existing_full_series_bridge_max_abs']=checks
    target=CACHE/'validation_one.json';need(not target.exists(),'Validation receipt already exists; inspect instead of overwriting')
    atomic(target,dict(status='one_subject_six_conditions_computed_no_classifier_fit',subject_id=sid,conditions=notes,
        scientific_source_sha256=receipt['source_sha256'],code_sha256=sha(__file__),one_subject_only=True,
        limitations=['One-subject numerical bridge/timing is not full 200-subject QC or a total-runtime guarantee.']))
    print(canonical(dict(subject_id=sid,seconds_by_condition={row['condition']:row['elapsed_seconds'] for row in notes},receipt=str(target))),flush=True)
    return notes


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('select','validate-one','build'))
    parser.add_argument('--subject');parser.add_argument('--conditions',nargs='+',choices=tuple(CONDITIONS),default=list(CONDITIONS))
    args=parser.parse_args()
    if args.action=='select':print(canonical(select_ids()),flush=True)
    elif args.action=='validate-one':validate_one(args.subject)
    else:build(args.conditions)
