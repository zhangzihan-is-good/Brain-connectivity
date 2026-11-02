"""Read-only verification of all fixed ABIDE cohort/fold identity contracts.

Reads HDF5 metadata and phenotype FD only. No feature arrays or models are
loaded, no train/test split is generated, and no output file is written.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
P4=BASE/'code/main/source_tree/freeenergy/第四部分'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def check(data):
    load('_identity_integrity',BASE/'tools/integrity.py').check()
    # Reuse the original process-local DLL initializer before h5py/NumPy.
    load('_identity_runtime',P4/'code/runtime.py')
    import h5py
    import numpy as np
    runner=load('_identity_selected_runner',BASE/'code/supplements/run_selected.py')
    controls=load('_identity_controls',P4/'ABIDE补充实验_FCN_LR_本机_20261030/code/control_features.py')
    def need(value,message):
        if not value:raise RuntimeError(message)
    with h5py.File(data/'dataset/abide_data/04_rgc_intermediate_data/rgc_base.hdf5','r') as base, h5py.File(data/'dataset/abide_data/03_pearson_fc/abide.hdf5','r') as old:
        ids=sorted(base['patients'])
        need(ids==sorted(old['patients']) and len(ids)==1035,'Original HDF5 cohort differs')
        index={sid:i for i,sid in enumerate(ids)}
        y=np.asarray([base['patients'][sid].attrs['y'] for sid in ids],dtype=np.int64)
        sites=np.asarray([str(base['patients'][sid].attrs['site']) for sid in ids])
        need(int((y==0).sum())==505 and int((y==1).sum())==530,'Original ASD0/TC1 labels differ')
        splits={}
        for atlas in ('aal','cc200','ez'):
            splits[atlas]={}; test_count=np.zeros(1035,dtype=np.int64)
            for fold in range(10):
                group=old['experiments'][atlas+'_whole'][str(fold)]
                roles={role:np.asarray([index[v.decode() if isinstance(v,bytes) else str(v)] for v in group[role][:]],dtype=np.int64) for role in ('train','valid','test')}
                controls.validate_roles(roles,1035,require_cover=True)
                need(all(len(np.unique(y[rows]))==2 for rows in roles.values()),'Original role has one diagnosis')
                test_count[roles['test']]+=1;splits[atlas][str(fold)]=roles
            need(np.all(test_count==1),'Original test roles do not cover participants once')
    for fold in range(10):
        need(all(np.array_equal(splits[atlas][str(fold)]['test'],splits['aal'][str(fold)]['test']) for atlas in splits),'Original atlas test order differs')
    metadata=dict(ids=np.asarray(ids),y=y,sites=sites,splits=splits)
    engine=SimpleNamespace(metadata=lambda:metadata,roles=lambda atlas,fold:splits[atlas][str(fold)])
    clean=lambda values:np.asarray(values).tolist()
    cfg,items=runner.plan()
    low_ids=runner.hydrate_identities(SimpleNamespace(controls=controls,clean=clean),engine,items,
        data/'dataset/abide_data/01_subject_metadata/Phenotypic_V1_0b_preprocessed1.csv')
    selected=runner.read(BASE/'configs/reproduction/selected_configurations.json')['rows']
    for row in selected:
        actual={role:len(rows) for role,rows in splits[row['atlas']][str(row['fold'])].items()}
        need(actual==row['roles'],'Fixed main configuration role counts differ')
    return dict(status='PASS_ALL_ORIGINAL_COHORT_AND_FOLD_IDENTITIES',full_participants=1035,
        lowFD_participants=len(low_ids),atlases=3,original_fold_role_sets=30,main_configuration_role_counts=len(selected),
        supplementary_identity_axes=len(cfg['expected_oof_axes']),ordered_array_SHA_matches=len(cfg['expected_oof_axes'])*3,
        supplementary_tasks=len(items),identity_arrays_retained_in_code=False,
        source='external HDF5 fixed roles and phenotype FD; no new split or fit',scientific_training_performed=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-workspace',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(check(args.data_workspace.resolve()),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
