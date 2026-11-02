"""Refit archived final ABIDE configurations with active shared-RGIC sources.

This is a fixed-configuration reproduction, not a new hyperparameter search.
All-label feature screening, original folds, train-only scaling, direction beta,
and native model code remain the archived scientific protocol.
"""
from __future__ import annotations
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
from _paths import BASE, CORE, MODELS, activate
activate()
from _execution import read, sha, validate_output, verify




def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def engine(data_workspace,output):
    # Source imports stay inside the migrated tree; only input locations bind
    # to the explicitly supplied external data workspace.
    for directory in (CORE,MODELS):
        sys.path.insert(0,str(directory))
    import joint_features as jf
    import linear_models as lm
    matched=jf._matched
    matched.WORKSPACE=data_workspace;matched.PART4=data_workspace/'第四部分'
    matched.OLD=matched.PART4/'按论文报告复现_20261027'
    matched.PHENOTYPES=data_workspace/'dataset/abide_data/01_subject_metadata/Phenotypic_V1_0b_preprocessed1.csv'
    matched.original.WORKSPACE=data_workspace;matched.original.PART4=matched.PART4
    raw=matched._ReadOnlyRawData.__new__(matched._ReadOnlyRawData)
    raw.root=matched.OLD;raw.inputs=raw.root/'inputs';raw.data=data_workspace/'dataset/abide_data'
    raw.base_h5=raw.data/'04_rgc_intermediate_data/rgc_base.hdf5'
    raw.pearson_h5=raw.data/'03_pearson_fc/abide.hdf5'
    raw.raw_root=matched.PART4/'inputs/raw_timeseries'
    raw._meta=raw._headers=None;raw._base={};raw._native={};raw._gates={}
    data=matched.PaperData.__new__(matched.PaperData)
    data.root=output;data.inputs=output/'inputs';data.inputs.mkdir(parents=True)
    data.reader=raw;data._metadata=None;data._demos={}
    obj=jf.FeatureEngine.__new__(jf.FeatureEngine)
    obj.root=output;obj.data=data;obj.reader=raw;obj._components=OrderedDict();obj._scores=OrderedDict()
    return obj,jf,lm


def execute(args):
    verification=verify()
    out,data=validate_output(args.output,args.data_workspace)
    selected=read(BASE/'configs/main_models.json')['rows']
    rows=[r for r in selected if r['classifier']==args.classifier and r['atlas']==args.atlas and r['window']==args.window and r['fold']==args.fold]
    if len(rows)!=1:raise ValueError('Exactly one accepted archived configuration is required')
    candidate=rows[0]['selected_candidate']
    # Runtime qualification occurs before creating output or loading data.
    for directory in (CORE,MODELS):
        sys.path.insert(0,str(directory))
    import linear_models  # Establish the original process-local DLL runtime first.
    import numpy as np
    import scipy, sklearn, h5py
    expected=dict(python='3.9.7',numpy='1.26.4',scipy='1.10.1',sklearn='1.4.2',h5py='3.6.0')
    actual=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,h5py=h5py.__version__)
    if actual!=expected:raise RuntimeError('Exact original classifier/data runtime required: '+str(actual))
    out.mkdir(parents=True,exist_ok=True)  # validate_output already required empty/new.
    (out/'models').mkdir()
    feature_engine,jf,lm=engine(data,out)
    methods=jf.METHODS if args.method=='all' else (args.method,)
    nm=None
    if args.classifier not in ('svm','logistic'):
        nm=load('_merged_abide_neural',MODELS/'neural_models.py');nm.ROOT=out
        if str(nm.torch.__version__)!='2.5.1+cu121':raise RuntimeError('Original neural torch2.5.1+cu121 runtime required')
    protocol=dict(status='fixed_accepted_configuration_refit',classifier=args.classifier,atlas=args.atlas,
        window=args.window,fold=args.fold,candidate=candidate,seed=42,
        fixed_configuration_sha256=sha(BASE/'configs/main_models.json'),
        data_workspace=str(data),active_verification=verification,all_labels_for_F_screen=True,
        independent_test_generalization=False,new_hyperparameter_search=False,runtime=actual)
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    results=[]
    with lm.threadpool_limits(limits=1):
        for method in methods:
            started=time.perf_counter()
            view=feature_engine.prepare(args.atlas,args.window,args.fold,candidate['features'],method=method,include_test=True)
            prep=lm.prepare_inputs(view,candidate['inputs'])
            if args.classifier in ('svm','logistic'):
                ev=lm.PreparedEvaluator(prep['a'],view['y_train'],prep['z'],seed=42,positive_label=0,direction_mask=prep['direction_mask'])
                fitted=ev.fit(candidate['model']);pred=lm.predict_model(fitted['model'],prep['x'])
                model=out/'models'/(method+'.pkl')
                with model.open('xb') as stream:pickle.dump(dict(model=fitted['model'],transform=prep['transform']),stream,protocol=4)
                prediction,score=pred['prediction'],pred['score']
            else:
                values={k:np.array(prep[k],copy=True) for k in ('a','z','x')}
                for value in values.values():value[:,prep['direction_mask']]*=float(candidate['model']['beta'])
                fm=view['feature_metadata'];nconn=view['X_train'].shape[1]
                spec=dict(n_connection_features=nconn,n_demographics=values['a'].shape[1]-nconn,
                          n_rois=dict(aal=116,cc200=200,ez=116)[args.atlas],
                          selected_edge_ids=fm['selected_edge_ids'],selected_channel_ids=fm['selected_channel_ids'])
                model=out/'models'/(method+'.pt')
                fitted=nm.fit(args.classifier,values['a'],view['y_train'],values['z'],view['y_valid'],checkpoint_path=model,
                    config=candidate['model']['config'],seed=42,transductive_X=values['x'] if args.classifier=='gcn' else None,
                    sites_train=view['sites_train'],sites_valid=view['sites_valid'],sites_test=view['sites_test'],input_spec=spec)
                pred=nm.predict(args.classifier,model,values['a'],values['z'],values['x'],sites_train=view['sites_train'],
                    sites_valid=view['sites_valid'],sites_test=view['sites_test'],config=candidate['model']['config'],input_spec=spec)
                prediction,score=pred['test_prediction_TC1'],pred['test_score_ASD']
            # ASD=0 internally; scores are explicitly oriented toward ASD.
            from sklearn.metrics import roc_auc_score,accuracy_score
            record=dict(classifier=args.classifier,atlas=args.atlas,window=args.window,fold=args.fold,method=method,
                ids_test=np.asarray(view['ids_test']).tolist(),y_TC1=np.asarray(view['y_test']).tolist(),
                prediction_TC1=np.asarray(prediction).tolist(),score_ASD=np.asarray(score).tolist(),
                AUROC=float(roc_auc_score(1-view['y_test'],score)),ACC=float(accuracy_score(view['y_test'],prediction)),
                elapsed_seconds=time.perf_counter()-started,model_sha256=sha(model),
                full_label_F_screening=True,independent_test_generalization=False)
            (out/(method+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            results.append({k:record[k] for k in ('method','AUROC','ACC')})
    print(json.dumps(dict(status='COMPLETE_FIXED_CONFIGURATION_REFIT',output=str(out),results=results),ensure_ascii=False,indent=2))
