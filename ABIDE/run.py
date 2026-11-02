"""Unified ABIDE entry: verify, plan-main, main, supplements or figures."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
if str(BASE.parent) not in sys.path:sys.path.insert(0,str(BASE.parent))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('verify','plan-main','main','supplements','figures'),default='verify')
    p.add_argument('--classifier',choices=('svm','logistic','gcn','fcn','lstm'))
    p.add_argument('--atlas',choices=('aal','cc200','ez'))
    p.add_argument('--window',type=int,choices=(1,2));p.add_argument('--fold',type=int,choices=range(10))
    p.add_argument('--method',default='all',choices=('all','RGIC','Pearson','GCMI','Lag_peak','Gaussian_TE','VAR_Granger','PSI','PhiID_unique_transfer','Coherence'))
    p.add_argument('--data-workspace');p.add_argument('--output')
    args,extra=p.parse_known_args(argv)
    from migration import verify,read
    check=verify()
    if args.mode=='verify':
        if extra:p.error('Unexpected extra arguments')
        print(json.dumps(check,ensure_ascii=False,indent=2));return
    if args.mode=='plan-main':
        rows=read(BASE/'configs/reproduction/selected_configurations.json')['rows']
        print(json.dumps(dict(status='PLAN_FIXED_CONFIGURATION_REFIT',accepted_units=len(rows),methods_per_unit=9,
            total_fold_method_fits=len(rows)*9,classifiers=sorted({r['classifier'] for r in rows}),
            data_payloads_included=False,execution_performed=False,all_label_F_screening=True,
            new_hyperparameter_search=False,shared_RGIC=True),ensure_ascii=False,indent=2));return
    if args.mode=='main':
        if extra or any(getattr(args,k) is None for k in ('classifier','atlas','window','fold','data_workspace','output')):
            p.error('main requires --classifier --atlas --window --fold --data-workspace --output')
        from main_reproduce import execute
        return execute(args)
    if args.mode=='supplements':
        if args.classifier:extra+=['--classifier',args.classifier]
        if args.fold is not None:extra+=['--fold',str(args.fold)]
        if args.data_workspace:extra+=['--data-workspace',args.data_workspace]
        if args.output:extra+=['--output',args.output]
        return module('_merged_abide_selected',BASE/'code/supplements/run_selected.py').main(extra)
    if args.output:extra+=['--output',args.output]
    # The unchanged figure CLI parses sys.argv; isolate its arguments locally.
    old=sys.argv
    try:
        sys.argv=[str(BASE/'code/figures/plot_selected_auroc.py')]+extra
        return module('_merged_abide_figures',BASE/'code/figures/plot_selected_auroc.py').main()
    finally:sys.argv=old


if __name__=='__main__':main()
