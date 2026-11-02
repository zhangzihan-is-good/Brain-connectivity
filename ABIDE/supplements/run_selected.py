"""Refit 572 fixed supplementary groups; default is a standard-library plan.

External input recipes and the exact original runtime remain required. Ordered
participant/label/site identities are reconstructed from those external inputs
and checked against compact SHA256 contracts before any scientific computation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path: sys.path.insert(0,str(BASE))
from _paths import CORE, MODELS, SUPPLEMENTS, activate
activate()
CONFIG = BASE / 'configs/supplement_tasks.json'
SUPP = Path('第四部分/ABIDE补充实验_FCN_LR_本机_20261030')
V3 = SUPP / 'article_extension_v3_20261031'
FAMILIES = ('direction', 'components', 'population', 'raw_robustness')
CLASSES = ('fcn', 'logistic')
FACTORS = ('no_cost', 'no_reliability', 'no_compensation')
COUNTS = dict(direction=160, components=300, population=760, raw_robustness=4500)


def need(ok, text):
    if not ok:
        raise RuntimeError(text)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _write(root, relative, value):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _allow(task):
    family, condition, method = task['family'], task['condition'], task['method']
    if family == 'direction':
        return ((condition == 'signed_joint' and method in ('RGIC', 'Pearson', 'GCMI', 'PSI'))
                or (condition in ('IC_only', 'D_only', 'abs_fixed', 'abs_rescreen') and method == 'RGIC'))
    if family == 'components':
        if condition == 'full':
            return method == 'RGIC'
        mode = next((m for m in ('fixed', 'rescreen') if condition.endswith('_' + m)), None)
        removed = condition[:-(len(mode) + 1)].split('__') if mode else []
        return (method == 'RGIC' and bool(removed) and len(set(removed)) == len(removed)
                and all(x in FACTORS for x in removed) and removed == [x for x in FACTORS if x in removed])
    methods = ('RGIC', 'Pearson', 'GCMI', 'Lag_peak', 'Gaussian_TE', 'VAR_Granger',
               'PSI', 'PhiID_unique_transfer', 'Coherence')
    if family == 'population':
        return ((condition in (p + '_' + d for p in ('all', 'lowFD')
                 for d in ('brain_only', 'brain_plus_demo')) and method in methods)
                or (condition in ('all_demo_only', 'lowFD_demo_only') and method == 'demographics'))
    if family == 'raw_robustness':
        inputs = ['prefix_' + f for f in ('025', '050', '075')]
        inputs += [f'noise_{s}_seed{r}' for s in ('010', '020', '030') for r in (101, 202, 303)]
        return method in methods and (condition == 'prefix_100'
            or condition in (x + '__' + m for x in inputs for m in ('fixed', 'rescreen')))
    return False


def plan():
    cfg = read(CONFIG)
    need(cfg['expected_records'] == 5720 and cfg['expected_groups'] == 572, 'Wrong selected scope')
    axes = cfg['expected_oof_axes']
    need(set(axes) == {f'{p}/{f}' for p in ('full','lowFD') for f in range(10)}, 'Wrong OOF axis roster')
    for population, n in (('full',1035),('lowFD',884)):
        need(cfg['cohorts'][population]['n'] == n, 'Wrong fixed cohort size')
        need(sum(axes[f'{population}/{f}']['n'] for f in range(10)) == n, 'OOF participant count differs')
    for axis in axes.values():
        need(set(axis['canonical_sha256']) == {'ids_test','y_test','sites_test'}, 'Incomplete OOF identity contract')
        need(all(len(h)==64 and set(h)<=set('0123456789abcdef') for h in axis['canonical_sha256'].values()), 'Invalid OOF SHA256')
    items, groups = [], {}
    for row in cfg['tasks']:
        task = row['task']
        need(_allow(task) and task['classifier'] in CLASSES and task['atlas'] == 'aal'
             and task['window'] == 1 and task['seed'] == 42 and task['fold'] in range(10), 'Task outside selected allowlist')
        candidate = cfg['candidate_registry'][row['candidate_ref']]
        need(identity(candidate) == row['candidate_ref'], 'Changed embedded frozen candidate')
        need(identity(dict(task=task, candidate=candidate))[:24] == row['task_id'], 'Task scientific identity changed')
        need(row['expected_oof_ref'] in axes, 'Unknown fixed OOF identity axis')
        item = dict(task=task, task_id=row['task_id'], candidate=candidate,
                    parameter_source=dict(source='embedded_fixed_candidate',sha256=row['candidate_ref']),
                    expected_oof_ref=row['expected_oof_ref'], execution='fit')
        if 'raw_condition' in row:
            item.update(raw_condition=row['raw_condition'], selection_mode=row['selection_mode'])
        items.append(item)
        key = tuple(task[k] for k in ('family', 'condition', 'classifier', 'method'))
        groups.setdefault(key, []).append(task['fold'])
    need(len(items) == len({x['task_id'] for x in items}) == 5720, 'Missing/duplicate selected tasks')
    need(len(groups) == 572 and all(sorted(x) == list(range(10)) for x in groups.values()), 'Incomplete group/fold roster')
    need(dict(Counter(x['task']['family'] for x in items)) == COUNTS, 'Family counts changed')
    return cfg, items


def verify_sources(strict=True):
    checker = _load('_selected_active_integrity', BASE/'tools/integrity.py')
    checker.check()
    return checker.bindings(), []


def hydrate_identities(c, original, items, phenotype_csv):
    """Restore only the original ordered roles, then require every frozen SHA.

    No split, randomization, label reassignment, FD imputation, or rescreening
    occurs here. Numerical feature selection remains in the scientific engine.
    """
    cfg, meta = read(CONFIG), original.metadata()
    fd, _ = c.controls.load_mean_fd(phenotype_csv, meta['ids'])
    eligible, _ = c.controls.low_motion_mask(fd)
    axes = {}
    cohorts = dict(full=sorted(c.clean(meta['ids'])), lowFD=sorted(c.clean(meta['ids'][eligible])))
    for population, ids in cohorts.items():
        spec = cfg['cohorts'][population]
        need(len(ids)==len(set(ids))==spec['n'] and identity(ids)==spec['sorted_ids_sha256'], 'Fixed cohort identity differs: '+population)
    need(set(cohorts['lowFD']) <= set(cohorts['full']), 'LowFD is not the original subset')
    for fold in range(10):
        full_roles = original.roles('aal', fold)
        low_roles, _ = c.controls.filter_roles(full_roles, eligible, labels=meta['y'])
        for population, roles in (('full',full_roles),('lowFD',low_roles)):
            key = f'{population}/{fold}'
            axis = {name+'_test':c.clean(meta[field][roles['test']]) for name,field in
                    (('ids','ids'),('y','y'),('sites','sites'))}
            expected = cfg['expected_oof_axes'][key]
            need(len(axis['ids_test'])==expected['n'], 'Fixed OOF count differs: '+key)
            for name, value in axis.items():
                need(identity(value)==expected['canonical_sha256'][name], 'Ordered OOF identity differs: '+key+'/'+name)
            axes[key] = axis
    for item in items:
        item.update({'expected_'+key:value for key,value in axes[item['expected_oof_ref']].items()})
    return set(cohorts['lowFD'])


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _data_bindings(data):
    """Verify exact historical input bytes; never repair/rewrite their recipes."""
    result = {}
    for ref in read(CONFIG)['original_input_bindings']:
        target = (data / ref['relative_workspace']).resolve()
        target.relative_to(data)
        need(target.is_file() and sha(target) == ref['sha256'], 'Frozen input changed/missing: ' + str(target))
        result[str(target)] = ref['sha256']
    return result


def _output(args):
    need(args.output and args.data_workspace, '--execute requires --output and --data-workspace')
    out, data = Path(args.output).resolve(), Path(args.data_workspace).resolve()
    protected = [BASE.resolve(), SUPPLEMENTS.resolve(), data / 'dataset', data / '第四部分', data / '第三部分最终版']
    need(out != data and out not in data.parents, 'Output cannot replace the data workspace')
    need(all(out != p and p not in out.parents and out not in p.parents for p in protected),
         'Output must be a new reproduction directory outside archived code and original scientific/input roots')
    marker = dict(config_sha256=sha(CONFIG), runner_sha256=sha(__file__), data_workspace=str(data),
                  source='selected-only packaging adapter; not a historical admission', records=5720, groups=572)
    mp = out / 'selected_reproduction.json'
    if out.exists() and any(out.iterdir()):
        need(mp.is_file() and read(mp) == marker, 'Nonempty output belongs to another run; no overwrite')
    else:
        out.mkdir(parents=True, exist_ok=True)
        _write(out, mp.name, marker)
    return out, data


def _runtime(out, data):
    need(os.name == 'nt' and not any(os.environ.get(k) for k in
         ('SLURM_JOB_ID', 'SLURM_JOBID', 'SLURM_CLUSTER_NAME')), 'Local Windows execution only')
    activate()
    c = _load('_selected_article_science', SUPPLEMENTS / 'article_core.py')
    actual = dict(python=sys.version.split()[0], numpy=c.np.__version__,
                  scipy=c.jf._matched.original.scipy.__version__, sklearn=c.lm.sklearn.__version__,
                  h5py=c.jf._matched.original.h5py.__version__, torch=str(c.nm.torch.__version__))
    expected = dict(python='3.9.7', numpy='1.26.4', scipy='1.10.1', sklearn='1.4.2',
                    h5py='3.6.0', torch='2.5.1+cu121')
    need(actual == expected, 'Exact scientific runtime required: ' + str(actual))
    c.ROOT = c.legacy.ROOT = c.nm.ROOT = out
    matched = c.jf._matched
    matched.WORKSPACE, matched.PART4 = data, data / '第四部分'
    matched.OLD = matched.PART4 / '按论文报告复现_20261027'
    matched.PHENOTYPES = data / 'dataset/abide_data/01_subject_metadata/Phenotypic_V1_0b_preprocessed1.csv'
    # Legacy snapshot recipe keys remain original absolute paths and are checked
    # by the original read-only reader.  They are not rewritten or bypassed.
    matched.original.WORKSPACE, matched.original.PART4 = data, data / '第四部分'
    mechanism = _load('_selected_mechanism_science', SUPPLEMENTS / 'mechanism_features.py')
    raw = _load('_selected_raw_science', SUPPLEMENTS / 'raw_v3.py')
    raw.RAW = data / '第四部分/inputs/raw_timeseries'
    raw.COHORT = data / '第四部分/ABIDE后续分类器队列_20261027/inputs/original_cohort_and_folds.json'
    raw.PHENOTYPES, raw.BASE_H5 = matched.PHENOTYPES, data / 'dataset/abide_data/04_rgc_intermediate_data/rgc_base.hdf5'
    return c, mechanism, raw, actual


def _demo_view(c, engine, fold):
    """Original run_local.py91-106: four direct columns, no connection F/K."""
    roles, meta = engine.roles('aal', fold), engine.metadata()
    demos, _ = engine.data.demographics('aal', fold)
    view = dict(direction_mask=c.np.zeros(4, dtype=bool))
    for role in c.ROLES:
        view['X_' + role] = demos[role]
        for key, array in (('y', meta['y']), ('ids', meta['ids']), ('sites', meta['sites'])):
            view[key + '_' + role] = array[roles[role]]
        view['y_' + role + '_ASD1'] = 1 - meta['y'][roles[role]]
        view['demos_' + role] = view['demo_' + role] = demos[role]
    view['feature_metadata'] = dict(K=0, n_direction_selected=0, n_scalar_selected=0,
        selected_edge_ids=[], selected_channel_ids=[], direct_covariate_columns=['SEX','AGE','HAND','FIQ'],
        recipe=dict(F_fit_scope='none', predictors='four_demographics_only_no_connection_features'))
    return view


def _fit_demo(c, item, view, protocol_sha):
    prior = c.completed(item, protocol_sha)
    if prior is not None:
        return prior
    task, fm = item['task'], view['feature_metadata']
    need(task['family'] == 'population' and task['method'] == 'demographics'
         and task['condition'] in ('all_demo_only', 'lowFD_demo_only'), 'Strict demo-only task required')
    need(fm['K'] == 0 and fm['selected_edge_ids'] == fm['selected_channel_ids'] == []
         and fm['recipe']['F_fit_scope'] == 'none' and not view['direction_mask'].any(), 'Demo-only connection invariants differ')
    need(item['candidate']['inputs']['demographics'] == 'none', 'Four direct predictors must not be appended twice')
    need(all(view['X_'+r].shape[1] == 4 for r in c.ROLES)
         and sum(len(view['ids_'+r]) for r in c.ROLES) == task['expected_test_n'], 'Demo-only active roles changed')
    for field in ('ids_test', 'y_test', 'sites_test'):
        need(c.clean(view[field]) == item['expected_' + field], 'Demo-only OOF identity differs')
    row = dict(task, task_id=item['task_id'], protocol_sha=protocol_sha, candidate=item['candidate'],
        parameter_source=item['parameter_source'], feature_metadata=fm, status='running',
        full_test_labels_used_in_F=False, label_semantics='ASD0_TC1', score_semantics='higher_ASD',
        no_new_parameter_search=True, parameter_policy='frozen_original_validation_selected_configuration_counterfactual_refit',
        record_origin='recomputed_selected_demographics_only',
        demographic_predictors_not_used_in_RGIC_estimator=True,
        article_task_manifest_sha256=c.sha(c.ROOT/'inputs/task_manifest.json'))
    row.update({f:view[f] for f in ('ids_test','y_test','sites_test')})
    try:
        row.update(c.fit_view(view, task['classifier'], item['candidate'], item['task_id'], task['seed']))
        row.update(status='complete', finished_local=c.now())
        c._commit(item, row, protocol_sha)
    except BaseException as error:
        row.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        c.write(c.record_path(item), row)
        raise
    return row


def _raw_setup(raw, args, out, data, bindings, existing_cache=None):
    if args.rebuild_raw:
        raw.ROOT, raw.CACHE, raw.PLAN = out, out/'raw_cache', out/'raw_cache/raw_plan.json'
        declared = raw.declare()
        for ref in declared['source_sha256'].values():
            bindings[str(Path(ref['path']).resolve())] = ref['sha256']
        bindings[str(raw.PLAN.resolve())] = raw.sha(raw.PLAN)
        bindings[str(Path(raw.__file__).resolve())] = raw.sha(raw.__file__)
    else:
        cache = existing_cache or (Path(args.raw_cache).resolve() if args.raw_cache else data/V3/'raw_cache')
        raw.ROOT, raw.CACHE, raw.PLAN = cache.parent, cache, cache/'raw_plan.json'
        raw.plan()  # Original absolute path, source-byte and cohort guards remain.
        bindings[str(raw.PLAN.resolve())] = raw.sha(raw.PLAN)
    return bindings


def _summarize(c, items, protocol_sha):
    st = _load('_selected_statistics', Path(__file__).with_name('_statistics.py'))
    records = []
    for item in items:
        record = c.completed(item, protocol_sha)
        need(record is not None, 'All5720 tasks must complete before selected summary')
        records.append(st.normalize_record(record, c.record_path(item), c.ROOT/'records'))
    groups = st.aggregate_records(records)
    need(len(groups) == 572 and all(g['status'] == 'complete' for g in groups), 'Selected group coverage incomplete')
    lookup = {(g['family'],g['condition'],g['classifier'],g['method']):g for g in groups}
    pairs, cache = [], {}
    for target in groups:
        f, cond, cls, method = target['family'],target['condition'],target['classifier'],target['method']
        reference = lookup[(f,'full',cls,'RGIC')] if f == 'components' else lookup[(f,'prefix_100',cls,method)] if f == 'raw_robustness' else None
        if reference is None or reference is target:
            continue
        value = st.paired_comparison(reference, target, 500, 20261030, cache)
        # The selected plotting schema is target-reference. Flip both saved-law
        # endpoints, not just the point; shared helpers may calculate other
        # metrics internally, without an extra graph or inference entry.
        bounds = value['conditional_CI95_AUROC']
        need(bounds is not None, 'AUROC interval unavailable; preserve records without inventing a bound')
        pairs.append(dict(reference=[reference[k] for k in ('family','condition','classifier','method')],
            target=[target[k] for k in ('family','condition','classifier','method')],
            delta_AUROC=-value['mean_fold_delta_AUROC'], conditional_CI95_AUROC=[-bounds[1],-bounds[0]],
            orientation='target_minus_reference', bootstrap_requested=500, bootstrap_seed=20261030,
            bootstrap_valid_AUROC=value['bootstrap_valid_AUROC'], bootstrap_undefined_AUROC=value['bootstrap_undefined_AUROC']))
    need(len(pairs) == 460, 'Selected AUROC comparison roster differs')
    groups = [dict(**{k:g[k] for k in ('family','condition','classifier','method','atlas','window','seed',
                   'status','complete_folds','expected_folds','n_test','expected_test_n')},
                   mean_AUROC=g['mean_fold_AUROC'], sd_AUROC=g['sd_fold_AUROC'],
                   n_defined_folds_AUROC=g['n_defined_folds_AUROC']) for g in groups]
    c.write(c.ROOT/'reports/selected_AUROC_results.json', dict(status='complete_selected_AUROC_only',
        source='newrun_recomputed_selected_tasks', selected_fold_records=5720,
        expected_groups=572, groups=groups, comparisons=pairs, protocol_sha256=protocol_sha,
        full_label_joint_screening=True, selected_display_scope_not_entire_historical_study=True,
        conditional_bootstrap=dict(count=500,seed=20261030), limitation=st.LIMITATION))


def execute(args, items, bindings):
    out, data = _output(args)
    bindings = dict(bindings, **_data_bindings(data))
    c, mechanism, raw, actual = _runtime(out, data)
    original = c.original_engine()
    low_ids = hydrate_identities(c, original, items, c.jf._matched.PHENOTYPES)
    with c.claim('selected_supplements'):
        c.write(out/'inputs/task_manifest.json', dict(status='selected_frozen_tasks', tasks=items, expected_records=5720, expected_groups=572))
        pp = out/'protocol_selected.json'
        previous = read(pp) if pp.exists() else None
        cache = Path(previous['raw_cache']).resolve() if previous else None
        if previous:
            need(not args.raw_cache or Path(args.raw_cache).resolve() == cache,
                 'Cannot switch the bound raw cache inside one reproduction output')
            need(not args.rebuild_raw or cache == (out/'raw_cache').resolve(),
                 'Cannot rebuild a historical/read-only raw cache')
        # The whole selected manifest includes raw inputs. Even a filtered run
        # binds the same cache up front, so later summary/subset resumes do not
        # change its execution protocol or reinterpret earlier model records.
        bindings = _raw_setup(raw, args, out, data, dict(bindings), cache)
        protocol = dict(status='frozen_before_new_test_scoring', version='selected-only-reproduction-adapter',
            execution_sha256=bindings, config_sha256=sha(CONFIG), runtime=actual,
            expected_records=5720, expected_groups=572, full_label_F=True,
            original_data_workspace=str(data), raw_cache=str(raw.CACHE))
        if pp.exists():
            need(all(previous[k] == v for k,v in protocol.items() if k != 'execution_sha256'),
                 'Existing selected execution protocol differs; no silent reset')
            need(all(previous['execution_sha256'].get(p) == h for p,h in bindings.items()),
                 'Bound source/input hash changed since the selected protocol was created')
        else:
            c.write(pp, protocol)
        protocol_sha = sha(pp)
        if args.action == 'summarize':
            return _summarize(c, items, protocol_sha)
        if args.rebuild_raw:
            raw.build(pp)
        if args.action == 'prepare-raw':
            return
        entries = [x for x in items if (args.family is None or x['task']['family'] == args.family)
                   and (args.classifier is None or x['task']['classifier'] == args.classifier)
                   and (args.fold is None or x['task']['fold'] == args.fold)]
        # prefix100 is always committed before transformed fixed-column tasks.
        entries.sort(key=lambda x:(FAMILIES.index(x['task']['family']),
            0 if x['task']['condition']=='prefix_100' else 1,x.get('raw_condition',''),
            x['task']['classifier'],x['task']['fold'],x['task']['condition'],x['task']['method']))
        active_raw_condition, reader = None, None
        with c.threadpool_limits(limits=1):
            for item in entries:
                if c.completed(item, protocol_sha) is not None:
                    continue
                t, candidate = item['task'], item['candidate']
                family, cond, fold = t['family'], t['condition'], t['fold']
                if family == 'raw_robustness':
                    if active_raw_condition != item['raw_condition']:
                        reader = raw.RawV3Reader(raw.CACHE,item['raw_condition'])
                        reader.base_arrays('aal'); reader.native_arrays('aal')
                        active_raw_condition = item['raw_condition']
                    engine = c.CohortEngine(original,'aal',fold,metadata=reader.metadata(),reader=reader)
                    if item['selection_mode']=='fixed':
                        ref = next(x for x in items if x['task']['family']==family and x['task']['condition']=='prefix_100'
                                   and all(x['task'][k]==t[k] for k in ('classifier','fold','method')))
                        saved = c.completed(ref, protocol_sha)
                        need(saved is not None, 'Full-input reference must be completed before fixed columns')
                        strength,direction,detail = engine.pool('aal',1,fold,candidate['features']['core'],t['method'])
                        view = engine.from_pool('aal',1,fold,candidate['features'],strength,direction,t['method'],detail,
                            selection_scope='all',include_test=True,fixed_columns=saved['feature_metadata']['selected_columns'])
                    else:
                        view = engine.prepare('aal',1,fold,candidate['features'],method=t['method'],include_test=True,selection_scope='all')
                else:
                    roles = None
                    if family=='population' and cond.startswith('lowFD_'):
                        eligible = c.np.isin(original.metadata()['ids'],list(low_ids))
                        roles,_ = c.controls.filter_roles(original.roles('aal',fold),eligible,labels=original.metadata()['y'])
                    engine = c.CohortEngine(original,'aal',fold,roles=roles)
                    if family=='direction' and cond!='signed_joint':
                        mf = mechanism.MechanismFeatures(engine,c.jf)
                        view = mf.prepare('aal',1,fold,candidate['features'],channels={'IC_only':'IC','D_only':'D'}.get(cond,'IC+D'),
                            direction='abs' if cond.startswith('abs_') else 'signed',
                            selection='fixed' if cond=='abs_fixed' else 'rescreen',include_test=True,selection_scope='all')
                    elif family=='components' and cond!='full':
                        mode = 'fixed' if cond.endswith('_fixed') else 'rescreen'
                        removed = tuple(cond[:-(len(mode)+1)].split('__'))
                        view = mechanism.MechanismFeatures(engine,c.jf).prepare('aal',1,fold,candidate['features'],
                            components=removed,selection=mode,include_test=True,selection_scope='all')
                    elif t['method']=='demographics':
                        view = _demo_view(c,engine,fold)
                    else:
                        view = engine.prepare('aal',1,fold,candidate['features'],method=t['method'],include_test=True,selection_scope='all')
                if t['method']=='demographics':
                    _fit_demo(c,item,view,protocol_sha)
                else:
                    c.fit_record(item,view,protocol_sha)
                del view, engine


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--plan',action='store_true',help='Default: standard-library plan/source checks only')
    choice.add_argument('--execute',action='store_true',help='Explicitly authorize selected scientific computation in a new output')
    parser.add_argument('--action',choices=('run','prepare-raw','summarize'),default='run')
    parser.add_argument('--output'); parser.add_argument('--data-workspace'); parser.add_argument('--raw-cache')
    parser.add_argument('--rebuild-raw',action='store_true',help='Recompute all13 raw inputs in the new output, never old caches')
    parser.add_argument('--family',choices=FAMILIES); parser.add_argument('--classifier',choices=CLASSES)
    parser.add_argument('--fold',type=int,choices=range(10))
    args = parser.parse_args(argv)
    _, items = plan()
    bindings, missing = verify_sources(strict=args.execute)
    if not args.execute:
        print(json.dumps(dict(status='read_only_plan',records=5720,groups=572,records_by_family=COUNTS,
            historical_reuse_recomputed=True,metric_for_figures='AUROC',scientific_imports=False,
            execution_performed=False,identity_arrays='reconstructed from original external roles and FD; hard SHA256 match required',
            portability='external manifests/recipes and exact Windows runtime remain strict'),ensure_ascii=False,indent=2))
        return
    need(args.action!='prepare-raw' or args.rebuild_raw, 'prepare-raw requires --rebuild-raw')
    need(args.action=='run' or not any((args.family,args.classifier,args.fold is not None)),
         'Task filters apply only to selected run, not whole-scope raw build/summary')
    execute(args,items,bindings)


if __name__ == '__main__':
    main()
