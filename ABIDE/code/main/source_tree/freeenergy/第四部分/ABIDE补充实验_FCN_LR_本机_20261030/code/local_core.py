"""Local-only supplementary experiment engine; frozen source modules are read-only."""

# Preserve caller-specific encodings/block sizes; serializers stay local.
import sys as _io_sys
from pathlib import Path as _IOPath
_IO_MERGE_ROOT = next(p for p in _IOPath(__file__).resolve().parents if (p / 'io_shared.py').is_file())
if str(_IO_MERGE_ROOT) not in _io_sys.path:
    _io_sys.path.insert(0, str(_IO_MERGE_ROOT))
import io_shared as _io_shared
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import OrderedDict

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PART4 = ROOT.parent
LINEAR = PART4 / 'SVM_LR_全标签联合预筛_广泛搜索_20261027'
NEURAL = PART4 / 'ABIDE后续分类器队列_20261027' / 'neural_extension'
sys.path.insert(0, str(LINEAR / 'code'))
sys.path.insert(0, str(NEURAL / 'code'))
import joint_features as jf
import linear_models as lm
import neural_models as nm
# This process-local variable is only the checkpoint output allowlist.
# CODE and every network/source hash remain the original read-only module.
assert nm.CODE.resolve() == (NEURAL/'code').resolve()
nm.ROOT = ROOT
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits

METHODS = ('RGIC', 'Pearson', 'GCMI', 'PSI')
ROLES = ('train', 'valid', 'test')

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value

def sha(path):
    return _io_shared.sha256_file(path, block_size=4*1024*1024)

def identity(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read(path):
    return _io_shared.read_json(path, encoding="utf-8-sig")

def write(path, value):
    path = Path(path).resolve()
    if ROOT not in path.parents: raise RuntimeError('Write outside supplementary directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(str(temporary), str(path))

def metrics(y, prediction, score):
    y, prediction, score = np.asarray(y), np.asarray(prediction), np.asarray(score)
    tn, fp, fn, tp = confusion_matrix(1-y, 1-prediction, labels=[0, 1]).ravel()
    return dict(ACC=float(accuracy_score(y, prediction)),
                AUROC=float(roc_auc_score(1-y, score)) if len(np.unique(y)) == 2 else None,
                balanced_ACC=float(balanced_accuracy_score(y, prediction)),
                sensitivity=float(tp/(tp+fn)) if tp+fn else None,
                specificity=float(tn/(tn+fp)) if tn+fp else None)

class DataView:
    def __init__(self, metadata): self.meta = metadata
    def metadata(self): return self.meta
    def demographics(self, atlas, fold):
        roles = self.meta['splits'][atlas][str(fold)]
        raw = self.meta['demographics_raw']
        with np.errstate(invalid='ignore'):
            observed = np.isfinite(raw[roles['train']])
            fill = np.divide(np.where(observed, raw[roles['train']], 0).sum(0), observed.sum(0),
                             out=np.zeros(raw.shape[1], dtype=float), where=observed.sum(0)>0)
        filled = np.where(np.isfinite(raw), raw, fill).astype(np.float32)
        return {r: filled[ix] for r, ix in roles.items()}, dict(fit_scope='train_only', imputation_values=fill.tolist())

class CohortEngine(jf.FeatureEngine):
    def __init__(self, original, atlas, fold, roles=None, y=None, metadata=None, reader=None):
        self.root = ROOT
        self.reader = reader or original.reader
        meta = dict(metadata or original.metadata())
        meta['splits'] = {a: dict(v) for a, v in meta['splits'].items()}
        if roles is not None: meta['splits'][atlas][str(fold)] = {r: np.asarray(roles[r], dtype=np.int64) for r in ROLES}
        if y is not None: meta['y'] = np.asarray(y, dtype=np.int64)
        self.data = DataView(meta)
        self._components, self._scores = OrderedDict(), OrderedDict()

    def pool(self, atlas, window, fold, core, method='RGIC'):
        if method == 'RGIC':
            arrays = self.reader.base_arrays(atlas)
            if core['alpha'] == 0: arrays = dict(arrays, r0=self.reader.native_arrays(atlas)['r0'])
            eta, detail = self._gate(atlas, window, fold, core)
            n, e = arrays['r0'].shape
            s, d = self._rgic_values(arrays, window, np.arange(n), np.arange(e), core, eta)
            return s, d, detail
        return self.reader.candidates(atlas, method, window, fold)

    def from_pool(self, atlas, window, fold, params, strength, direction, method, detail=None,
                  selection_scope='all', include_test=True, fixed_columns=None):
        cfg = jf.normalize_params(params, atlas)
        roles, meta = self.roles(atlas, fold), self.metadata()
        selection_rows = roles['train'] if selection_scope == 'train' else np.concatenate([roles[r] for r in ROLES])
        width, has_d = strength.shape[1], direction is not None
        scores = np.empty(width*(2 if has_d else 1), dtype=np.float32)
        for lo in range(0, width, jf.F_CHUNK_COLUMNS):
            hi=min(width,lo+jf.F_CHUNK_COLUMNS)
            scores[lo:hi]=self._f(strength[selection_rows,lo:hi],meta['y'][selection_rows])
            if has_d: scores[width+lo:width+hi]=self._f(direction[selection_rows,lo:hi],meta['y'][selection_rows])
        if fixed_columns is None:
            selected, pool, target = jf.select_joint_pool(scores, width, has_d, cfg)
        else:
            selected = np.asarray(fixed_columns, dtype=np.int64); pool = selected; target = int(np.sum(selected >= width))
        mask = selected >= width
        edges = selected % width
        demos, demo_detail = self.data.demographics(atlas, fold)
        out = dict(direction_mask=mask.copy())
        for r in ROLES if include_test else ('train', 'valid'):
            ix = roles[r]
            x = np.asarray(strength[np.ix_(ix, edges)], dtype=np.float32).copy()
            if mask.any(): x[:, mask] = direction[np.ix_(ix, edges[mask])]
            out.update({'X_'+r:x, 'y_'+r:meta['y'][ix], 'y_'+r+'_ASD1':1-meta['y'][ix],
                        'demos_'+r:demos[r], 'demo_'+r:demos[r], 'ids_'+r:meta['ids'][ix], 'sites_'+r:meta['sites'][ix]})
        out['feature_metadata'] = dict(params=cfg, K=int(len(selected)), selected_columns=selected.tolist(),
            selected_column_ids=selected.tolist(), selected_edge_ids=edges.tolist(), selected_channel_ids=mask.astype(int).tolist(),
            selected_scores=[float(v) if np.isfinite(v) else None for v in scores[selected]],
            direction_mask=mask.tolist(), n_direction_selected=int(mask.sum()), n_scalar_selected=int((~mask).sum()),
            full_joint_candidate_count=int(width*(2 if has_d else 1)), joint_preselected_columns=pool.tolist(),
            target_direction_count=int(target), estimator=detail or {}, demographic=demo_detail,
            recipe=dict(version='local-supplement-v1', atlas=atlas, effective_window=int(window), fold=str(fold), method=method,
                        params=cfg, F_fit_scope=selection_scope, selection_scope=selection_scope,
                        selection_ids_sha256=jf._array_sha(meta['ids'][selection_rows]),
                        role_sha256={r:jf._array_sha(ix) for r,ix in roles.items()},
                        all_F_label_sha256=jf._array_sha(meta['y'][selection_rows]), full_test_labels_used_in_F=selection_scope=='all'))
        return out

    def prepare(self, atlas, window, fold, params, method='RGIC', include_test=False, selection_scope='all'):
        cfg = jf.normalize_params(params, atlas)
        s, d, detail = self.pool(atlas, window, fold, cfg['core'], method)
        return self.from_pool(atlas, window, fold, cfg, s, d, method, detail, selection_scope, include_test)

def source_candidate(classifier, atlas, window, fold):
    folder = NEURAL if classifier == 'fcn' else LINEAR
    path = folder / 'selection' / classifier / atlas / ('W%d'%window) / str(fold) / 'selection.json'
    doc = read(path)
    return copy.deepcopy(doc['chosen']['candidate']), dict(path=str(path), sha256=sha(path))

def default_candidate(classifier, k=1000, alpha=1.):
    return dict(features=dict(core=dict(jf.CORE_DEFAULTS, alpha=float(alpha)), K=int(k), readout='joint_topk',
                              quota=.25, preselect_factor=1.), inputs=dict(scaler='standard', demographics='four'),
                model=(dict(kind='fcn', beta=1., config=nm.default_config('fcn')) if classifier=='fcn' else
                       dict(kind='logistic', beta=1., C=1., penalty='l2', class_weight=None)))

def fit_view(view, classifier, candidate, label, seed=42, predict_test=True):
    started = time.perf_counter()
    prep = lm.prepare_inputs(view, candidate['inputs'])
    checkpoint = ROOT / 'models' / (label + '.pt')
    if classifier == 'fcn':
        beta = float(candidate['model']['beta'])
        values = {k: np.array(prep[k], copy=True) for k in ('a', 'z', 'x') if k in prep}
        for value in values.values(): value[:, prep['direction_mask']] *= beta
        n_conn = view['X_train'].shape[1]
        fm = view['feature_metadata']
        spec = dict(n_connection_features=n_conn, n_demographics=values['a'].shape[1]-n_conn,
                    n_rois=116, selected_edge_ids=fm['selected_edge_ids'], selected_channel_ids=fm['selected_channel_ids'])
        cfg = dict(candidate['model']['config'], device='cuda')
        result = nm.fit('fcn', values['a'], view['y_train'], values['z'], view['y_valid'],
                        checkpoint_path=checkpoint, config=cfg, seed=seed, input_spec=spec)
        valid = metrics(view['y_valid'], result['validation_prediction_TC1'], result['validation_score_ASD'])
        test_output = None
        if predict_test:
            out = nm.predict('fcn', checkpoint, values['a'], values['z'], values['x'], config=cfg, input_spec=spec, device='cuda')
            test_output = dict(prediction=out['test_prediction_TC1'], score=out['test_score_ASD'])
        bill = result.get('training_bill', {})
        model = dict(path=str(checkpoint), sha256=sha(checkpoint))
        del result, values
        nm.torch.cuda.empty_cache()
    else:
        evaluation = lm.PreparedEvaluator(prep['a'], view['y_train'], prep['z'], seed=seed, positive_label=0,
                                          direction_mask=prep['direction_mask'])
        result = evaluation.fit(candidate['model'])
        valid = metrics(view['y_valid'], result['prediction'], result['score'])
        test_output = lm.predict_model(result['model'], prep['x']) if predict_test else None
        bill = result.get('metadata', {})
        model = dict(kind='logistic', config=candidate['model'])
    answer = dict(valid=valid, fit_seconds=time.perf_counter()-started, training_bill=bill, model=model)
    if test_output is not None:
        answer.update(test=metrics(view['y_test'], test_output['prediction'], test_output['score']),
                      prediction_test=test_output['prediction'], score_test=test_output['score'])
    gc.collect()
    return answer

def record_task(task, view, candidate, protocol_sha, parameter_source, extra=None):
    tid = identity(dict(task=task, candidate=candidate, protocol_sha=protocol_sha))[:24]
    path = ROOT / 'records' / task['family'] / (tid+'.json')
    if path.exists():
        previous = read(path)
        if previous['status'] != 'complete': raise RuntimeError('Prior failed record requires explicit diagnosis: '+str(path))
        return previous
    import datetime
    row = dict(task, task_id=tid, status='running', protocol_sha=protocol_sha, parameter_source=parameter_source,
               candidate=candidate, label_semantics='ASD0_TC1', score_semantics='higher_ASD',
               feature_metadata=view['feature_metadata'], expected_folds=task.get('expected_folds', 10),
               ids_test=view['ids_test'], sites_test=view['sites_test'], y_test=view['y_test'],
               full_test_labels_used_in_F=view['feature_metadata']['recipe'].get('F_fit_scope', 'all')=='all',
               parameter_policy='frozen_original_validation_selected_configuration_counterfactual_refit',
               created_local=datetime.datetime.now().astimezone().isoformat())
    if extra: row.update(extra)
    write(ROOT/'run_state.json', dict(status='running', current_task=task, updated_local=row['created_local']))
    try:
        row.update(fit_view(view, task['classifier'], candidate, tid, task.get('seed',42)))
        row['status'] = 'complete'
    except Exception as e:
        import traceback
        row.update(status='failed', error=repr(e), traceback=traceback.format_exc())
        write(path, row)
        raise
    write(path, row)
    print(json.dumps(clean(dict(event='task_complete', **task, test=row['test'], fit_seconds=row['fit_seconds'])), ensure_ascii=False), flush=True)
    return row
