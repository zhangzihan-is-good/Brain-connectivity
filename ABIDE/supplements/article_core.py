"""Isolated local article extension: unchanged scientific modules, new outputs.

Import initializes the existing local DLL/runtime chain, never fits a model.
The only process-local source-module setting changed is nm.ROOT (write allowlist).
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
from _paths import BASE, activate
activate()
ROOT = BASE
import local_core as legacy
import control_features as controls

jf, lm, nm, np = legacy.jf, legacy.lm, legacy.nm, legacy.np
NEURAL = legacy.NEURAL
nm.ROOT = ROOT
assert nm.CODE.resolve() == NEURAL.resolve()
ROLES = ('train', 'valid', 'test')
clean, sha, read, metrics = legacy.clean, legacy.sha, legacy.read, legacy.metrics
threadpool_limits = legacy.threadpool_limits
TZ = timezone(timedelta(hours=8))


def now(): return datetime.now(TZ).isoformat()


def need(value, message):
    if not value: raise RuntimeError(message)


def require_local():
    need(os.name == 'nt' and not any(os.environ.get(k) for k in
         ('SLURM_JOB_ID', 'SLURM_JOBID', 'SLURM_CLUSTER_NAME', 'SLURM_STEP_ID')),
         'This article extension is restricted to the local Windows computer')


def output(path):
    path = Path(path).resolve()
    need(ROOT.resolve() in path.parents, 'Write outside new article extension')
    return path


def write(path, value):
    path = output(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


@contextmanager
def claim(name):
    """Windows process lock, released by the OS even on process termination."""
    require_local()
    import msvcrt
    path = output(ROOT / 'locks' / (name + '.lock')); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0: stream.write(b'\0'); stream.flush()
        stream.seek(0)
        try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error: raise RuntimeError('Another live owner holds ' + name) from error
        try: yield
        finally: stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)




def original_engine():
    """Initialize the unchanged read-only data methods without old output mkdir.

    Constructors restrict roots to direct fourth-part children. This nested
    execution adapter supplies the exact constructor fields, with PaperData's
    metadata destination inside V3. Missing old raw snapshots fail closed.
    """
    matched = jf._matched
    raw = matched._ReadOnlyRawData.__new__(matched._ReadOnlyRawData)
    raw.root = matched.OLD.resolve(); raw.inputs = raw.root / 'inputs'
    raw.data = matched.WORKSPACE / 'dataset/abide_data'
    raw.base_h5 = raw.data / '04_rgc_intermediate_data/rgc_base.hdf5'
    raw.pearson_h5 = raw.data / '03_pearson_fc/abide.hdf5'
    raw.raw_root = matched.PART4 / 'inputs/raw_timeseries'
    raw._meta = raw._headers = None
    raw._base, raw._native, raw._gates = {}, {}, {}
    for family in ('report_base', 'native_1035'):
        need((raw.inputs / family / 'aal/manifest.json').is_file(), 'Missing existing AAL raw snapshot')
    data = matched.PaperData.__new__(matched.PaperData)
    data.root = ROOT; data.inputs = output(ROOT / 'inputs')
    data.inputs.mkdir(parents=True, exist_ok=True)
    data.reader = raw; data._metadata = None; data._demos = {}
    obj = jf.FeatureEngine.__new__(jf.FeatureEngine)
    from collections import OrderedDict
    obj.root = ROOT; obj.data = data; obj.reader = raw
    obj._components, obj._scores = OrderedDict(), OrderedDict()
    obj.metadata()
    return obj


class CohortEngine(legacy.CohortEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = ROOT








def record_path(item): return ROOT / 'records' / item['task']['family'] / (item['task_id'] + '.json')


def _artifacts(row):
    out = []
    if row.get('model', {}).get('path'): out.append(row['model'])
    if row.get('oof'): out.append(row['oof'])
    if row.get('transform_artifact'): out.append(row['transform_artifact'])
    return out


def completed(item, protocol_sha):
    path = record_path(item)
    if not path.exists(): return None
    row = read(path)
    need(row.get('status') == 'complete', 'Existing incomplete/failed record needs diagnosis: ' + str(path))
    need(row.get('protocol_sha') == protocol_sha and row['task_id'] == item['task_id'] and
         row['candidate'] == item['candidate'], 'Completed record identity differs')
    for k, value in item['task'].items(): need(row.get(k) == value, 'Completed task field differs: ' + k)
    for name in ('ids_test','y_test','sites_test'):
        need(row[name] == item['expected_'+name], 'Completed OOF identity differs: '+name)
    receipt = ROOT / 'receipts' / (item['task_id'] + '.json')
    doc = read(receipt)
    need(doc['record_sha256'] == sha(path) and doc['protocol_sha256'] == protocol_sha,
         'Completed record receipt differs')
    for ref in _artifacts(row): need(sha(ref['path']) == ref['sha256'], 'Saved artifact differs')
    if row.get('old_source'):
        ref = row['old_source']; need(sha(ref['path']) == ref['sha256'], 'Reused source record differs')
    return row


def _commit(item, row, protocol_sha):
    row['manifest_task_id'] = item['task_id']
    row['artifacts'] = _artifacts(row)
    path = record_path(item); write(path, row)
    write(ROOT / 'receipts' / (item['task_id'] + '.json'),
          dict(status='complete', task_id=item['task_id'], protocol_sha256=protocol_sha,
               record_path=str(path), record_sha256=sha(path), artifacts=_artifacts(row), finished_local=now()))




def fit_view(view, classifier, candidate, tid, seed=42):
    """Use the original classifier/scaler arithmetic and persist both families."""
    need(nm.ROOT.resolve() == ROOT.resolve(), 'Neural output allowlist changed')
    prep = lm.prepare_inputs(view, candidate['inputs'])
    modelpath = output(ROOT / 'models' / (tid + ('.pt' if classifier == 'fcn' else '.pkl')))
    need(not modelpath.exists(), 'Orphan model artifact: refuse an automatic refit')
    modelpath.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if classifier == 'fcn':
        beta = float(candidate['model']['beta'])
        values = {k: np.array(prep[k], copy=True) for k in ('a', 'z', 'x')}
        for value in values.values(): value[:, prep['direction_mask']] *= beta
        fm = view['feature_metadata']; n_conn = view['X_train'].shape[1]
        spec = dict(n_connection_features=n_conn, n_demographics=values['a'].shape[1]-n_conn,
                    n_rois=116, selected_edge_ids=fm['selected_edge_ids'], selected_channel_ids=fm['selected_channel_ids'])
        cfg = dict(candidate['model']['config'], device='cuda')
        result = nm.fit('fcn', values['a'], view['y_train'], values['z'], view['y_valid'],
                        checkpoint_path=modelpath, config=cfg, seed=seed, input_spec=spec)
        prediction = nm.predict('fcn', modelpath, values['a'], values['z'], values['x'],
                                config=cfg, input_spec=spec, device='cuda')
        valid = metrics(view['y_valid'], result['validation_prediction_TC1'], result['validation_score_ASD'])
        pred, score = prediction['test_prediction_TC1'], prediction['test_score_ASD']
        bill = result.get('training_bill', {})
        del result, values, prediction
        nm.torch.cuda.empty_cache()
    else:
        evaluator = lm.PreparedEvaluator(prep['a'], view['y_train'], prep['z'], seed=seed,
                                        positive_label=0, direction_mask=prep['direction_mask'])
        result = evaluator.fit(candidate['model']); prediction = lm.predict_model(result['model'], prep['x'])
        valid = metrics(view['y_valid'], result['prediction'], result['score'])
        pred, score = prediction['prediction'], prediction['score']; bill = result.get('metadata', {})
        tmp = modelpath.with_name(modelpath.name + '.' + uuid.uuid4().hex + '.partial')
        with tmp.open('xb') as stream: pickle.dump(dict(model=result['model'], transform=prep['transform']), stream, protocol=4)
        os.replace(tmp, modelpath)
    transform_path = ROOT / 'models' / (tid + '.transform.json'); write(transform_path, prep['transform'])
    oofpath = ROOT / 'oof' / (tid + '.json')
    write(oofpath, dict(ids_test=view['ids_test'], sites_test=view['sites_test'], y_test=view['y_test'],
                       prediction_test=pred, score_test=score, score_semantics='higher_ASD', label_semantics='ASD0_TC1'))
    result = dict(valid=valid, test=metrics(view['y_test'], pred, score), prediction_test=pred,
                  score_test=score, fit_seconds=time.perf_counter()-started, training_bill=bill,
                  model=dict(path=str(modelpath), sha256=sha(modelpath)),
                  transform_artifact=dict(path=str(transform_path), sha256=sha(transform_path)),
                  oof=dict(path=str(oofpath), sha256=sha(oofpath)))
    gc.collect(); return result


def fit_record(item, view, protocol_sha, extra=None):
    prior = completed(item, protocol_sha)
    if prior is not None: return prior
    need(len(view['ids_test']) == len(set(map(str, view['ids_test']))), 'Duplicate test IDs')
    for name in ('ids_test','y_test','sites_test'):
        need(clean(view[name]) == item['expected_'+name], 'Current OOF identity differs: '+name)
    need(view['feature_metadata']['K'] == item['candidate']['features']['K'], 'K changed')
    need(sum(len(view['ids_'+r]) for r in ROLES) == item['task']['expected_test_n'], 'Active cohort differs')
    row = dict(item['task'], task_id=item['task_id'], protocol_sha=protocol_sha,
               candidate=item['candidate'], parameter_source=item['parameter_source'], status='running',
               feature_metadata=view['feature_metadata'], ids_test=view['ids_test'], sites_test=view['sites_test'],
               y_test=view['y_test'], full_test_labels_used_in_F=True, label_semantics='ASD0_TC1',
               score_semantics='higher_ASD', record_origin='new_counterfactual_fit', no_new_parameter_search=True,
               parameter_policy='frozen_original_validation_selected_configuration_counterfactual_refit',
               demographic_predictors_not_used_in_RGIC_estimator=True, created_local=now(),
               article_task_manifest_sha256=sha(ROOT/'inputs/task_manifest.json'))
    if extra: row.update(extra)
    path = record_path(item); write(path, row)
    write(ROOT/'run_state.json', dict(status='running', current_task=item['task'], updated_local=now()))
    try:
        row.update(fit_view(view, item['task']['classifier'], item['candidate'], item['task_id'], item['task']['seed']))
        row.update(status='complete', finished_local=now()); _commit(item, row, protocol_sha)
    except BaseException as error:
        row.update(status='failed', error=repr(error), traceback=traceback.format_exc(), finished_local=now())
        write(path, row); raise
    print(json.dumps(clean(dict(event='task_complete', **item['task'], task_id=item['task_id'])), ensure_ascii=False), flush=True)
    return row
