"""Local-only supplementary experiment engine; frozen source modules are read-only."""

# Preserve caller-specific encodings/block sizes; serializers stay local.
import sys as _io_sys
from pathlib import Path as _IOPath
_IO_MERGE_ROOT = next(p for p in _IOPath(__file__).resolve().parents if (p / 'io_shared.py').is_file())
if str(_IO_MERGE_ROOT) not in _io_sys.path:
    _io_sys.path.insert(0, str(_IO_MERGE_ROOT))
import io_shared as _io_shared
import hashlib
import json
from pathlib import Path
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True
from _paths import BASE, MODELS, activate
activate()
ROOT = BASE
LINEAR = NEURAL = MODELS
import joint_features as jf
import linear_models as lm
import neural_models as nm
# This process-local variable is only the checkpoint output allowlist.
# CODE and every network/source hash remain the original read-only module.
assert nm.CODE.resolve() == NEURAL.resolve()
nm.ROOT = ROOT
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits

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




