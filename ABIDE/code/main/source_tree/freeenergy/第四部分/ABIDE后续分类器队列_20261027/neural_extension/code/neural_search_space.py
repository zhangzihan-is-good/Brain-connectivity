"""Deterministic proposals for the neural extension; never fits any model."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
PART4 = ROOT.parent.parent
LINEAR_ROOT = PART4 / 'SVM_LR_全标签联合预筛_广泛搜索_20261027'
SOURCE = LINEAR_ROOT / 'code/search_space.py'
_spec = importlib.util.spec_from_file_location('_neural_original_parameter_axes', SOURCE)
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)
KINDS = ('gcn', 'fcn', 'lstm', 'latent_feature_transformer', 'made')
VERSION = 'neural-joint-first-deterministic-space-v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def unique(values):
    result, seen = [], set()
    for value in values:
        key = canonical(value)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def feature(core=None, atlas='aal', **overrides):
    return dict(core=dict(ss.DEFAULT_CORE, **(core or {})), K=overrides.get('K', ss.BUDGETS[atlas]),
                readout=overrides.get('readout', 'joint_topk'), quota=overrides.get('quota', .25),
                preselect_factor=overrides.get('preselect_factor', 1.))


def anchors(atlas, window, fold, default_model):
    candidates = [dict(features=feature(atlas=atlas), inputs=dict(scaler='standard', demographics='four'),
                       model=copy.deepcopy(default_model)),
                  dict(features=feature(atlas=atlas, readout='joint_then_quota', preselect_factor='all'),
                       inputs=dict(scaler='standard', demographics='four'), model=copy.deepcopy(default_model)),
                  dict(features=feature(dict(alpha=0.), atlas), inputs=dict(scaler='standard', demographics='four'),
                       model=copy.deepcopy(default_model))]
    sources = []
    for kind in ('svm', 'logistic'):
        path = LINEAR_ROOT / 'selection' / kind / atlas / ('W%d' % int(window)) / str(int(fold)) / 'selection.json'
        if not path.is_file():
            raise RuntimeError('Required finished linear warm seed missing: ' + str(path))
        raw = path.read_bytes()
        record = json.loads(raw.decode('utf-8'))
        if (record.get('classifier'), record.get('atlas'), int(record.get('window', -1)), int(record.get('fold', -1))) != (kind, atlas, int(window), int(fold)):
            raise RuntimeError('Linear warm parameter identity mismatch')
        old = record['chosen']['candidate']
        model = copy.deepcopy(default_model)
        model['beta'] = float(old['model']['beta'])
        candidates.append(dict(features=copy.deepcopy(old['features']), inputs=copy.deepcopy(old['inputs']), model=model))
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                            fields_used_for_seeding=['chosen.candidate.features', 'chosen.candidate.inputs', 'chosen.candidate.model.beta'],
                            model_weights_or_test_predictions_reused=False))
    return unique(candidates), sources


def proxy_features(atlas, window, fold, count=512):
    """Independent stratified proxy proposals; old fitted proxy scores are unused."""
    count = int(count)
    rng = random.Random(20261028 + ss.ATLASES.index(atlas) * 1000 + int(window) * 100 + int(fold))
    forced = [feature(atlas=atlas), feature(dict(alpha=0.), atlas),
              feature(atlas=atlas, readout='joint_then_quota', preselect_factor='all')]
    for name, values in ss.CORE_AXES.items():
        for value in (values[0], values[-1]):
            forced.append(feature({name: value}, atlas))
    for k in (ss._k_axis(atlas)[0], ss._k_axis(atlas)[-1]):
        forced.append(feature(atlas=atlas, K=k))
    for quota in (0., .25, 1.):
        for factor in ss.PRESELECT_FACTORS:
            forced.append(feature(atlas=atlas, readout='joint_then_quota', quota=quota, preselect_factor=factor))
    seen, result = set(), []
    def add(value):
        key = canonical(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    for value in forced:
        add(value)
    dimensions = list(ss.CORE_AXES.values()) + [ss._k_axis(atlas), ['joint_topk', 'joint_then_quota'],
                                               ss.QUOTAS, ss.PRESELECT_FACTORS]
    strata = [rng.sample(range(max(count, 1)), max(count, 1)) for _ in dimensions]
    draw = 0
    while len(result) < count:
        picks = []
        for axis, values in enumerate(dimensions):
            u = (strata[axis][draw] + rng.random()) / count if draw < count else rng.random()
            picks.append(values[min(len(values)-1, int(u * len(values)))])
        draw += 1
        core = dict(zip(ss.CORE_AXES, picks[:len(ss.CORE_AXES)]))
        k, readout, quota, factor = picks[len(ss.CORE_AXES):]
        if readout == 'joint_topk':
            quota, factor = .25, 1.
        add(feature(core, atlas, K=k, readout=readout, quota=quota, preselect_factor=factor))
    return result[:count]


def initial_state(atlas, kind, model_axes):
    axes = copy.deepcopy(ss.CORE_AXES)
    axes.update(K=ss._k_axis(atlas), quota=list(ss.QUOTAS), beta=list(ss.BETAS))
    axes.update({'model.' + key: copy.deepcopy(values) for key, values in model_axes.items()})
    return dict(version=VERSION, atlas=atlas, kind=kind, round=0, axes=axes, expanded_bounds=[])


def neighbors(candidate, atlas, kind, model_axes, numeric_model_axes=None, state=None):
    state = initial_state(atlas, kind, model_axes) if state is None else copy.deepcopy(state)
    if (state['version'], state['atlas'], state['kind']) != (VERSION, atlas, kind):
        raise ValueError('Coordinate state belongs to a different unit/schema')
    state['round'] += 1
    proposals, events = [], []
    def numeric(name, value, setter, **bounds):
        values, axis, logs = ss._numeric_neighbors(state['axes'][name], float(value), **bounds)
        state['axes'][name] = axis
        for log in logs:
            event = dict(round=state['round'], axis=name, **log)
            events.append(event)
            if event['event'] == 'soft_boundary_expanded':
                state['expanded_bounds'].append(event)
        for value in values:
            proposal = copy.deepcopy(candidate)
            setter(proposal, value)
            proposals.append(proposal)
    for name in ss.DEFAULT_CORE:
        numeric(name, candidate['features']['core'][name],
                lambda c, v, name=name: c['features']['core'].__setitem__(name, v),
                lower=0., upper=(1. if name in ('alpha', 'cap') else 100. if name == 'percentile' else None),
                lower_open=name in ('cap', 'percentile'), logarithmic=name in ('gamma', 'lambda_cost', 'alpha', 'delta'))
    numeric('K', candidate['features']['K'], lambda c, v: c['features'].__setitem__('K', int(v)),
            lower=1, upper=ss.SCALAR_POOLS[atlas], integer=True, logarithmic=False)
    numeric('beta', candidate['model']['beta'], lambda c, v: c['model'].__setitem__('beta', v), lower=0., lower_open=True)
    if candidate['features']['readout'] == 'joint_then_quota':
        numeric('quota', candidate['features']['quota'], lambda c, v: c['features'].__setitem__('quota', v),
                lower=0., upper=1., logarithmic=False)
    for factor in ss.PRESELECT_FACTORS:
        proposal = copy.deepcopy(candidate)
        proposal['features'].update(readout='joint_then_quota', preselect_factor=factor)
        proposals.append(proposal)
    proposal = copy.deepcopy(candidate)
    proposal['features'].update(readout=('joint_then_quota' if candidate['features']['readout'] == 'joint_topk' else 'joint_topk'),
                                preselect_factor=('all' if candidate['features']['readout'] == 'joint_topk' else 1.), quota=.25)
    proposals.append(proposal)
    for field, values in (('scaler', ss.SCALERS), ('demographics', ss.DEMOGRAPHICS)):
        for value in values:
            proposal = copy.deepcopy(candidate)
            proposal['inputs'][field] = value
            proposals.append(proposal)
    for key, values in model_axes.items():
        if key in (numeric_model_axes or {}):
            numeric('model.' + key, candidate['model']['config'][key],
                    lambda c, v, key=key: c['model']['config'].__setitem__(key, v),
                    **numeric_model_axes[key])
        else:
            for value in values:
                proposal = copy.deepcopy(candidate)
                proposal['model']['config'][key] = copy.deepcopy(value)
                proposals.append(proposal)
    return [p for p in unique(proposals) if canonical(p) != canonical(candidate)], events, state


def expanded_value(candidate, axis):
    if axis.startswith('model.'):
        return candidate['model']['config'].get(axis[6:])
    if axis in candidate['features']['core']:
        return candidate['features']['core'][axis]
    if axis in candidate['features']:
        return candidate['features'][axis]
    return candidate['model'].get(axis)
