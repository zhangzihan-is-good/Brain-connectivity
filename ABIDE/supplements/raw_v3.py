"""Local AAL/W1, all-1035 true-series robustness inputs, isolated v3 cache.

Thirteen declared conditions; original nine native methods are recomputed.
No downloads, classifiers, score-driven filtering or implicit build on import.
"""
from __future__ import annotations

# Preserve caller-specific encodings/block sizes; serializers stay local.
import sys as _io_sys
from pathlib import Path as _IOPath
_IO_MERGE_ROOT = next(p for p in _IOPath(__file__).resolve().parents if (p / 'io_shared.py').is_file())
if str(_IO_MERGE_ROOT) not in _io_sys.path:
    _io_sys.path.insert(0, str(_IO_MERGE_ROOT))
import io_shared as _io_shared

from contextlib import contextmanager
from datetime import datetime
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
from _paths import BASE, SUPPLEMENTS, activate
activate()
from shared_paths import workspace_root
ROOT = BASE
WORKSPACE = workspace_root()
PART4 = WORKSPACE / '第四部分'
CACHE = ROOT / 'raw_cache'
PLAN = CACHE / 'raw_plan.json'
OLD_RAW = SUPPLEMENTS / 'raw_robustness.py'
COHORT = PART4 / 'ABIDE后续分类器队列_20261027/inputs/original_cohort_and_folds.json'
RAW = PART4 / 'inputs/raw_timeseries'
PHENOTYPES = WORKSPACE / 'dataset/abide_data/01_subject_metadata/Phenotypic_V1_0b_preprocessed1.csv'
BASE_H5 = WORKSPACE / 'dataset/abide_data/04_rgc_intermediate_data/rgc_base.hdf5'
METHODS = ('RGIC', 'Lag_peak', 'Gaussian_TE', 'VAR_Granger', 'PSI',
           'PhiID_unique_transfer', 'Pearson', 'Coherence', 'GCMI')
BASE_KEYS = ('r0', 'W1_gain', 'W1_cost', 'W1_peak_lag', 'W1_peak_rho', 'W1_potential')
NATIVE_KEYS = ('r0', 'W1_rstar', 'W1_lag', 'W1_tie', 'te_h1_d1', 'var_p1',
               'psi', 'coherence', 'gcmi', 'phiid_d1')
PAIRS = frozenset(('te_h1_d1', 'var_p1', 'phiid_d1'))
CONDITIONS = {f'prefix_{int(f*100):03d}': dict(fraction=f, sigma=0., seed=None)
              for f in (.25, .5, .75, 1.)}
CONDITIONS.update({f'noise_{int(s*100):03d}_seed{seed}': dict(fraction=1., sigma=s, seed=seed)
                   for s in (.1, .2, .3) for seed in (101, 202, 303)})
_OLD = None
_SOURCE_INDEX = None


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read(path):
    return _io_shared.read_json(path, encoding="utf-8-sig")


def sha(path):
    return _io_shared.sha256_file(path, block_size=8*1024*1024)


def output(path):
    path = Path(path).resolve()
    CACHE.resolve().relative_to(ROOT.resolve())
    path.relative_to(CACHE.resolve())
    return path


def atomic(path, value):
    path = output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temporary.open('x', encoding='utf-8') as handle:
        handle.write(canonical(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def local_only():
    need(os.name == 'nt' and not os.environ.get('SLURM_JOB_ID'),
         'Article extension raw construction is restricted to this local Windows computer')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old():
    """Reuse hash-bound estimator bodies; no old select/build/cache API calls."""
    global _OLD
    if _OLD is None:
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        # Same proven DLL/runtime bootstrap as the original local experiment.
        # It changes process-local output variables only; no model is fitted.
        if 'numpy' not in sys.modules:
            load_module('_article_raw_v3_runtime', SUPPLEMENTS / 'local_core.py')
        _OLD = load_module('_article_raw_v3_readonly_old', OLD_RAW)
        need(_OLD.np.__version__ == '1.26.4', 'Wrong numerical runtime: NumPy1.26.4 required')
    return _OLD


def source_paths():
    paths = dict(old().SOURCES)
    paths.update(old_raw_robustness=OLD_RAW, runtime_bootstrap=SUPPLEMENTS / 'local_core.py',
                 original_cohort=COHORT, phenotype=PHENOTYPES, original_HDF5_metadata=BASE_H5,
                 AAL_source_manifest=RAW / 'aal_source_manifest.json',
                 TR_manifest=RAW / 'header_tr_manifest.json')
    return paths


def sources():
    return {key: dict(path=str(path.resolve()), sha256=sha(path)) for key, path in source_paths().items()}


def source_index():
    global _SOURCE_INDEX
    if _SOURCE_INDEX is None:
        rows = read(RAW / 'aal_source_manifest.json')['files']
        _SOURCE_INDEX = {row['subject_id']: row for row in rows}
        need(len(rows) == len(_SOURCE_INDEX) == 1035, 'AAL source manifest has duplicate/missing identities')
    return _SOURCE_INDEX


def inventory(verify_bytes=False):
    cohort = read(COHORT)
    ids, index = cohort['ids'], source_index()
    need(len(ids) == len(set(ids)) == 1035, 'Original full cohort is not exactly1035 unique people')
    records = []
    for sid in ids:
        need(sid in index, 'Subject absent from AAL provenance: ' + sid)
        path = RAW / 'aal' / (sid + '_rois_aal.1D')
        need(path.is_file(), 'Missing original ROI series: ' + sid)
        row = index[sid]
        need(int(row['n_rois']) == 116 and int(row['n_timepoints']) >= 48,
             'Source violates116-ROI or minimum12-point quarter-prefix requirement: ' + sid)
        if verify_bytes:
            need(sha(path) == row['sha256'], 'Original ROI source bytes changed: ' + sid)
        records.append(dict(subject_id=sid, path=str(path), sha256=row['sha256'],
                            bytes=path.stat().st_size, n_timepoints=int(row['n_timepoints']),
                            tr_seconds=float(row['tr_seconds'])))
    return dict(status='all_1035_sources_present' if not verify_bytes else 'all_1035_source_hashes_verified',
                n=1035, atlas='aal', window=1, n_rois=116, n_edges=6670,
                selection_ids=ids, selection_ids_sha256=identity(ids),
                source_bytes=sum(row['bytes'] for row in records),
                n_timepoints_min=min(row['n_timepoints'] for row in records),
                n_timepoints_max=max(row['n_timepoints'] for row in records),
                all_source_file_bytes_hashed=bool(verify_bytes), files=records,
                conditions=CONDITIONS, n_conditions=len(CONDITIONS),
                n_subject_condition_recomputations=1035 * len(CONDITIONS),
                original_preprocessing='Public PCP CPAC/filt_global delivered ROI series; no renewed acquisition preprocessing or refiltering',
                no_classifier_results_read=True)


def declare():
    """Freeze full cohort identities, input bytes and source bodies before build."""
    local_only()
    inv = inventory(verify_bytes=True)
    cohort, raw = read(COHORT), old()
    import h5py
    attributes = {}
    with h5py.File(BASE_H5, 'r') as handle:
        for sid in inv['selection_ids']:
            a = handle['patients'][sid].attrs
            attributes[sid] = dict(site=str(a['site']), y=int(a['y']), sex=float(a['sex']), age=float(a['age']))
    need(len({a['site'] for a in attributes.values()}) == 17, 'Original17 merged sites changed')
    need(sum(a['y'] == 0 for a in attributes.values()) == 505, 'Original ASD505/TC530 labels changed')
    with PHENOTYPES.open(encoding='utf-8-sig', newline='') as handle:
        rows = {r['FILE_ID']: r for r in csv.DictReader(handle) if r['FILE_ID'] in attributes}
    need(set(rows) == set(attributes), 'Phenotype join does not cover all1035 identities')
    matched = raw.modules()['matched']
    for sid, a in attributes.items():
        row = rows[sid]
        demo = [matched.numeric(row['SEX']), matched.numeric(row['AGE_AT_SCAN']),
                matched.HAND_MAP.get(row.get('HANDEDNESS_CATEGORY', '').strip().lower(), float('nan')),
                matched.numeric(row.get('FIQ'))]
        a['demographics_raw'] = [float(x) if raw.np.isfinite(x) else None for x in demo]
    value = dict(version='article_raw_AAL_W1_all1035_v3', n=1035, atlas='aal', windows=[1],
                 selection_ids=inv['selection_ids'], selection_ids_sha256=inv['selection_ids_sha256'],
                 conditions=CONDITIONS, methods=list(METHODS), attributes=attributes,
                 inventory=inv, source_sha256=sources(), code_sha256=sha(__file__),
                 original_fold_roles_sha256=identity(cohort['folds']),
                 noise_rule='PCG64 seed from SHA256([article_raw_v3, original_numeric_seed, subject_id]); common standard-normal field across sigma and all methods; add sigma*full-original-ROI-std(ddof0) before original standardizations; originally constant ROI amplitude0',
                 prefix_rule='First floor(original_T*fraction) delivered preprocessed frames; no resampling, refiltering, padding or cross-run manipulation',
                 no_exclusions_or_score_based_rules=True, no_old_200_values_reused=True)
    if PLAN.exists():
        need(read(PLAN) == value, 'Existing raw declaration differs; no silent redeclaration')
    else:
        atomic(PLAN, value)
    return value


def plan():
    value = read(PLAN)
    need(value['version'] == 'article_raw_AAL_W1_all1035_v3' and value['n'] == 1035,
         'Wrong raw declaration/version/population')
    need(value['conditions'] == CONDITIONS and value['methods'] == list(METHODS), 'Declared condition/method set changed')
    need(value['selection_ids'] == read(COHORT)['ids'] and identity(value['selection_ids']) == value['selection_ids_sha256'],
         'All1035 identity order changed')
    need(value['code_sha256'] == sha(__file__), 'Raw implementation changed after declaration')
    for item in value['source_sha256'].values():
        need(sha(item['path']) == item['sha256'], 'Bound scientific/input source changed: ' + item['path'])
    return value


def freeze_guard(protocol_path):
    local_only()
    protocol_path = Path(protocol_path).resolve()
    protocol_path.relative_to(ROOT.resolve())
    protocol = read(protocol_path)
    need(protocol.get('status') == 'frozen_before_new_test_scoring', 'Article extension protocol is not frozen')
    bindings = {str(Path(p).resolve()): digest for p, digest in protocol['execution_sha256'].items()}
    value = plan()
    required = [Path(__file__).resolve(), PLAN.resolve()] + [Path(x['path']).resolve() for x in value['source_sha256'].values()]
    for path in required:
        need(bindings.get(str(path)) == sha(path), 'Protocol does not bind current raw source/declaration: ' + str(path))
    return value, sha(protocol_path)


def series(subject_id):
    np = old().np
    row = source_index()[subject_id]
    path = RAW / 'aal' / (subject_id + '_rois_aal.1D')
    need(sha(path) == row['sha256'], 'Raw ROI source changed: ' + subject_id)
    with path.open(encoding='ascii') as handle:
        tokens = handle.readline().split()
    order = np.argsort(np.asarray([int(token[1:]) for token in tokens]))
    x = np.loadtxt(path, dtype=np.float64)[:, order]
    need(x.shape == (int(row['n_timepoints']), 116) and np.isfinite(x).all(), 'Invalid full source array')
    return x, dict(path=str(path), sha256=row['sha256'], roi_ids_sorted=[tokens[int(i)] for i in order],
                   tr_seconds=float(row['tr_seconds']))


def condition_series(values, condition, subject_id):
    np = old().np
    need(condition in CONDITIONS, 'Unknown declared raw condition')
    cfg = CONDITIONS[condition]
    n = int(np.floor(len(values) * cfg['fraction']))
    need(n >= 12, 'Declared prefix has fewer than12frames; no length fallback')
    x = np.array(values[:n], copy=True)
    derived_seed = None
    if cfg['sigma']:
        std = values.std(axis=0, ddof=0)
        std = np.where(np.ptp(values, axis=0) == 0, 0., std)
        derived_seed = int(identity(['article_raw_v3', cfg['seed'], subject_id])[:16], 16)
        rng = np.random.Generator(np.random.PCG64(derived_seed))
        x += cfg['sigma'] * std[None, :] * rng.standard_normal(x.shape)
    return x, dict(condition=condition, original_n_timepoints=len(values), n_timepoints=len(x),
                   prefix_fraction=cfg['fraction'], sigma_original_full_ROI_std=cfg['sigma'],
                   declared_noise_seed=cfg['seed'], derived_PCG64_seed=derived_seed,
                   common_perturbation_across_all_methods=True, common_standard_noise_across_sigma=True,
                   noise_before_standardization=True, temporal_refiltering=False,
                   preprocessing='Previously delivered CPAC/filt_global ROI series')


def compute_subject(subject_id, condition):
    """Literal old scientific calls/order; only identities/conditions/output differ."""
    started = time.perf_counter()
    raw = old()
    np, m = raw.np, raw.modules()
    x0, source = series(subject_id)
    x, details = condition_series(x0, condition, subject_id)
    active = np.flatnonzero(np.ptp(x, axis=0) != 0)
    need(len(active) >= 2, 'Fewer than two nonconstant ROIs; no subject exclusion fallback')
    y = x[:, active]
    m['estimators']._validate(y)
    with raw.threadpool_limits(limits=1):
        standardized = m['rgc'].standardize_rows(y.T)
        report = m['rgc'].compute_subject_rgc_base(standardized, windows=(1,), gamma=.2, delta=0., c_max=.9)
        out = m['estimators'].correlation_primitives(y)
        prediction, pred_diag = m['preparation'].prediction_required(y)
        out.update(prediction)
        spectral, spectral_diag = m['preparation'].spectral_required(y, source['tr_seconds'])
        out.update(spectral)
        out.update(m['estimators'].gcmi_primitives(y))
        phi = m['estimators'].phiid_primitives(y)
        out.update(phi)
    base_source = dict(r0=report.pearson, W1_gain=report.gain[1], W1_cost=report.cost[1],
                       W1_peak_lag=report.peak_lag[1], W1_peak_rho=report.peak_rho[1], W1_potential=report.potential[1])
    hi, lo = np.tril_indices(len(active), -1)
    base_index = active[hi] * (active[hi] - 1) // 2 + active[lo]
    i, j = np.triu_indices(len(active), 1)
    native_index = active[j] * (active[j] - 1) // 2 + active[i]
    base, native = {}, {}
    for key in BASE_KEYS:
        value = np.asarray(base_source[key])
        need(np.isfinite(value).all(), 'Nonfinite RGIC primitive: ' + key)
        dest = np.zeros(6670, dtype=np.int16 if key == 'W1_peak_lag' else np.float32)
        dest[base_index] = value
        base[key] = dest
    for key in NATIVE_KEYS:
        value = np.asarray(out[key])
        need(np.isfinite(value).all(), 'Nonfinite native primitive: ' + key)
        dtype = np.int8 if key == 'W1_lag' else np.bool_ if key == 'W1_tie' else np.float32
        dest = np.zeros((6670, 2) if key in PAIRS else 6670, dtype=dtype)
        dest[native_index] = value
        native[key] = dest
    details.update(subject_id=subject_id, n_rois=116, n_edges=6670, active_rois=active.tolist(),
                   constant_roi_indices=np.setdiff1d(np.arange(116), active).tolist(), tr_seconds=source['tr_seconds'],
                   input_source=source, prediction_diagnostics=pred_diag, spectral_diagnostics=spectral_diag,
                   phiid_failed_d1=int(np.sum(phi['phiid_failed_d1'])),
                   elapsed_seconds=time.perf_counter() - started,
                   estimator_order='Archived literal double RGIC standardization; native estimators see perturbed raw values',
                   constant_policy='Same original1035 native policy: compute on nonconstant ROIs; zero all incident fields after scatter',
                   edge_order='np.tril_indices(R,-1); native channels lowROI->highROI and reverse unchanged',
                   group_gate='Inputs only; eta must be fitted on current condition training identities by downstream reader/engine')
    return base, native, details


def validate_arrays(base, native):
    np = old().np
    need(set(base) == set(BASE_KEYS) and set(native) == set(NATIVE_KEYS), 'Wrong primitive field set')
    for family, arrays in (('report_base', base), ('native_1035', native)):
        for key, value in arrays.items():
            shape = (6670, 2) if family == 'native_1035' and key in PAIRS else (6670,)
            dtype = np.int16 if family == 'report_base' and key == 'W1_peak_lag' else (
                np.int8 if family == 'native_1035' and key == 'W1_lag' else
                np.bool_ if family == 'native_1035' and key == 'W1_tie' else np.float32)
            need(value.shape == shape and value.dtype == np.dtype(dtype) and np.isfinite(value).all(),
                 'Invalid cached field: ' + family + '/' + key)


def recipe(value, subject_id, condition):
    return dict(plan_sha256=sha(PLAN), code_sha256=value['code_sha256'],
                selection_ids_sha256=value['selection_ids_sha256'], subject_id=subject_id,
                input_sha256=source_index()[subject_id]['sha256'], condition=condition,
                parameters=CONDITIONS[condition])


def checkpoint(value, subject_id, condition):
    np = old().np
    folder = output(CACHE / condition / 'subject_checkpoints')
    folder.mkdir(parents=True, exist_ok=True)
    path, note_path = folder / (subject_id + '.npz'), folder / (subject_id + '.json')
    prepared = folder / (subject_id + '.prepared.json')
    expected = recipe(value, subject_id, condition)
    if prepared.exists():
        item = read(prepared)
        need(item['recipe'] == expected, 'Interrupted checkpoint has another recipe')
        tmp = output(item['temporary_path'])
        source = path if path.exists() else tmp
        need(source.is_file() and sha(source) == item['npz_sha256'], 'Interrupted checkpoint bytes do not match')
        if not path.exists():
            os.replace(tmp, path)
        if note_path.exists():
            need(read(note_path) == item['receipt'], 'Interrupted checkpoint receipt differs')
        else:
            atomic(note_path, item['receipt'])
        prepared.unlink()
    if path.exists() or note_path.exists():
        need(path.is_file() and note_path.is_file(), 'Unjournaled incomplete checkpoint; explicit review required')
        note = read(note_path)
        need(note['recipe'] == expected and sha(path) == note['npz_sha256'], 'Checkpoint recipe/bytes changed')
        with np.load(path, allow_pickle=False) as handle:
            base = {key: handle['b__' + key] for key in BASE_KEYS}
            native = {key: handle['n__' + key] for key in NATIVE_KEYS}
        validate_arrays(base, native)
        return base, native, note['details']
    base, native, details = compute_subject(subject_id, condition)
    validate_arrays(base, native)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with tmp.open('xb') as handle:
        np.savez(handle, **{'b__' + k: v for k, v in base.items()}, **{'n__' + k: v for k, v in native.items()})
        handle.flush()
        os.fsync(handle.fileno())
    note = dict(status='complete', recipe=expected, npz_sha256=sha(tmp), details=details)
    atomic(prepared, dict(recipe=expected, temporary_path=str(tmp), npz_sha256=note['npz_sha256'], receipt=note))
    os.replace(tmp, path)
    atomic(note_path, note)
    prepared.unlink()
    return base, native, details


def recover_aggregate(condition, value):
    folder = output(CACHE / condition)
    pending = folder / 'aggregate_prepared.json'
    if not pending.exists():
        return False
    journal = read(pending)
    need(journal['plan_sha256'] == sha(PLAN) and journal['condition'] == condition, 'Interrupted aggregate recipe differs')
    for item in journal['publications']:
        final, temporary = output(item['path']), output(item['temporary_path'])
        source = final if final.exists() else temporary
        need(source.is_file() and sha(source) == item['sha256'], 'Interrupted aggregate bytes changed')
        if not final.exists():
            os.replace(temporary, final)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        need(read(manifest) == journal['manifest'], 'Interrupted aggregate manifest differs')
    else:
        atomic(manifest, journal['manifest'])
    pending.unlink()
    return True


def aggregate(condition, value):
    np = old().np
    folder = output(CACHE / condition)
    if recover_aggregate(condition, value) or (folder / 'manifest.json').exists():
        reader = RawV3Reader(CACHE, condition, declaration=value)
        reader.base_arrays('aal')
        reader.native_arrays('aal')
        return
    arrays, publications, subject_receipts = {}, [], []
    token = uuid.uuid4().hex
    handles = {}
    try:
        for row, sid in enumerate(value['selection_ids']):
            base, native, details = checkpoint(value, sid, condition)
            for family, values in (('report_base', base), ('native_1035', native)):
                for key, v in values.items():
                    pair = (family, key)
                    if pair not in handles:
                        directory = output(folder / family / 'aal')
                        directory.mkdir(parents=True, exist_ok=True)
                        path = directory / (key + '.npy')
                        need(not path.exists(), 'Aggregate without publication journal already exists: ' + str(path))
                        tmp = path.with_name(path.name + '.' + token + '.partial')
                        handles[pair] = (np.lib.format.open_memmap(tmp, mode='w+', dtype=v.dtype, shape=(1035,) + v.shape), path, tmp)
                    handles[pair][0][row] = v
            receipt = folder / 'subject_checkpoints' / (sid + '.json')
            subject_receipts.append(dict(subject_id=sid, path=str(receipt), sha256=sha(receipt), npz_sha256=read(receipt)['npz_sha256']))
            atomic(CACHE / 'build_state.json', dict(status='building', condition=condition, completed_subjects=row + 1,
                   n_subjects=1035, n_conditions=13, last_subject=sid, updated_local=datetime.now().astimezone().isoformat()))
            print(canonical(dict(condition=condition, n_complete=row + 1, n_subjects=1035,
                                 subject=sid, elapsed_seconds=details['elapsed_seconds'])), flush=True)
        for (family, key), (handle, path, tmp) in handles.items():
            handle.flush()
            desc = dict(shape=list(handle.shape), dtype=handle.dtype.str)
            handle._mmap.close()
            desc.update(path=str(path), sha256=sha(tmp), bytes=tmp.stat().st_size)
            arrays.setdefault(family, {})[key] = desc
            publications.append(dict(path=str(path), temporary_path=str(tmp), sha256=desc['sha256']))
        handles.clear()
        manifest = dict(status='complete', condition=condition, condition_parameters=CONDITIONS[condition],
                        n_subjects=1035, atlas='aal', window=1, methods=list(METHODS),
                        selection_ids_sha256=value['selection_ids_sha256'], plan_sha256=sha(PLAN),
                        code_sha256=sha(__file__), arrays=arrays, subject_receipts=subject_receipts,
                        no_subject_exclusions=True, no_old_200_values_reused=True,
                        read_validation_scope='All16 aggregate array bytes and1035 receipt metadata hashes; per-subject NPZ bytes checked during build/resume, not re-read by model readers')
        atomic(folder / 'aggregate_prepared.json', dict(condition=condition, plan_sha256=sha(PLAN),
               publications=publications, manifest=manifest))
        recover_aggregate(condition, value)
    finally:
        for handle, _, _ in handles.values():
            if not handle._mmap.closed:
                handle._mmap.close()


@contextmanager
def build_claim():
    """One local builder, actual OS file lock; stale owner metadata is not a lease."""
    local_only()
    import msvcrt
    output(CACHE).mkdir(parents=True, exist_ok=True)
    lock = CACHE / 'build.lock'
    with lock.open('a+b') as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        token = uuid.uuid4().hex
        owner = CACHE / 'build_owner.json'
        previous = read(owner) if owner.exists() else None
        atomic(owner, dict(pid=os.getpid(), token=token, lock='Windows kernel byte-range lock; local disk only',
                          previous_owner_metadata=previous, started_local=datetime.now().astimezone().isoformat()))
        try:
            yield
        finally:
            if owner.exists() and read(owner).get('token') == token:
                owner.unlink()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def build(protocol_path, conditions=tuple(CONDITIONS)):
    need(len(conditions) == len(set(conditions)) and set(conditions).issubset(CONDITIONS), 'Unknown/duplicate conditions')
    value, protocol_sha = freeze_guard(protocol_path)
    current = None
    with build_claim():
        try:
            for current in conditions:
                aggregate(current, value)
            complete = []
            for name in CONDITIONS:
                if (CACHE / name / 'manifest.json').exists():
                    verified = RawV3Reader(CACHE, name, declaration=value)
                    verified.base_arrays('aal')
                    verified.native_arrays('aal')
                    complete.append(name)
            atomic(CACHE / 'build_state.json', dict(status='complete' if len(complete) == 13 else 'requested_conditions_complete',
                   completed_conditions=complete, requested_conditions=list(conditions), n_subjects=1035,
                   expected_conditions=13, protocol_sha256=protocol_sha, updated_local=datetime.now().astimezone().isoformat()))
        except BaseException as error:
            atomic(CACHE / 'build_state.json', dict(status='failed_preserved_results', condition=current, n_subjects=1035,
                   error=repr(error), traceback=traceback.format_exc(), protocol_sha256=protocol_sha,
                   updated_local=datetime.now().astimezone().isoformat(), no_exclusions_or_numerical_fallback=True))
            raise


class RawV3Reader:
    """Read-only original ReportData candidate/gate API, full1035 original roles."""
    def __init__(self, cache_root=CACHE, condition='prefix_100', declaration=None):
        need(Path(cache_root).resolve() == CACHE.resolve(), 'Raw cache root must be the isolated article-v3 root')
        need(condition in CONDITIONS, 'Unknown declared condition')
        self.cache, self.condition = CACHE, condition
        self.inputs = output(CACHE / condition)
        self.receipt = plan()
        if declaration is not None:
            need(identity(declaration) == identity(self.receipt), 'Supplied declaration does not equal verified raw_plan.json')
        self.done = read(self.inputs / 'manifest.json')
        need(self.done['status'] == 'complete' and self.done['n_subjects'] == 1035 and
             self.done['condition'] == condition and self.done['condition_parameters'] == CONDITIONS[condition] and
             self.done['selection_ids_sha256'] == self.receipt['selection_ids_sha256'] and
             self.done['plan_sha256'] == sha(PLAN) and self.done['code_sha256'] == sha(__file__),
             'Condition manifest is incomplete or belongs to another declaration')
        need([r['subject_id'] for r in self.done['subject_receipts']] == self.receipt['selection_ids'],
             'Condition subject receipts do not cover exact1035 identity order')
        self._base, self._native, self._gates, self._meta = {}, {}, {}, None
        self._receipt_metadata_verified = False

    def metadata(self):
        if self._meta is not None:
            return self._meta
        np = old().np
        ids, cohort = self.receipt['selection_ids'], read(COHORT)
        index = {sid: i for i, sid in enumerate(ids)}
        splits = {a: {str(f): {role: np.asarray([index[s] for s in subjects], dtype=np.int64)
                  for role, subjects in roles.items()} for f, roles in folds.items()} for a, folds in cohort['folds'].items()}
        attrs = self.receipt['attributes']
        self._meta = dict(ids=np.asarray(ids), y=np.asarray([attrs[s]['y'] for s in ids], dtype=np.int64),
                          sites=np.asarray([attrs[s]['site'] for s in ids]),
                          sex=np.asarray([attrs[s]['sex'] for s in ids]), age=np.asarray([attrs[s]['age'] for s in ids]),
                          demographics_raw=np.asarray([[np.nan if x is None else x for x in attrs[s]['demographics_raw']] for s in ids], dtype=np.float32),
                          splits=splits, n=1035, original_cohort=True, original_subject_indices=np.arange(1035),
                          demographics_columns=['SEX', 'AGE_AT_SCAN', 'HANDEDNESS_CATEGORY', 'FIQ'],
                          raw_condition=self.condition, raw_condition_parameters=CONDITIONS[self.condition],
                          source_sha256={str(PLAN): sha(PLAN), str(self.inputs / 'manifest.json'): sha(self.inputs / 'manifest.json')})
        return self._meta

    def _arrays(self, atlas, family, cache):
        need(atlas == 'aal', 'Only declared AAL/W1 inputs are supported')
        if not self._receipt_metadata_verified:
            for item in self.done['subject_receipts']:
                path = output(self.inputs / 'subject_checkpoints' / (item['subject_id'] + '.json'))
                need(str(path) == item['path'] and sha(path) == item['sha256'], 'Subject receipt metadata changed')
                note = read(path)
                need(note['status'] == 'complete' and note['npz_sha256'] == item['npz_sha256'] and
                     note['recipe'] == recipe(self.receipt, item['subject_id'], self.condition),
                     'Subject receipt source/condition changed')
            self._receipt_metadata_verified = True
        if atlas not in cache:
            np = old().np
            out = {}
            keys = BASE_KEYS if family == 'report_base' else NATIVE_KEYS
            need(set(self.done['arrays'][family]) == set(keys), 'Wrong aggregate field set')
            for key, desc in self.done['arrays'][family].items():
                path = self.inputs / family / atlas / (key + '.npy')
                need(str(path) == desc['path'] and sha(path) == desc['sha256'], 'Aggregate array bytes/path changed')
                a = np.load(path, mmap_mode='r', allow_pickle=False)
                need(list(a.shape) == desc['shape'] and a.dtype.str == desc['dtype'] and a.shape[0] == 1035,
                     'Malformed aggregate array: ' + key)
                out[key] = a
            cache[atlas] = out
        return cache[atlas]

    def base_arrays(self, atlas):
        return self._arrays(atlas, 'report_base', self._base)

    def native_arrays(self, atlas):
        return self._arrays(atlas, 'native_1035', self._native)

    def _gate(self, atlas, window, fold):
        need(atlas == 'aal' and int(window) == 1, 'Only AAL/W1 group gating is declared')
        return old().modules()['original'].ReportData._gate(self, atlas, int(window), str(fold))

    def candidates(self, atlas, method, window, fold):
        need(atlas == 'aal' and int(window) == 1 and method in METHODS, 'Unknown native candidate request')
        return old().modules()['original'].ReportData.candidates(self, atlas, method, int(window), str(fold))




