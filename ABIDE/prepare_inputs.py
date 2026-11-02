"""Read-only preflight for an existing ABIDE derived-input package.

Run from any directory with an explicit --data-workspace. Paths beneath that
directory are contract keys, not instructions to find other experiment folders.
This program does not download, copy, link, regenerate, or edit any input.
It preserves the execution readers' exact historical recipe/path/hash checks.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ATLAS_ROIS = dict(aal=116, cc200=200, ez=116)
SNAPSHOTS = Path('第四部分/按论文报告复现_20261027/inputs')
RAW = Path('第四部分/inputs/raw_timeseries')
COHORT = Path('第四部分/ABIDE后续分类器队列_20261027/inputs/original_cohort_and_folds.json')
SOURCE_PATHS = (Path('第三部分最终版/code/estimators.py'), Path('第四部分/code/prepare_raw.py'))

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def identity(value):
    raw = json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): value.update(block)
    return value.hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def recipe_workspace_roots(keys, relative_sources):
    """Infer diagnostics from known suffixes only; never alter a recipe key."""
    roots, unmatched = set(), 0
    for key in keys:
        windows = PureWindowsPath(key)
        cls = PureWindowsPath if windows.is_absolute() else PurePosixPath
        path = cls(key)
        if not path.is_absolute() or '..' in path.parts:
            unmatched += 1; continue
        matches = []
        for relative in relative_sources:
            suffix = cls(Path(relative).as_posix()).parts
            actual = path.parts[-len(suffix):]
            equal = tuple(p.casefold() for p in actual)==tuple(p.casefold() for p in suffix) if cls is PureWindowsPath else actual==suffix
            if equal: matches.append(str(path.parents[len(suffix)-1]))
        if len(matches)==1: roots.add(matches[0])
        else: unmatched += 1
    return dict(declared_workspace_roots=sorted(roots),unmatched_source_keys=unmatched,
                unique_declared_root=next(iter(roots)) if len(roots)==1 and unmatched==0 else None)

def check(data_workspace, *, scope='main', hashes=False, identities=False, details=False):
    integrity = load('_abide_input_integrity', BASE/'tools/integrity.py')
    integrity.check()
    data = Path(data_workspace).expanduser().resolve()
    contract = read(BASE/'configs/main/input_contract.json')
    selected = read(BASE/'configs/supplements/selected_tasks.json')
    required, errors = {}, []
    def need(ok, message):
        if not ok: errors.append(message)
    def add(relative, role, expected=None):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Input contract path must be workspace-relative')
        key = relative.as_posix()
        if key not in required: required[key] = dict(path=data/relative, role=role, sha256=expected)
        elif expected:
            need(required[key]['sha256'] in (None, expected), 'Conflicting input SHA256: '+key)
            required[key]['sha256'] = expected
        return data/relative
    for row in contract['requirements']:
        add(row['relative_to_workspace'], row['role'], row['expected_sha256'])
    # These bindings also pin HDF5/phenotype and the AAL snapshot manifests.
    for row in selected['original_input_bindings']:
        add(row['relative_workspace'], 'fixed_input_binding', row['sha256'])
    common_id_hash = None
    report_source = BASE/'code/main/source_tree/freeenergy/第四部分/按论文报告复现_20261027/code/report_data.py'
    text = report_source.read_text(encoding='utf-8-sig')
    nodes = [n for c in ast.parse(text).body if isinstance(c, ast.ClassDef) and c.name=='ReportData'
             for n in c.body if isinstance(n, ast.FunctionDef) and n.name=='_compute_missing_native']
    node, = nodes
    completion_sha = hashlib.sha256(''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno]).encode('utf-8')).hexdigest()
    metadata_count = 0
    path_bindings = []
    for atlas, rois in ATLAS_ROIS.items():
        for family in ('report_base', 'native_1035'):
            relative = SNAPSHOTS/family/atlas/'manifest.json'
            path = add(relative, 'snapshot_recipe_manifest')
            if not path.is_file(): continue
            doc = read(path); recipe = doc['recipe']; metadata_count += 1
            need(doc['status']=='complete' and recipe['atlas']==atlas, 'Snapshot status/atlas differs: '+str(relative))
            if common_id_hash is None: common_id_hash = recipe['subject_ids_sha256']
            need(recipe['subject_ids_sha256']==common_id_hash, 'Snapshot cohort hashes differ: '+str(relative))
            arrays = doc['arrays']
            need(len(arrays)==(11 if family=='report_base' else 16), 'Snapshot array roster differs: '+str(relative))
            for name, item in arrays.items():
                need(Path(name).name==name and '/' not in name and '\\' not in name, 'Invalid snapshot key')
                expected_shape = [1035, rois*(rois-1)//2]
                if name.startswith(('te_', 'var_', 'phiid_')): expected_shape += [2]
                need(item['shape']==expected_shape, 'Snapshot shape differs: '+str(relative.parent/name))
                add(relative.parent/(name+'.npy'), 'snapshot_array', item['sha256'])
            if family=='report_base':
                need(recipe['version']=='original-1035-lower-base-v1' and recipe['n_rois']==rois, 'Base recipe differs: '+atlas)
                add('dataset/abide_data/04_rgc_intermediate_data/rgc_base.hdf5', 'base_hdf5', recipe['base_sha256'])
            else:
                expected_runtime = {key:contract['exact_runtime'][key] for key in ('python','numpy','scipy','sklearn','h5py')}
                need(recipe['version']=='native-1035-constant-edge-zero-v1', 'Native recipe version differs: '+atlas)
                need(recipe['runtime_versions']==expected_runtime, 'Native recipe runtime differs: '+atlas)
                need(recipe['completion_implementation_sha256']==completion_sha, 'Native completion body differs: '+atlas)
                add(Path('第四部分/inputs')/(atlas+'_primitives.npz'), 'native_npz', recipe['original_npz_sha256'])
                validation = add(Path('第四部分/inputs')/(atlas+'_validated.json'), 'native_validation', recipe['validation_sha256'])
                if validation.is_file():
                    value = read(validation)
                    need(value['status']=='ok' and value['npz_sha256']==recipe['original_npz_sha256'], 'Native NPZ/validation binding differs: '+atlas)
                sources = (*SOURCE_PATHS, RAW/(atlas+'_source_manifest.json'), RAW/'header_tr_manifest.json')
                # Match the execution reader's literal path keys; never rebase.
                expected_keys = {str(data/p) for p in sources}
                diagnostic = recipe_workspace_roots(recipe['code_source_sha256'], sources)
                path_bindings.append(dict(atlas=atlas,current_resolved_workspace=str(data),
                    literal_keys_match=set(recipe['code_source_sha256'])==expected_keys,**diagnostic))
                need(set(recipe['code_source_sha256'])==expected_keys,
                     'Native recipe absolute path identity differs for '+atlas+
                     '; current resolved workspace='+str(data)+
                     '; manifest-declared workspace roots='+json.dumps(diagnostic['declared_workspace_roots'],ensure_ascii=False)+
                     '; unrecognized source keys='+str(diagnostic['unmatched_source_keys'])+
                     '. This checker does not rewrite recipe keys; the resolved workspace must match the declared root.')
                for relative_source in sources:
                    expected = recipe['code_source_sha256'].get(str(data/relative_source))
                    add(relative_source, 'recipe_source_binding', expected)
    raw_count = 0
    cohort_doc = None
    if scope=='supplements':
        cohort = add(COHORT, 'raw_cohort_metadata')
        source = add(RAW/'aal_source_manifest.json', 'raw_series_metadata')
        tr = add(RAW/'header_tr_manifest.json', 'raw_TR_metadata')
        if all(p.is_file() for p in (cohort,source,tr)):
            cohort_doc = read(cohort)
            ids = cohort_doc['ids']; rows = read(source)['files']; timing = read(tr)['subjects']
            index = {row['subject_id']:row for row in rows}
            need(len(ids)==len(set(ids))==1035 and len(rows)==len(index)==1035 and set(ids)==set(index), 'Raw cohort/source identity roster differs')
            need(identity(sorted(ids))==selected['cohorts']['full']['sorted_ids_sha256'], 'Raw cohort differs from fixed participant contract')
            main_rows = read(BASE/'configs/reproduction/selected_configurations.json')['rows']
            for atlas in ATLAS_ROIS:
                for fold in range(10):
                    roles = cohort_doc['folds'][atlas][str(fold)]
                    need(set(roles)=={'train','valid','test'}, 'Raw cohort role keys differ')
                    flat = [sid for role in ('train','valid','test') for sid in roles[role]]
                    need(len(flat)==len(set(flat))==1035 and set(flat)==set(ids), 'Raw cohort roles overlap or omit participants')
                    need(identity(roles['test'])==selected['expected_oof_axes'][f'full/{fold}']['canonical_sha256']['ids_test'], 'Raw test identity/order differs')
                    expected_counts = [r['roles'] for r in main_rows if r['atlas']==atlas and r['fold']==fold]
                    need(all({key:len(value) for key,value in roles.items()}==counts for counts in expected_counts), 'Raw role counts differ from fixed configurations')
            need(set(ids)<=set(timing), 'Raw TR metadata does not cover the cohort')
            for sid in ids:
                need(Path(sid).name==sid and '/' not in sid and '\\' not in sid, 'Invalid raw subject path key')
                row = index[sid]
                need(int(row['n_rois'])==116 and int(row['n_timepoints'])>=48, 'Raw source geometry differs')
                seconds = float(row['tr_seconds'])
                need(math.isfinite(seconds) and seconds>0, 'Invalid source TR')
                add(RAW/'aal'/(sid+'_rois_aal.1D'), 'raw_ROI_timeseries', row['sha256']); raw_count += 1
    present = hash_checked = payload_hash_skipped = 0
    for relative, row in required.items():
        path = row['path']
        if not path.is_file(): errors.append('Missing input: '+relative); continue
        present += 1
        if row['sha256']:
            if hashes or path.suffix.lower() in ('.json','.py','.csv'):
                need(sha(path)==row['sha256'], 'Input SHA256 differs: '+relative); hash_checked += 1
            else: payload_hash_skipped += 1
    identity_result = None
    if identities and not errors:
        identity_result = load('_abide_input_identities', BASE/'tools/check_identities.py').check(data)
        if cohort_doc is not None:
            import h5py
            with h5py.File(data/'dataset/abide_data/03_pearson_fc/abide.hdf5','r') as handle:
                for atlas in ATLAS_ROIS:
                    for fold in range(10):
                        for role in ('train','valid','test'):
                            original = [v.decode() if isinstance(v,bytes) else str(v) for v in handle[f'experiments/{atlas}_whole/{fold}/{role}'][:]]
                            need(cohort_doc['folds'][atlas][str(fold)][role]==original, 'Raw cohort role/order differs from original HDF5')
            identity_result['raw_cohort_role_arrays_checked'] = 90
    result = dict(status='FAIL' if errors else 'PASS_FILE_HASHES' if hashes else 'PASS_PATHS_AND_METADATA',
        scope=scope, data_workspace=str(data), contract=str(BASE/'configs/main/input_contract.json'),
        contract_files=len(contract['requirements']),required_files=len(required),present_files=present,
        snapshot_manifests_checked=metadata_count,raw_ROI_files=raw_count,SHA256_files_checked=hash_checked,
        large_payload_SHA256_not_checked=payload_hash_skipped,identity_check=identity_result,
        recipe_path_bindings=path_bindings,
        errors=errors,inputs_modified=False,links_or_copies_created=False,scientific_training_performed=False,
        source_data='https://fcon_1000.projects.nitrc.org/indi/abide/preprocessed.html',
        derived_input_package='Supply the existing recipe-bound derived-input package; this command does not download or reconstruct it.')
    if details:
        result['files'] = [dict(relative_to_workspace=relative,role=row['role'],expected_sha256=row['sha256']) for relative,row in required.items()]
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-workspace',type=Path,required=True,help='Root of the existing derived-input package; relative contract keys resolve here')
    parser.add_argument('--scope',choices=('main','supplements'),default='main',help='supplements additionally checks the cohort metadata and 1035 AAL ROI inputs')
    parser.add_argument('--hashes',action='store_true',help='Read and SHA256-check large HDF5/NPZ/NPY/ROI payloads as well as metadata/source files')
    parser.add_argument('--identities',action='store_true',help='Also read fixed HDF5/FD identities using the NumPy/h5py environment; no fitting')
    parser.add_argument('--list',action='store_true',dest='details',help='Print exact relative input keys, roles and required hashes')
    args = parser.parse_args(argv)
    try: result = check(args.data_workspace,scope=args.scope,hashes=args.hashes,identities=args.identities,details=args.details)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        result = dict(status='FAIL',error=str(error),inputs_modified=False,scientific_training_performed=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if result['status']=='FAIL' else 0

if __name__=='__main__':raise SystemExit(main())
