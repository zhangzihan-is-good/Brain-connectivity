"""Report-aligned ABIDE data with training-only selection and fixed 1035 roles.

Scientific inputs under older experiments are read-only. All generated files
live under the supplied new experiment root. Old globally supervised direction
HDF5 files are never opened. Native arrays use lower-triangle *coordinate order*
while retaining the native low-ROI -> high-ROI directional channel convention.
"""
from __future__ import annotations

# Preserve caller-specific encodings/block sizes; serializers stay local.
import sys as _io_sys
from pathlib import Path as _IOPath
_IO_MERGE_ROOT = next(p for p in _IOPath(__file__).resolve().parents if (p / 'io_shared.py').is_file())
if str(_IO_MERGE_ROOT) not in _io_sys.path:
    _io_sys.path.insert(0, str(_IO_MERGE_ROOT))
import io_shared as _io_shared

# Shared RGIC execution: IC commit/profile provenance is verified at entry.
import sys as _rgic_sys
from pathlib import Path as _RGICPath
_RGIC_MERGE_ROOT = next(p for p in _RGICPath(__file__).resolve().parents if (p / 'rgic_shared.py').is_file())
if str(_RGIC_MERGE_ROOT) not in _rgic_sys.path:
    _rgic_sys.path.insert(0, str(_RGIC_MERGE_ROOT))
import rgic_shared as _rgic_shared


import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import warnings
from _paths import CORE

WORKSPACE = Path(__file__).resolve().parents[3]
PART4 = WORKSPACE / "第四部分"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
sys.path.insert(0, str(CORE))
import runtime  # noqa: E402,F401
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

METHODS = ("RGIC", "Lag_peak", "Gaussian_TE", "VAR_Granger", "PSI",
           "PhiID_unique_transfer", "Pearson", "Coherence", "GCMI")
DATA_METHODS = METHODS + ("RGIC_scalar",)
ATLASES = ("aal", "cc200", "ez")
ROI_COUNTS = dict(aal=116, cc200=200, ez=116)
BUDGETS = dict(aal=1000, cc200=3000, ez=1000)
INDEPENDENT = frozenset(("PSI", "Pearson", "Coherence", "GCMI"))
NATIVE_KEYS = ("r0", "W1_rstar", "W1_lag", "W1_tie", "W2_rstar", "W2_lag", "W2_tie",
               "te_h1_d1", "te_h1_d2", "var_p1", "var_p2", "psi", "coherence", "gcmi",
               "phiid_d1", "phiid_d2")
DIRECTIONAL_PAIRS = frozenset(("te_h1_d1", "te_h1_d2", "var_p1", "var_p2", "phiid_d1", "phiid_d2"))
BASE_KEYS = ("r0",) + tuple("W%d_%s" % (w, k) for w in (1, 2)
                            for k in ("gain", "cost", "peak_lag", "peak_rho", "potential"))
_SHA_MEMO = {}
_MODULES = {}
RUNTIME_VERSIONS = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                        sklearn=sklearn.__version__, h5py=h5py.__version__)


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return _io_shared.read_json(path, encoding="utf-8")


def _serialize(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(_serialize(value).encode("utf-8")).hexdigest()


def array_sha(value):
    value = np.asarray(value)
    h = hashlib.sha256(_serialize(dict(shape=list(value.shape), dtype=value.dtype.str)).encode())
    h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def sha(path):
    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_MEMO:
        _SHA_MEMO[key] = _io_shared.sha256_file(path, block_size=4*1024*1024)
    return _SHA_MEMO[key]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".%d.partial" % os.getpid())
    temp.write_bytes((_serialize(value) + "\n").encode("utf-8"))
    os.replace(temp, path)


def _module(name, path):
    key = (name, str(path))
    if key not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, str(path))
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        _MODULES[key] = value
    return _MODULES[key]


def estimator_module():
    import estimators_shared
    return estimators_shared


def preparation_module():
    return _module("report_native_preparation", CORE / "prepare_raw.py")


def lower_to_upper_indices(n_rois):
    """Index the canonical upper vector at each old lower-vector coordinate.

    Old corr_at_lag(high,low,+d) correlates high_late with low_early.
    New upper correlation(+d) correlates low_early with high_late. They are
    identical scalar products, so reorder alone; do not negate native lag.
    Native two-channel scores remain [low -> high, high -> low].
    """
    high, low = np.tril_indices(n_rois, -1)
    return low * (2 * n_rois - low - 1) // 2 + high - low - 1


def _npy_write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".%d.partial" % os.getpid())
    with temp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temp, path)


def _native_dtype(key):
    return np.bool_ if key.endswith("_tie") else np.int8 if key.endswith("_lag") else np.float32


def _base_dtype(key):
    return np.int16 if key.endswith("peak_lag") else np.float32


def group_eta(gain, cost, lag, rho, lambda_cost=1., percentile=95., eps=1e-8):
    return _rgic_shared.abide_group_eta(gain, cost, lag, rho, lambda_cost, percentile, eps)


class ReportData:
    def __init__(self, newroot=None):
        self.root = Path(newroot or Path(__file__).resolve().parents[1]).resolve()
        require(self.root.parent == PART4.resolve(), "New experiment root must be a child of fourth part")
        self.inputs = self.root / "inputs"
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.data = WORKSPACE / "dataset" / "abide_data"
        self.base_h5 = self.data / "04_rgc_intermediate_data" / "rgc_base.hdf5"
        self.pearson_h5 = self.data / "03_pearson_fc" / "abide.hdf5"
        self.raw_root = PART4 / "inputs" / "raw_timeseries"
        self._meta = None
        self._base = {}
        self._native = {}
        self._headers = None
        self._gates = {}

    def metadata(self):
        if self._meta is not None:
            return self._meta
        with h5py.File(self.base_h5, "r") as base, h5py.File(self.pearson_h5, "r") as old:
            ids = sorted(base["patients"])
            require(ids == sorted(old["patients"]) and len(ids) == 1035, "Original cohort must contain exactly 1035 matched IDs")
            index = {sid: i for i, sid in enumerate(ids)}
            y = np.array([base["patients"][s].attrs["y"] for s in ids], dtype=np.int64)
            sites = np.asarray([str(base["patients"][s].attrs["site"]) for s in ids])
            sex = np.asarray([base["patients"][s].attrs["sex"] for s in ids], dtype=np.float64)
            age = np.asarray([base["patients"][s].attrs["age"] for s in ids], dtype=np.float64)
            require(int((y == 0).sum()) == 505 and int((y == 1).sum()) == 530, "Expected original ASD0/TC1 labels")
            require(np.isfinite(sex).all() and np.isfinite(age).all() and np.isin(sex, [1, 2]).all(), "Unexpected phenotype values")
            splits = {}
            for atlas in ATLASES:
                splits[atlas] = {}
                test_count = np.zeros(len(ids), dtype=np.int8)
                for fold in range(10):
                    group = old["experiments"][atlas + "_whole"][str(fold)]
                    roles = {role: np.asarray([index[v.decode() if isinstance(v, bytes) else str(v)]
                                               for v in group[role][:]], dtype=np.int64)
                             for role in ("train", "valid", "test")}
                    flat = np.concatenate(list(roles.values()))
                    require(len(flat) == len(ids) and len(np.unique(flat)) == len(ids), "Fold roles overlap or miss subjects")
                    require(all(len(np.unique(y[ix])) == 2 for ix in roles.values()), "Original fold has a single class")
                    test_count[roles["test"]] += 1
                    splits[atlas][str(fold)] = roles
                require(np.all(test_count == 1), "Test folds must cover each original subject once")
        self._meta = dict(ids=np.asarray(ids), y=y, sites=sites, sex=sex, age=age,
                          splits=splits, label_semantics=dict(ASD=0, TC=1),
                          original_cohort=True, n=1035,
                          source_sha256={str(self.base_h5): sha(self.base_h5), str(self.pearson_h5): sha(self.pearson_h5)})
        return self._meta

    def _load_arrays(self, directory, names, manifest):
        result = {}
        for key in names:
            path = directory / (key + ".npy")
            require(path.is_file(), "Snapshot array missing: " + str(path))
            require(sha(path) == manifest["arrays"][key]["sha256"], "Snapshot array hash mismatch: " + str(path))
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            require(list(a.shape) == manifest["arrays"][key]["shape"], "Snapshot shape mismatch")
            result[key] = a
        return result

    def base_arrays(self, atlas):
        require(atlas in ATLASES, "Unknown atlas")
        if atlas in self._base:
            return self._base[atlas]
        meta = self.metadata()
        directory = self.inputs / "report_base" / atlas
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "manifest.json"
        recipe = dict(version="original-1035-lower-base-v1", base_sha256=sha(self.base_h5),
                      subject_ids_sha256=array_sha(meta["ids"]), atlas=atlas, n_rois=ROI_COUNTS[atlas])
        if receipt.exists():
            manifest = read_json(receipt)
            require(manifest["recipe"] == recipe and manifest["status"] == "complete", "Base cache recipe differs")
            self._base[atlas] = self._load_arrays(directory, BASE_KEYS, manifest)
            return self._base[atlas]
        shape = (1035, ROI_COUNTS[atlas] * (ROI_COUNTS[atlas] - 1) // 2)
        arrays = {k: np.lib.format.open_memmap(directory / (k + ".npy.partial"), mode="w+", dtype=_base_dtype(k), shape=shape)
                  for k in BASE_KEYS}
        with h5py.File(self.base_h5, "r") as handle:
            for i, sid in enumerate(meta["ids"]):
                group = handle["patients"][str(sid)][atlas]
                arrays["r0"][i] = group["pearson"][:]
                for window in (1, 2):
                    for field in ("gain", "cost", "peak_lag", "peak_rho", "potential"):
                        key = "W%d_%s" % (window, field)
                        value = group["w%d" % window][field][:]
                        require(value.shape == shape[1:] and np.isfinite(value).all(), "Nonfinite or malformed original base: " + str(sid))
                        arrays[key][i] = value
                require(np.isfinite(arrays["r0"][i]).all(), "Nonfinite original Pearson")
                if (i + 1) % 200 == 0:
                    print("BASE_CACHE", atlas, i + 1, "/1035", flush=True)
        for key in BASE_KEYS:
            arrays[key].flush()
        arrays.clear()
        import gc
        gc.collect()
        descriptions = {}
        for key in BASE_KEYS:
            temp, path = directory / (key + ".npy.partial"), directory / (key + ".npy")
            os.replace(temp, path)
            descriptions[key] = dict(shape=list(shape), dtype=np.dtype(_base_dtype(key)).str, sha256=sha(path))
        manifest = dict(status="complete", recipe=recipe, arrays=descriptions,
                        edge_order="np.tril_indices(R,-1)", original_global_selected_direction_opened=False)
        write_json(receipt, manifest)
        self._base[atlas] = self._load_arrays(directory, BASE_KEYS, manifest)
        print("BASE_CACHE_COMPLETE", atlas, flush=True)
        return self._base[atlas]

    def _compute_missing_native(self, atlas, sid):
        """Compute only previously excluded subjects, zeroing constant ROI edges.

        Every unexpected nonfinite value on two nonconstant ROIs is a failure.
        Removing a constant ROI only for estimator arithmetic does not alter
        the atlas: complete pair arrays are scattered back with zero edges.
        """
        path = self.raw_root / atlas / (str(sid) + "_rois_" + atlas + ".1D")
        tokens = path.open("r", encoding="ascii").readline().split()
        roi_ids = np.array([int(v[1:]) for v in tokens])
        order = np.argsort(roi_ids)
        x = np.loadtxt(path, dtype=np.float64)[:, order]
        require(x.ndim == 2 and x.shape[1] == ROI_COUNTS[atlas] and x.shape[0] >= 12, "Raw time-series geometry differs")
        require(np.isfinite(x).all(), "Unexpected nonfinite input; only constant-ROI zero-edge policy is allowed: " + str(sid))
        active = np.flatnonzero(np.ptp(x, axis=0) != 0)
        require(len(active) >= 2, "Fewer than two nonconstant ROIs")
        estimators, preparation = estimator_module(), preparation_module()
        y = x[:, active]
        estimators._validate(y)
        out = estimators.correlation_primitives(y)
        prediction, pred_diag = preparation.prediction_required(y)
        out.update(prediction)
        if self._headers is None:
            self._headers = read_json(self.raw_root / "header_tr_manifest.json")
        tr = float(self._headers["subjects"][str(sid)]["tr_seconds"])
        spectral, spectral_diag = preparation.spectral_required(y, tr)
        out.update(spectral)
        out.update(estimators.gcmi_primitives(y))
        phi = estimators.phiid_primitives(y)
        out.update(phi)
        r = ROI_COUNTS[atlas]
        i, j = np.triu_indices(len(active), 1)
        a, b = active[i], active[j]
        upper_index = a * (2 * r - a - 1) // 2 + b - a - 1
        e = r * (r - 1) // 2
        result = {}
        for key in NATIVE_KEYS:
            value = np.asarray(out[key])
            require(np.isfinite(value).all(), "Unexpected nonfinite native estimator on nonconstant ROI pair: %s %s %s" % (atlas, sid, key))
            shape = (e, 2) if key in DIRECTIONAL_PAIRS else (e,)
            dest = np.zeros(shape, dtype=_native_dtype(key))
            dest[upper_index] = value
            result[key] = dest
        return result, dict(subject_id=str(sid), raw_sha256=sha(path), n_rois=r,
                            n_timepoints=len(x), constant_roi_indices=np.setdiff1d(np.arange(r), active).tolist(),
                            constant_edge_count=int(e - len(i)), tr_seconds=tr,
                            prediction=pred_diag, spectral=spectral_diag,
                            phiid_failures=[int(out["phiid_failed_d%d" % d].sum()) for d in (1, 2)])

    def native_arrays(self, atlas):
        require(atlas in ATLASES, "Unknown atlas")
        if atlas in self._native:
            return self._native[atlas]
        meta = self.metadata()
        source = PART4 / "inputs" / (atlas + "_primitives.npz")
        validated = read_json(PART4 / "inputs" / (atlas + "_validated.json"))
        require(validated["status"] == "ok" and validated["npz_sha256"] == sha(source), "Original native primitive snapshot failed its independent binding")
        directory = self.inputs / "native_1035" / atlas
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "manifest.json"
        dependencies = [WORKSPACE / "第三部分最终版" / "code" / "estimators.py", PART4 / "code" / "prepare_raw.py",
                        self.raw_root / (atlas + "_source_manifest.json"), self.raw_root / "header_tr_manifest.json"]
        recipe = dict(version="native-1035-constant-edge-zero-v1", atlas=atlas, original_npz_sha256=sha(source),
                      runtime_versions=RUNTIME_VERSIONS,
                      validation_sha256=sha(PART4 / "inputs" / (atlas + "_validated.json")),
                      subject_ids_sha256=array_sha(meta["ids"]),
                      code_source_sha256={str(p): sha(p) for p in dependencies},
                      completion_implementation_sha256=hashlib.sha256(inspect.getsource(self._compute_missing_native).encode()).hexdigest(),
                      direction_mapping="upper low->high native channels retained; lower coordinate reordering only")
        if receipt.exists():
            manifest = read_json(receipt)
            require(manifest["recipe"] == recipe and manifest["status"] == "complete", "Native cache recipe differs")
            self._native[atlas] = self._load_arrays(directory, NATIVE_KEYS, manifest)
            return self._native[atlas]
        index = {str(s): i for i, s in enumerate(meta["ids"])}
        r = ROI_COUNTS[atlas]
        edges = r * (r - 1) // 2
        arrays = {key: np.lib.format.open_memmap(directory / (key + ".npy.partial"), mode="w+", dtype=_native_dtype(key),
                    shape=(1035, edges, 2) if key in DIRECTIONAL_PAIRS else (1035, edges)) for key in NATIVE_KEYS}
        permutation = lower_to_upper_indices(r)
        with np.load(source, allow_pickle=False) as old:
            old_ids = [str(s) for s in old["subject_ids"]]
            require(len(old_ids) == 939 and len(set(old_ids)) == 939, "Expected original audited 939 native subjects")
            dest = np.array([index[s] for s in old_ids], dtype=np.int64)
            for key in NATIVE_KEYS:
                value = old[key]
                require(np.isfinite(value).all(), "Nonfinite original native snapshot")
                # Bounded row chunks avoid one full extra CC200 pair tensor.
                for lo in range(0, len(dest), 32):
                    arrays[key][dest[lo:lo + 32]] = value[lo:lo + 32, permutation]
                del value
                print("NATIVE_REUSE", atlas, key, flush=True)
        missing = sorted(set(index) - set(old_ids))
        require(len(missing) == 96, "Expected exactly 96 shared-QC exclusions to complete")
        completion = []
        checkpoints = directory / "completion"
        checkpoints.mkdir(exist_ok=True)
        for number, sid in enumerate(missing, 1):
            checkpoint, record = checkpoints / (sid + ".npz"), checkpoints / (sid + ".json")
            raw_path = self.raw_root / atlas / (sid + "_rois_" + atlas + ".1D")
            if checkpoint.exists() and record.exists():
                detail = read_json(record)
                require(detail["recipe_sha256"] == digest(recipe) and detail["raw_sha256"] == sha(raw_path) and detail["npz_sha256"] == sha(checkpoint), "Native completion checkpoint changed")
                with np.load(checkpoint, allow_pickle=False) as handle:
                    values = {k: handle[k] for k in NATIVE_KEYS}
            else:
                values, detail = self._compute_missing_native(atlas, sid)
                temp = checkpoint.with_suffix(".npz.partial")
                with temp.open("wb") as stream:
                    np.savez(stream, **values)
                os.replace(temp, checkpoint)
                detail.update(recipe_sha256=digest(recipe), npz_sha256=sha(checkpoint))
                write_json(record, detail)
            for key, value in values.items():
                require(np.isfinite(value).all(), "Nonfinite completion checkpoint")
                arrays[key][index[sid]] = value[permutation]
            completion.append(detail)
            if number % 8 == 0 or number == len(missing):
                print("NATIVE_COMPLETE_SUBJECT", atlas, number, "/96", flush=True)
        for key in NATIVE_KEYS:
            arrays[key].flush()
        arrays.clear()
        import gc
        gc.collect()
        descriptions = {}
        for key in NATIVE_KEYS:
            temp, path = directory / (key + ".npy.partial"), directory / (key + ".npy")
            os.replace(temp, path)
            shape = (1035, edges, 2) if key in DIRECTIONAL_PAIRS else (1035, edges)
            descriptions[key] = dict(shape=list(shape), dtype=np.dtype(_native_dtype(key)).str, sha256=sha(path))
        manifest = dict(status="complete", recipe=recipe, arrays=descriptions, n_reused=939, n_completed=96,
                        completion_records=completion, edge_order="np.tril_indices(R,-1)",
                        native_direction_semantics="low ROI -> high ROI minus reverse; Lag_peak positive lag same arrow; PSI native sign retained",
                        constant_policy="All edges incident on a temporally constant ROI are zero for every native estimator; all other nonfinite values fail")
        write_json(receipt, manifest)
        self._native[atlas] = self._load_arrays(directory, NATIVE_KEYS, manifest)
        print("NATIVE_CACHE_COMPLETE", atlas, flush=True)
        return self._native[atlas]

    def _gate(self, atlas, window, fold):
        key = (atlas, window, str(fold))
        if key not in self._gates:
            arrays = self.base_arrays(atlas)
            train = self.metadata()["splits"][atlas][str(fold)]["train"]
            prefix = "W%d_" % window
            self._gates[key] = group_eta(*[arrays[prefix + field][train] for field in ("gain", "cost", "peak_lag", "peak_rho")])
        return self._gates[key]

    def candidates(self, atlas, method, window, fold):
        """Raw full-cohort scalar and native-direction blocks; no selection."""
        require(method in DATA_METHODS and window in (1, 2), "Unknown method/window")
        if method in ("RGIC", "RGIC_scalar"):
            p = self.base_arrays(atlas)
            gate = self._gate(atlas, window, fold)
            eta = gate["eta"]
            ic = _rgic_shared.abide_matrix_blend(p["r0"], p["W%d_potential" % window], eta)
            d = _rgic_shared.abide_direction(p["W%d_peak_lag" % window], ic - p["r0"])
            detail = dict(representation="original_shared_group_eta_IC_plus_signed_abs_delta", group_q_eta=gate["q_eta"],
                          eta_sha256=array_sha(eta), eta_mean=float(eta.mean()),
                          eta_train_ids_sha256=array_sha(self.metadata()["ids"][self.metadata()["splits"][atlas][str(fold)]["train"]]),
                          core=dict(window=window, gamma=.2, lambda_cost=1., delta=0., cap=.9, percentile=95., eps=1e-8),
                          tie_policy="Original exact argmax over ordered nonzero lags; no opposite-sign tie suppression")
            if method == "RGIC_scalar":
                detail["representation"] = "original_shared_group_eta_IC_only_fixed_total_K"
            return ic, None if method == "RGIC_scalar" else d, detail
        p = self.native_arrays(atlas)
        w = "W%d" % window
        d = None
        if method == "Pearson":
            ic = p["r0"]
        elif method == "Lag_peak":
            ic = p[w + "_rstar"]
            d = np.where(p[w + "_tie"], 0., np.sign(p[w + "_lag"])) * np.abs(ic)
        elif method == "PSI":
            ic, d = np.abs(p["psi"]), p["psi"]
        elif method == "Coherence":
            ic = np.abs(p["coherence"])
        elif method == "GCMI":
            ic = np.maximum(p["gcmi"], 0.)
        elif method == "VAR_Granger":
            value = p["var_p%d" % window]
            ic, d = np.abs(value).max(-1), value[..., 0] - value[..., 1]
        elif method in ("Gaussian_TE", "PhiID_unique_transfer"):
            prefix = "te_h1" if method == "Gaussian_TE" else "phiid"
            a = p[prefix + "_d1"]
            if window == 1:
                value = a
                magnitude = np.abs(a)
            else:
                b = p[prefix + "_d2"]
                magnitude = np.maximum(np.abs(a), np.abs(b))
                value = np.where(np.abs(a) >= np.abs(b), a, b) if method == "PhiID_unique_transfer" else np.maximum(a, b)
            ic, d = magnitude.max(-1), value[..., 0] - value[..., 1]
        else:
            raise ValueError(method)
        detail = dict(representation="matched_native_strength_plus_native_direction" if d is not None else "native_scalar",
                      effective_window=None if method in INDEPENDENT else window,
                      native_direction_available=d is not None, native_direction_forced_to_RGIC=False,
                      lag_peak_includes_zero=False if method == "Lag_peak" else None,
                      strength_signed=method in ("Pearson", "Lag_peak"),
                      estimator_source_sha256=sha(WORKSPACE / "第三部分最终版" / "code" / "estimators.py"),
                      direction_policy="signed_native; low->high minus high->low; PSI native phase sign retained")
        return np.asarray(ic, dtype=np.float32), None if d is None else np.asarray(d, dtype=np.float32), detail

    def prepare(self, atlas, method, window, fold):
        atlas, method, fold = str(atlas), str(method), str(fold)
        window = int(str(window).lstrip("Ww"))
        require(atlas in ATLASES and method in DATA_METHODS and window in (1, 2) and fold in map(str, range(10)), "Unknown preparation unit")
        meta = self.metadata()
        roles = meta["splits"][atlas][fold]
        effective_window = 0 if method in INDEPENDENT else window
        directory = self.inputs / "prepared" / atlas / method / ("W%d" % effective_window)
        directory.mkdir(parents=True, exist_ok=True)
        npz_path, receipt = directory / (fold + ".npz"), directory / (fold + ".json")
        sources = {str(Path(__file__)): sha(__file__), **meta["source_sha256"]}
        if method in ("RGIC", "RGIC_scalar"):
            self.base_arrays(atlas)
            source_manifest = self.inputs / "report_base" / atlas / "manifest.json"
        else:
            self.native_arrays(atlas)
            source_manifest = self.inputs / "native_1035" / atlas / "manifest.json"
        sources[str(source_manifest)] = sha(source_manifest)
        recipe = dict(version="report-training-only-joint-K-v1", atlas=atlas, method=method,
                      runtime_versions=RUNTIME_VERSIONS,
                      effective_window=effective_window, fold=fold, K=BUDGETS[atlas],
                      roles_sha256={r: array_sha(ix) for r, ix in roles.items()}, source_sha256=sources,
                      labels_sha256=array_sha(meta["y"][roles["train"]]),
                      refit_development=False, selection_scope="train_only", scaler_scope="train_only")
        if receipt.exists() and npz_path.exists():
            stored = read_json(receipt)
            require(stored["recipe"] == recipe and stored["npz_sha256"] == sha(npz_path), "Prepared feature cache changed or code/source differs")
            with np.load(npz_path, allow_pickle=False) as handle:
                result = {key: handle[key] for key in handle.files}
            result["feature_metadata"] = stored["feature_metadata"]
            result["feature_metadata"] = dict(result["feature_metadata"], requested_window=window)
            return result
        ic, d, estimator = self.candidates(atlas, method, window, fold)
        edges = ic.shape[1]
        candidates = ic if d is None else np.concatenate((ic, d), axis=1)
        require(np.isfinite(candidates).all(), "Nonfinite candidate features")
        require(np.array_equal(np.sort(np.concatenate(list(roles.values()))), np.arange(1035)), "Role mismatch")
        train = roles["train"]
        # Match original selector argpartition+descending score order, but fit
        # on train only. Infinity treatment preserves the original convention.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            scores, _ = f_classif(np.asarray(candidates[train], dtype=np.float32), meta["y"][train])
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf)
        k = min(BUDGETS[atlas], candidates.shape[1])
        selected = np.argpartition(scores, -k)[-k:]
        selected = selected[np.argsort(scores[selected])[::-1]]
        raw = {role: np.asarray(candidates[ix][:, selected], dtype=np.float32) for role, ix in roles.items()}
        scaler = StandardScaler()
        scaler.fit(raw["train"])
        result = {}
        for role, ix in roles.items():
            result["X_" + role + "_raw"] = raw[role]
            result["X_" + role] = scaler.transform(raw[role]).astype(np.float32)
            result["demos_" + role] = np.column_stack((meta["sex"][ix], meta["age"][ix]))
            result["y_" + role] = meta["y"][ix]
            result["sites_" + role] = meta["sites"][ix]
            result["ids_" + role] = meta["ids"][ix]
        feature_metadata = dict(recipe_sha256=digest(recipe), atlas=atlas, method=method, fold=fold,
            effective_window=effective_window, n_rois=ROI_COUNTS[atlas], edge_order="np.tril_indices(R,-1)",
            direction_coordinate="positive means low-ROI->high-ROI for lag/TE/VAR/PhiID; PSI native sign retained",
            n_candidate_scalar=edges, n_candidate_direction=0 if d is None else d.shape[1],
            selected_k=k, selected_ic=int((selected < edges).sum()), selected_direction=int((selected >= edges).sum()),
            selected_column_ids=selected.tolist(), selected_edge_ids=(selected % edges).tolist(),
            selected_channel_ids=(selected >= edges).astype(int).tolist(),
            selected_scores=[float(v) if np.isfinite(v) else None for v in scores[selected]],
            scaler=dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), var=scaler.var_.tolist(), n_fit=len(train)),
            estimator=estimator, source_sha256=sources, label_semantics=dict(ASD=0, TC=1),
            roles={role: dict(n=len(ix), subject_ids=meta["ids"][ix].tolist(), indices=ix.tolist(),
                  n_ASD=int((meta["y"][ix] == 0).sum()), n_TC=int((meta["y"][ix] == 1).sum())) for role, ix in roles.items()},
            phenotypes=dict(columns=["SEX", "AGE"], appended_to_classifier_separately=True, imputed=False),
            training_only=True, refit_development=False, old_global_selected_direction_opened=False)
        require(all(np.isfinite(a).all() for k, a in result.items() if k.startswith(("X_", "demos_"))), "Nonfinite prepared features")
        temp = npz_path.with_suffix(".npz.%d.partial" % os.getpid())
        with temp.open("wb") as stream:
            np.savez(stream, **result)
        os.replace(temp, npz_path)
        write_json(receipt, dict(status="complete", recipe=recipe, npz_sha256=sha(npz_path), feature_metadata=feature_metadata))
        result["feature_metadata"] = dict(feature_metadata, requested_window=window)
        return result

    def source_manifest(self):
        return dict(module_sha256=sha(__file__), original_sources=self.metadata()["source_sha256"],
                    methods=list(METHODS), n_original=1035, label_semantics=dict(ASD=0, TC=1),
                    atlases=list(ATLASES), K=BUDGETS, core=dict(gamma=.2, lambda_cost=1., delta=0., cap=.9),
                    supervised_selection="train only", scaler="train only", refit_train_plus_valid=False,
                    native_reused=939, native_completed=96, old_global_direction_is_input=False)


def orientation_self_test():
    """Hand-check conventions on a clean low-leading-high delayed signal."""
    rng = np.random.RandomState(314159)
    low = rng.normal(size=128)
    high = np.concatenate(([0.], low[:-1])) + .01 * rng.normal(size=128)
    x = np.column_stack((low, high))
    old_positive = np.corrcoef(high[1:], low[:-1])[0, 1]
    native = estimator_module().correlation_primitives(x)
    row = dict(old_corr_high_low_plus1=float(old_positive), new_upper_low_high_plus1=float(native["rho"][2, 0]),
               native_W1_lag=int(native["W1_lag"][0]), lag_negation_required=False,
               native_channel_order=["low_to_high", "high_to_low"])
    require(abs(old_positive - row["new_upper_low_high_plus1"]) < 1e-12 and row["native_W1_lag"] == 1, "Lag convention check failed")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--atlas", choices=ATLASES)
    parser.add_argument("--build-base", action="store_true")
    parser.add_argument("--build-native", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--method", choices=DATA_METHODS, default="RGIC")
    parser.add_argument("--window", type=int, choices=(1, 2), default=1)
    parser.add_argument("--fold", type=int, choices=range(10), default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(_serialize(orientation_self_test()), flush=True)
    data = ReportData(args.root)
    meta = data.metadata()
    print("METADATA", len(meta["ids"]), "ASD0", int((meta["y"] == 0).sum()), "TC1", int((meta["y"] == 1).sum()), flush=True)
    for atlas in (ATLASES if args.atlas is None else (args.atlas,)):
        if args.build_base:
            data.base_arrays(atlas)
        if args.build_native:
            data.native_arrays(atlas)
        if args.prepare:
            result = data.prepare(atlas, args.method, args.window, args.fold)
            print("PREPARED", atlas, args.method, args.window, args.fold,
                  result["X_train"].shape, result["X_valid"].shape, result["X_test"].shape,
                  "D", result["feature_metadata"]["selected_direction"], flush=True)


if __name__ == "__main__":
    main()
