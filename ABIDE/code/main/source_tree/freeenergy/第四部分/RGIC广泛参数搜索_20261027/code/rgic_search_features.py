# -*- coding: utf-8 -*-
"""Training-only RGIC feature search; outer labels are never loaded here.

The explicit alpha=0 endpoint loads the completed Pearson feature cache, rather
than relying on approximate formula equivalence.  All other candidates use the
original 1035-person, atlas-specific train/valid/test roles and fixed total K.
The analytical screening score is a ridge-regularized diagonal LDA fitted on
training labels and ranked only by validation AUROC.  It is a computational
screen, not an estimate of final classifier performance.
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
import itertools
import json
import os
from pathlib import Path
import time
import warnings

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import h5py
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

NEW = Path(__file__).resolve().parents[1]
PART4 = NEW.parent
OLD = PART4 / "按论文报告复现_20261027"
WORKSPACE = PART4.parent
ATLASES = ("aal", "cc200", "ez")
BUDGETS = dict(aal=1000, cc200=3000, ez=1000)
GRID = dict(gamma=[0., .05, .1, .2, .4, .8], cap=[.8, .9, .95, .99],
            lambda_cost=[0., .1, .3, 1., 3., 10.], percentile=[75., 95., 99.],
            alpha=[.001, .003, .01, .03, .1, .3, .6, 1.])
REPORT = dict(gamma=.2, cap=.9, lambda_cost=1., percentile=95., alpha=1.,
              readout="joint", beta=1.)
PEARSON = dict(gamma=.2, cap=.9, lambda_cost=1., percentile=95., alpha=0.,
               readout="scalar", beta=0.)
VERSION = "broad-rgic-train-only-v1"
_SHA_CACHE = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return _io_shared.read_json(path, encoding="utf-8")


def serialize(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(serialize(value).encode("utf-8")).hexdigest()


def file_sha(path):
    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = _io_shared.sha256_file(path, block_size=4*1024*1024)
    return _SHA_CACHE[key]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".%d.partial" % os.getpid())
    tmp.write_text(serialize(value) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def normalize_spec(spec):
    out = dict(REPORT, **spec)
    for key in ("gamma", "cap", "lambda_cost", "percentile", "alpha", "beta"):
        out[key] = float(out[key])
    require(0 <= out["alpha"] <= 1 and out["gamma"] >= 0 and 0 < out["cap"] <= 1,
            "Invalid core parameter")
    require(out["lambda_cost"] >= 0 and 0 < out["percentile"] <= 100 and out["beta"] >= 0,
            "Invalid gate/weight")
    require(out["readout"] in ("scalar", "joint"), "Unknown readout")
    if out["alpha"] == 0:
        return dict(PEARSON)
    if out["beta"] == 0 or out["readout"] == "scalar":
        out.update(readout="scalar", beta=0.)
    return out


def candidate_id(spec):
    spec = normalize_spec(spec)
    return "pearson_endpoint" if spec["alpha"] == 0 else "rgic_" + digest(spec)[:16]


def select_columns(scores, k):
    scores = np.nan_to_num(np.asarray(scores), nan=-np.inf,
                           posinf=np.finfo(np.float32).max, neginf=-np.inf)
    k = min(int(k), len(scores))
    ix = np.argpartition(scores, -k)[-k:]
    return ix[np.argsort(scores[ix])[::-1]]


def binary_moments(a, y):
    """Population moments in float64 without a whole float64 matrix copy."""
    out = {}
    for c in (0, 1):
        x = np.asarray(a[y == c], dtype=np.float32)
        n = len(x)
        mu = x.mean(0, dtype=np.float64)
        var = np.maximum(np.einsum("ij,ij->j", x, x, dtype=np.float64) / n - mu * mu, 0.)
        out[c] = dict(n=n, mean=mu, var=var)
    return out


def difference_moments(r, delta, y, rm=None):
    """Sufficient statistics for r+t*delta for any edgewise t."""
    rm = binary_moments(r, y) if rm is None else rm
    dm = binary_moments(delta, y)
    cov = {}
    for c in (0, 1):
        mask = y == c
        cov[c] = (np.einsum("ij,ij->j", r[mask], delta[mask], dtype=np.float64) /
                  rm[c]["n"] - rm[c]["mean"] * dm[c]["mean"])
    return rm, dm, cov


def mixture_stats(moments, t):
    rm, dm, cov = moments
    classes = {}
    for c in (0, 1):
        classes[c] = dict(n=rm[c]["n"], mean=rm[c]["mean"] + t * dm[c]["mean"],
                          var=np.maximum(rm[c]["var"] + 2 * t * cov[c] + t * t * dm[c]["var"], 0.))
    return class_stats(classes)


def class_stats(classes):
    n0, n1 = classes[0]["n"], classes[1]["n"]
    n = n0 + n1
    m0, m1 = classes[0]["mean"], classes[1]["mean"]
    v0, v1 = classes[0]["var"], classes[1]["var"]
    diff = m1 - m0
    within = n0 * v0 + n1 * v1
    between = (float(n0) * n1 / n) * diff * diff
    with np.errstate(divide="ignore", invalid="ignore"):
        f = between * (n - 2.) / within
    f = np.where((within == 0) & (between == 0), -np.inf, f)
    f = np.nan_to_num(f, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf)
    var = (within + between) / n
    mean = (n0 * m0 + n1 * m1) / n
    return dict(f=f, diff=diff, mean=mean, var=var)


def gate_components(gain, cost, lag, rho):
    return _rgic_shared.abide_gate_components(gain, cost, lag, rho)


def make_eta(components, lambda_cost, percentile):
    return _rgic_shared.abide_eta(components, lambda_cost, percentile)


def potential(rho, lag, gamma, cap):
    return _rgic_shared.abide_potential(rho, lag, gamma, cap)


class SearchFeatures:
    def __init__(self, oldroot=OLD, newroot=NEW):
        self.oldroot, self.root = Path(oldroot).resolve(), Path(newroot).resolve()
        require(self.root != self.oldroot, "New search must not overwrite frozen prior experiment")
        self.base_h5 = WORKSPACE / "dataset" / "abide_data" / "04_rgc_intermediate_data" / "rgc_base.hdf5"
        self.split_h5 = WORKSPACE / "dataset" / "abide_data" / "03_pearson_fc" / "abide.hdf5"
        with h5py.File(self.base_h5, "r") as handle:
            self.ids = np.asarray(sorted(handle["patients"]))
        require(len(self.ids) == 1035, "Original cohort must have 1035 subjects")
        self.index = {sid: i for i, sid in enumerate(self.ids)}
        self._roles, self._arrays, self._gates, self._source, self._phenotypes = {}, {}, {}, {}, {}
        self.original_sources = {str(p): file_sha(p) for p in (self.base_h5, self.split_h5)}

    def roles(self, atlas, fold):
        key = (atlas, int(fold))
        if key not in self._roles:
            with h5py.File(self.split_h5, "r") as handle:
                g = handle["experiments"][atlas + "_whole"][str(fold)]
                role_ids = {role: np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                                              for x in g[role][:]]) for role in ("train", "valid", "test")}
            self._roles[key] = {r: np.asarray([self.index[s] for s in ids], dtype=np.int64)
                                for r, ids in role_ids.items()}
        return self._roles[key]

    def phenotype(self, indices, labels=False):
        cache_key = (tuple(int(i) for i in indices), bool(labels))
        if cache_key in self._phenotypes:
            return self._phenotypes[cache_key]
        ids = self.ids[indices]
        with h5py.File(self.base_h5, "r") as handle:
            g = handle["patients"]
            demos = np.asarray([[g[s].attrs["sex"], g[s].attrs["age"]] for s in ids], dtype=np.float64)
            sites = np.asarray([str(g[s].attrs["site"]) for s in ids])
            y = np.asarray([g[s].attrs["y"] for s in ids], dtype=np.int64) if labels else None
        result = dict(ids=ids, demos=demos, sites=sites, y=y)
        self._phenotypes[cache_key] = result
        return result

    def arrays(self, atlas):
        require(atlas in ATLASES, "Unknown atlas")
        if atlas not in self._arrays:
            directory = self.oldroot / "inputs" / "report_base" / atlas
            receipt = directory / "manifest.json"
            manifest = read_json(receipt)
            require(manifest["status"] == "complete", "Incomplete source base snapshot")
            self._arrays[atlas] = {}
            for k, description in manifest["arrays"].items():
                path = directory / (k + ".npy")
                require(file_sha(path) == description["sha256"], "Base input array changed: " + str(path))
                self._arrays[atlas][k] = np.load(path, mmap_mode="r", allow_pickle=False)
                require(list(self._arrays[atlas][k].shape) == description["shape"], "Base input shape changed")
            self._source[atlas] = {str(receipt): file_sha(receipt), **self.original_sources}
        return self._arrays[atlas]

    def gate(self, atlas, window, fold, spec):
        key = (atlas, int(window), int(fold))
        if key not in self._gates:
            a, ix = self.arrays(atlas), self.roles(atlas, fold)["train"]
            prefix = "W%d_" % window
            self._gates[key] = gate_components(*[a[prefix + k][ix] for k in ("gain", "cost", "peak_lag", "peak_rho")])
        return make_eta(self._gates[key], spec["lambda_cost"], spec["percentile"])

    def _potential(self, arrays, window, ix, spec):
        prefix = "W%d_" % window
        if spec["gamma"] == .2 and spec["cap"] == .9:
            return np.asarray(arrays[prefix + "potential"][ix], dtype=np.float32)
        return potential(arrays[prefix + "peak_rho"][ix], arrays[prefix + "peak_lag"][ix], spec["gamma"], spec["cap"])

    def _cached_feature(self, atlas, fold, include_test, method, window, spec):
        directory = self.oldroot / "inputs" / "prepared" / atlas / method / ("W%d" % window)
        archive, receipt = directory / (str(fold) + ".npz"), directory / (str(fold) + ".json")
        detail = read_json(receipt)
        require(detail["recipe"]["selection_scope"] == "train_only", "Cached endpoint is not training-only")
        require(file_sha(archive) == detail["npz_sha256"], "Prepared endpoint archive changed")
        for path, value in self.original_sources.items():
            require(detail["recipe"]["source_sha256"].get(path) == value, "Prepared endpoint original source changed")
        roles = ("train", "valid", "test") if include_test else ("train", "valid")
        result = {}
        with np.load(archive, allow_pickle=False) as handle:
            for role in roles:
                for key in ("X_", "X_", "demos_", "sites_", "ids_"):
                    suffix = "_raw" if key == "X_" and "X_" + role in result else ""
                    name = key + role + suffix
                    result[name] = handle[name]
                if role != "test":
                    result["y_" + role] = handle["y_" + role]
        k = result["X_train"].shape[1]
        mask = np.asarray(detail["feature_metadata"]["selected_channel_ids"], dtype=bool)
        require(len(mask) == k, "Cached selected channel IDs disagree")
        for role in roles:
            result["X_" + role][:, mask] *= spec["beta"]
        result.update(direction_mask=mask, raw_beta_applied=False)
        result["feature_metadata"] = dict(detail["feature_metadata"], search_version=VERSION,
            spec=spec, candidate_id=candidate_id(spec), beta=spec["beta"],
            raw_beta_applied=False, post_standardization_direction_weight=True,
            pearson_endpoint_exact_cache=method == "Pearson", report_endpoint_exact_cache=method.startswith("RGIC"),
            test_labels_loaded=False,
            endpoint_cache_sha256=file_sha(archive), endpoint_receipt_sha256=file_sha(receipt))
        return result

    def _endpoint(self, atlas, fold, include_test):
        return self._cached_feature(atlas, fold, include_test, "Pearson", 0, dict(PEARSON))

    def prepare(self, atlas, window, fold, spec, include_test=False):
        """Return selected connection blocks, labels only for train and valid.

        X_*_raw is NEVER beta weighted. X_* is training-standardized connection
        data, then direction columns are multiplied by beta. A classifier that
        performs another scaler must instead scale raw+demos and apply beta to
        direction_mask after that scaler. include_test adds only X/site/ID/demo.
        """
        spec = normalize_spec(spec)
        if spec["alpha"] == 0:
            return self._endpoint(atlas, fold, include_test)
        if all(spec[k] == REPORT[k] for k in ("gamma", "cap", "lambda_cost", "percentile", "alpha")):
            method = "RGIC" if spec["readout"] == "joint" else "RGIC_scalar"
            return self._cached_feature(atlas, fold, include_test, method, int(window), spec)
        require(int(window) in (1, 2), "Unknown window")
        roles = self.roles(atlas, fold)
        use_roles = ("train", "valid", "test") if include_test else ("train", "valid")
        a = self.arrays(atlas)
        eta = self.gate(atlas, window, fold, spec)
        t = np.asarray(spec["alpha"] * eta, dtype=np.float32)
        raw, pheno = {}, {}
        for role in use_roles:
            ix = roles[role]
            r = np.asarray(a["r0"][ix], dtype=np.float32)
            p = self._potential(a, window, ix, spec)
            ic = _rgic_shared.abide_matrix_blend(r, p, t)
            if spec["readout"] == "joint":
                d = _rgic_shared.abide_direction(a["W%d_peak_lag" % window][ix], ic - r)
                raw[role] = np.concatenate((ic, d), axis=1)
            else:
                raw[role] = ic
            pheno[role] = self.phenotype(ix, labels=role != "test")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, _ = f_classif(raw["train"], pheno["train"]["y"])
        selected = select_columns(f, BUDGETS[atlas])
        edges = a["r0"].shape[1]
        direction_mask = selected >= edges
        selected_raw = {role: value[:, selected] for role, value in raw.items()}
        scaler = StandardScaler().fit(selected_raw["train"])
        result = dict(direction_mask=direction_mask, raw_beta_applied=False)
        for role in use_roles:
            result["X_" + role + "_raw"] = selected_raw[role]
            z = scaler.transform(selected_raw[role]).astype(np.float32)
            z[:, direction_mask] *= spec["beta"]
            result["X_" + role] = z
            for key in ("demos", "sites", "ids"):
                result[key + "_" + role] = pheno[role][key]
            if role != "test":
                result["y_" + role] = pheno[role]["y"]
        result["feature_metadata"] = dict(search_version=VERSION, spec=spec, candidate_id=candidate_id(spec),
            atlas=atlas, window=int(window), fold=int(fold), selected_k=len(selected),
            selected_column_ids=selected.tolist(), selected_edge_ids=(selected % edges).tolist(),
            selected_channel_ids=direction_mask.astype(int).tolist(), selected_ic=int((~direction_mask).sum()),
            selected_direction=int(direction_mask.sum()), beta=spec["beta"],
            raw_beta_applied=False, post_standardization_direction_weight=True,
            pearson_endpoint_exact_cache=False, selection_scope="train_only", scaler_scope="train_only",
            scalar_reselection_when_beta_zero=True, test_labels_loaded=False,
            scaler=dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), var=scaler.var_.tolist()),
            source_manifest_sha256=self._source[atlas], eta_mean=float(eta.mean()),
            train_ids_sha256=digest(pheno["train"]["ids"].tolist()),
            numeric_selection="float32 sklearn f_classif, original argpartition+descending-score order")
        return result

    def transductive_view(self, atlas, window, fold, spec):
        return self.prepare(atlas, window, fold, spec, include_test=True)

    def screen(self, atlas, window, fold, shortlist_size=37, grid=None, force=False):
        """Broad scalar/joint screening, six diverse cores, fixed beta expansion."""
        start = time.time()
        grid = GRID if grid is None else grid
        recipe = dict(version=VERSION, module_sha256=file_sha(__file__), grid=grid,
            atlas=atlas, window=int(window), fold=int(fold), k=BUDGETS[atlas], shortlist_size=shortlist_size,
            proxy="diagonal_LDA_ridge_1_on_training_standardized_features", proxy_demos=True,
            proxy_joint_beta=1., final_core_count=6, beta_expansion=[.03, .1, .3, 1., 3.],
            beta_selected_by_proxy=False,
            feature_scope="train_only", ranking_scope="valid_AUROC", outer_labels_loaded=False)
        outpath = self.root / "selection" / "proxy" / atlas / ("W%d" % window) / (str(fold) + ".json")
        if outpath.exists() and not force:
            old = read_json(outpath)
            require(old["recipe"] == recipe and old["status"] == "complete", "Proxy cache recipe mismatch")
            return old
        arrays, roles = self.arrays(atlas), self.roles(atlas, fold)
        tr, va = roles["train"], roles["valid"]
        train, valid = self.phenotype(tr, labels=True), self.phenotype(va, labels=True)
        ytr, yva = train["y"], valid["y"]
        rtr, rva = np.asarray(arrays["r0"][tr]), np.asarray(arrays["r0"][va])
        lagtr = np.asarray(arrays["W%d_peak_lag" % window][tr])
        lagva = np.asarray(arrays["W%d_peak_lag" % window][va])
        rm = binary_moments(rtr, ytr)
        demo_stats = class_stats(binary_moments(train["demos"].astype(np.float32), ytr))
        demo_coef = np.divide(demo_stats["diff"], demo_stats["var"],
                              out=np.zeros_like(demo_stats["diff"]), where=demo_stats["var"] > 0)
        demo_score = .5 * valid["demos"].dot(demo_coef)
        etas = {(l, q): self.gate(atlas, window, fold, dict(lambda_cost=l, percentile=q))
                for l, q in itertools.product(grid["lambda_cost"], grid["percentile"])}
        endpoint = self._endpoint(atlas, fold, False)
        e_stats = class_stats(binary_moments(endpoint["X_train_raw"], ytr))
        ecoef = np.divide(e_stats["diff"], e_stats["var"], out=np.zeros_like(e_stats["diff"]), where=e_stats["var"] > 0)
        endpoint_auc = float(roc_auc_score(yva, .5 * endpoint["X_valid_raw"].dot(ecoef) + demo_score))
        rows = [dict(candidate_id="pearson_endpoint", spec=dict(PEARSON), proxy_auc=endpoint_auc,
                     forced_pearson=True, forced_report=False)]
        # The table records all broad candidates, rather than only the winners.
        # Per-potential sufficient statistics avoid repeated N x E F-score work.
        def evaluate(spec, stats, dstats, delta_valid, direction_valid):
            t = np.asarray(spec["alpha"] * etas[(spec["lambda_cost"], spec["percentile"])], dtype=np.float32)
            st = mixture_stats(stats, t)
            active = t > 0
            ds = dict(dstats, f=np.where(active, dstats["f"], -np.inf))
            fs = st["f"] if spec["readout"] == "scalar" else np.concatenate((st["f"], ds["f"]))
            ix = select_columns(fs, BUDGETS[atlas])
            e = len(t)
            si, di = ix[ix < e], ix[ix >= e] - e
            coef = np.divide(st["diff"][si], st["var"][si], out=np.zeros(len(si)), where=st["var"][si] > 0)
            score = .5 * (np.einsum("ij,j->i", rva[:, si], coef) +
                            np.einsum("ij,j->i", delta_valid[:, si], coef * t[si])) + demo_score
            if len(di):
                dcoef = np.divide(ds["diff"][di], ds["var"][di], out=np.zeros(len(di)), where=ds["var"][di] > 0)
                factor = spec["beta"] ** 2 / (1. + spec["beta"] ** 2)
                score += factor * np.einsum("ij,j->i", direction_valid[:, di], dcoef)
            auc = float(roc_auc_score(yva, score))
            return dict(candidate_id=candidate_id(spec), spec=normalize_spec(spec), proxy_auc=auc,
                selected_ic=int(len(si)), selected_direction=int(len(di)),
                forced_pearson=False, forced_report=normalize_spec(spec) == normalize_spec(REPORT))

        for gamma, cap in itertools.product(grid["gamma"], grid["cap"]):
            p_spec = dict(gamma=gamma, cap=cap)
            ptr = self._potential(arrays, window, tr, p_spec)
            pva = self._potential(arrays, window, va, p_spec)
            dtr, dva = (ptr - rtr).astype(np.float32), (pva - rva).astype(np.float32)
            moments = difference_moments(rtr, dtr, ytr, rm)
            direction_train = _rgic_shared.abide_direction(lagtr, dtr)
            direction_valid = _rgic_shared.abide_direction(lagva, dva)
            dstats = class_stats(binary_moments(direction_train, ytr))
            block = []
            for l, q, alpha in itertools.product(grid["lambda_cost"], grid["percentile"], grid["alpha"]):
                if alpha == 0:
                    continue
                for readout in ("scalar", "joint"):
                    spec = dict(gamma=gamma, cap=cap, lambda_cost=l, percentile=q,
                                alpha=alpha, readout=readout, beta=0. if readout == "scalar" else 1.)
                    block.append(evaluate(spec, moments, dstats, dva, direction_valid))
            rows.extend(block)
            print("PROXY_BLOCK", atlas, window, fold, gamma, cap, len(rows),
                  round(time.time() - start, 3), flush=True)
        ranked = sorted({r["candidate_id"]: r for r in rows}.values(),
                        key=lambda r: (-r["proxy_auc"], r["candidate_id"]))
        core_keys = ("gamma", "cap", "lambda_cost", "percentile", "alpha")
        def core_of(spec):
            return {k: spec[k] for k in core_keys}
        def core_id(spec):
            return digest(core_of(spec))
        core_ranked, core_seen = [], set()
        for row in ranked:
            if row["spec"]["alpha"] == 0 or core_id(row["spec"]) in core_seen:
                continue
            core_ranked.append(dict(core=core_of(row["spec"]), proxy_auc=row["proxy_auc"],
                                    best_proxy_readout=row["spec"]["readout"]))
            core_seen.add(core_id(row["spec"]))
        original_core = core_of(REPORT)
        original_row = next((r for r in core_ranked if r["core"] == original_core), None)
        require(original_row is not None, "The broad grid must include the report core")
        cores = [dict(original_row, forced_report_core=True)]
        used_cores, used_alpha = {core_id(REPORT)}, {REPORT["alpha"]}
        # Predeclared diversity: first require a different alpha and potential
        # pair, then relax potential diversity, finally alpha if necessary.
        used_potentials = {(REPORT["gamma"], REPORT["cap"])}
        for diversity in (2, 1, 0):
            for row in core_ranked:
                c = row["core"]
                cid = digest(c)
                p = (c["gamma"], c["cap"])
                if cid in used_cores or (diversity >= 1 and c["alpha"] in used_alpha) or (diversity >= 2 and p in used_potentials):
                    continue
                if len(cores) >= 6:
                    break
                cores.append(dict(row, forced_report_core=False))
                used_cores.add(cid)
                used_alpha.add(c["alpha"])
                used_potentials.add(p)
        require(len(cores) == 6, "Could not select six distinct cores")
        shortlist = [rows[0]]
        # All five beta values survive regardless of proxy performance.  The
        # actual fixed classifier chooses among these 37 by validation AUROC.
        for row in cores:
            c = row["core"]
            ptr = self._potential(arrays, window, tr, c)
            pva = self._potential(arrays, window, va, c)
            dtr, dva = (ptr - rtr).astype(np.float32), (pva - rva).astype(np.float32)
            moments = difference_moments(rtr, dtr, ytr, rm)
            direction_train = _rgic_shared.abide_direction(lagtr, dtr)
            direction_valid = _rgic_shared.abide_direction(lagva, dva)
            dstats = class_stats(binary_moments(direction_train, ytr))
            variants = [dict(c, readout="scalar", beta=0.)]
            variants += [dict(c, readout="joint", beta=b) for b in (.03, .1, .3, 1., 3.)]
            for spec in variants:
                candidate = evaluate(spec, moments, dstats, dva, direction_valid)
                candidate["forced_report_core"] = row["forced_report_core"]
                shortlist.append(candidate)
        require(len(shortlist) == shortlist_size == 37, "Formal shortlist must contain exactly 37 candidates")
        result = dict(status="complete", recipe=recipe, candidate_count=len(ranked),
            shortlist=shortlist, core_shortlist=cores, candidates=ranked, seconds=time.time() - start,
            source_manifest_sha256=self._source[atlas], n_train=len(tr), n_valid=len(va),
            train_ids=train["ids"].tolist(), valid_ids=valid["ids"].tolist(),
            labels_semantics=dict(ASD=0, TC=1), test_labels_loaded=False,
            proxy_f_precision="float64 class moments of original float32 primitives; finalists recompute actual float32 F",
            core_count_including_unique_pearson=1 + len(grid["gamma"]) * len(grid["cap"]) * len(grid["lambda_cost"]) * len(grid["percentile"]) * sum(a > 0 for a in grid["alpha"]),
            selection_note="Proxy ranks candidates using validation only; actual classifier validation scores determine final choice; outer labels remain unavailable")
        write_json(outpath, result)
        print("PROXY_DONE", str(outpath), len(ranked), round(result["seconds"], 3), flush=True)
        return result


def self_test(root=NEW):
    rng = np.random.RandomState(318)
    y = np.repeat([0, 1], [37, 41])
    r = rng.normal(size=(78, 137)).astype(np.float32)
    d = rng.normal(size=r.shape).astype(np.float32)
    t = rng.uniform(size=r.shape[1]).astype(np.float64)
    st = mixture_stats(difference_moments(r, d, y), t)
    x = r.astype(np.float64) + t * d.astype(np.float64)
    ref, _ = f_classif(x, y)
    f_error = float(np.max(np.abs(st["f"] - ref)))
    require(f_error < 1e-10, "Analytical F mismatch")
    require(normalize_spec(dict(REPORT, beta=0))["readout"] == "scalar", "beta0 must reselect scalar")
    data = SearchFeatures(newroot=root)
    e = data.prepare("aal", 1, 0, PEARSON)
    cached = OLD / "inputs" / "prepared" / "aal" / "Pearson" / "W0" / "0.npz"
    with np.load(cached, allow_pickle=False) as handle:
        for name, value in e.items():
            if isinstance(value, np.ndarray) and name != "direction_mask":
                require(np.array_equal(value, handle[name]), "Pearson endpoint differs: " + name)
    require("X_test" not in e and "y_test" not in e, "Outer data exposed in train-valid API")
    j1 = data.prepare("aal", 1, 0, REPORT)
    j3 = data.prepare("aal", 1, 0, dict(REPORT, beta=.3))
    mask = j1["direction_mask"]
    require(mask.any(), "QA fixture has no selected direction")
    for role in ("train", "valid"):
        require(np.array_equal(j1["X_" + role + "_raw"], j3["X_" + role + "_raw"]), "Beta modified raw values")
        require(np.array_equal(j3["X_" + role][:, mask], j1["X_" + role][:, mask] * np.float32(.3)), "Beta not final operation")
        require(np.array_equal(j3["X_" + role][:, ~mask], j1["X_" + role][:, ~mask]), "Beta modified IC")
    s = data.prepare("aal", 1, 0, dict(REPORT, beta=0))
    require(not s["direction_mask"].any() and s["X_train"].shape[1] == BUDGETS["aal"], "Scalar reselect failed")
    old = OLD / "inputs" / "prepared" / "aal" / "RGIC" / "W1" / "0.npz"
    with np.load(old, allow_pickle=False) as handle:
        report_max_error = max(float(np.max(np.abs(j1[k] - handle[k])))
            for k in ("X_train_raw", "X_valid_raw", "X_train", "X_valid"))
    require(report_max_error == 0., "Original report configuration input changed")
    result = dict(status="PASS", analytical_f_max_abs_error=f_error, pearson_endpoint_byte_exact=True,
        report_configuration_max_abs_error=report_max_error, beta_final_operation=True,
        beta_zero_scalar_reselection=True, outer_labels_loaded=False,
        original_gain_cost_gamma_independent=True,
        independent_definition="gain=max(abs(peak_rho)-abs(rho0)-delta,0),cost=abs(peak_lag); gamma/cap affect only potential")
    write_json(Path(root) / "audit" / "search_features_qa.json", result)
    print(serialize(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--atlas", nargs="+", choices=ATLASES, default=list(ATLASES))
    parser.add_argument("--window", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--fold", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--shortlist-size", type=int, default=37)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.screen:
        data = SearchFeatures()
        for atlas, window, fold in itertools.product(args.atlas, args.window, args.fold):
            data.screen(atlas, window, fold, args.shortlist_size)


if __name__ == "__main__":
    main()
