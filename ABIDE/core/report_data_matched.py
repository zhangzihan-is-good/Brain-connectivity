"""Original-report ABIDE inputs with the three explicitly requested changes.

The original per-atlas fold roles and raw connection estimators are retained.
All methods use all-role labels for F selection; atlas voting is equal; the
explicit covariates are SEX, AGE, handedness and FIQ.  This experiment uses test
labels for supervised preprocessing and is not independent test generalization.
Only this new experiment receives prepared files.  Historical raw snapshots
are required to exist and are opened read-only; selected historical inputs,
models and predictions are not accepted.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import time
import warnings
from _paths import CORE

ROOT = Path(__file__).resolve().parents[1]
PART4 = ROOT.parent
OLD = PART4 / "按论文报告复现_20261027"
WORKSPACE = PART4.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(CORE))
import report_data as original
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

ATLASES = original.ATLASES
METHODS = original.METHODS
INDEPENDENT = original.INDEPENDENT
BUDGETS = original.BUDGETS
SPECS = ({"name": "original_joint", "readout": "joint"},)
CLASSIFIERS = ("svm", "fcn", "gcn", "transformer", "made")
ROLES = ("train", "valid", "test")
PHENOTYPES = WORKSPACE / "dataset/abide_data/01_subject_metadata/Phenotypic_V1_0b_preprocessed1.csv"
HAND_MAP = {"r": 1., "right": 1., "l": -1., "left": -1., "l->r": -1.,
            "ambi": 0., "ambidextrous": 0., "mixed": 0.}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return original.sha(path)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.partial" % os.getpid())
    temporary.write_text(original._serialize(value) + "\n", encoding="utf-8")
    for attempt in range(9):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 8:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def equal_atlas_weights():
    return np.full(3, 1. / 3., dtype=np.float64)


def numeric(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) and result > -999 else np.nan


def finite_values(values):
    return [float(v) if np.isfinite(v) else None for v in values]


class _ReadOnlyRawData(original.ReportData):
    """Fail instead of allowing the original reader to create missing caches."""
    def base_arrays(self, atlas):
        need((self.inputs / "report_base" / atlas / "manifest.json").is_file(),
             "Required existing raw base snapshot is missing: " + atlas)
        return super().base_arrays(atlas)

    def native_arrays(self, atlas):
        need((self.inputs / "native_1035" / atlas / "manifest.json").is_file(),
             "Required existing raw native snapshot is missing: " + atlas)
        return super().native_arrays(atlas)


class PaperData:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        need(self.root.parent == PART4.resolve() and self.root != OLD.resolve(),
             "Prepared outputs must use a new fourth-part experiment directory")
        self.inputs = self.root / "inputs"
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.reader = _ReadOnlyRawData(OLD)
        self._metadata = None
        self._demos = {}

    def metadata(self):
        if self._metadata is not None:
            return self._metadata
        source = self.reader.metadata()
        # Do not replace any atlas's original train/valid identities.
        meta = dict(source)
        identity_set = set(map(str, meta["ids"]))
        with PHENOTYPES.open(encoding="utf-8-sig", newline="") as stream:
            index = {}
            for row in csv.DictReader(stream):
                sid = row.get("FILE_ID", "").strip()
                if sid in identity_set:
                    need(sid not in index, "Duplicate phenotype FILE_ID: " + sid)
                    index[sid] = row
        demos = []
        for sid, y in zip(meta["ids"], meta["y"]):
            row = index.get(str(sid))
            need(row is not None, "Missing phenotype: " + str(sid))
            need(int(row["DX_GROUP"]) - 1 == int(y), "Phenotype diagnosis mismatch: " + str(sid))
            sex, age = numeric(row["SEX"]), numeric(row["AGE_AT_SCAN"])
            hand = HAND_MAP.get(row.get("HANDEDNESS_CATEGORY", "").strip().lower(), np.nan)
            demos.append((sex, age, hand, numeric(row.get("FIQ"))))
        demos = np.asarray(demos, dtype=np.float32)
        need(len(index) == 1035 and np.isfinite(demos[:, :2]).all(), "Phenotype cohort differs")
        need(np.array_equal(demos[:, 0], source["sex"].astype(np.float32)) and
             np.array_equal(demos[:, 1], source["age"].astype(np.float32)), "SEX/AGE differ from original")
        for fold in range(10):
            reference = meta["splits"]["aal"][str(fold)]["test"]
            need(all(np.array_equal(meta["splits"][a][str(fold)]["test"], reference) for a in ATLASES),
                 "Original three-atlas test identity/order differs")
        meta.update(demographics_raw=demos,
                    demographics_columns=["SEX", "AGE_AT_SCAN", "HANDEDNESS_CATEGORY", "FIQ"],
                    demographics_source=str(PHENOTYPES), demographics_source_sha256=sha(PHENOTYPES),
                    split_source="Original Pearson HDF5 atlas-specific experiments/{atlas}_whole roles")
        self._metadata = meta
        manifest = dict(status="complete", n=1035, n_ASD=int((meta["y"] == 0).sum()),
                        n_TC=int((meta["y"] == 1).sum()), ids=meta["ids"].tolist(),
                        original_atlas_specific_roles=True, split_source=meta["split_source"],
                        phenotype_columns=meta["demographics_columns"],
                        phenotype_missing=np.isnan(demos).sum(0).tolist(),
                        phenotype_source_sha256=meta["demographics_source_sha256"],
                        folds={a: {f: {r: meta["ids"][ix].tolist() for r, ix in roles.items()}
                                   for f, roles in meta["splits"][a].items()} for a in ATLASES},
                        raw_sources=meta["source_sha256"], internal_labels=dict(ASD=0, TC=1),
                        output_labels=dict(ASD=1, TC=0),
                        old_models_or_preselected_features_used=False,
                        supervised_selection_fit_scope="train_valid_test",
                        outer_labels_used_for_feature_selection=True,
                        independent_test_generalization=False)
        atomic(self.inputs / "original_cohort_and_folds.json", manifest)
        return meta

    def demographics(self, atlas, fold):
        key = (atlas, str(fold))
        if key not in self._demos:
            meta = self.metadata()
            roles = meta["splits"][atlas][str(fold)]
            raw = meta["demographics_raw"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fill = np.nanmean(raw[roles["train"]], axis=0)
            fill = np.where(np.isfinite(fill), fill, 0.)
            imputed = np.where(np.isnan(raw), fill, raw).astype(np.float32)
            self._demos[key] = (
                {r: imputed[ix] for r, ix in roles.items()},
                dict(columns=meta["demographics_columns"], imputation_values=fill.tolist(),
                     fit_subject_ids=meta["ids"][roles["train"]].tolist(), fit_scope="train_only",
                     scaling="none_raw_units_preserved", all_missing_train_fill=0.,
                     hand_mapping=HAND_MAP))
        return self._demos[key]

    def prepare(self, atlas, method, window, fold, spec=None, *, classifier="svm"):
        fold, window, classifier = str(fold), int(window), str(classifier).lower()
        need(atlas in ATLASES and method in METHODS and classifier in CLASSIFIERS,
             "Unknown atlas/method/classifier")
        need(window in (1, 2) or (window == 0 and method in INDEPENDENT), "Unknown connection window")
        if method == "RGIC":
            name = spec.get("name") if isinstance(spec, dict) else spec
            need(name in (None, "original_joint"), "RGIC is fixed original_joint; no readout search")
            resolved_spec = dict(SPECS[0])
        else:
            resolved_spec = dict(name="native", readout="joint")
        meta = self.metadata()
        roles = meta["splits"][atlas][fold]
        all_ids = np.concatenate([roles[r] for r in ROLES])
        need(len(all_ids) == 1035 and np.array_equal(np.sort(all_ids), np.arange(1035)),
             "Original role partition differs")
        effective = 0 if method in INDEPENDENT else window
        directory = self.inputs / "prepared" / classifier / atlas / method / resolved_spec["name"] / ("W%d" % effective)
        directory.mkdir(parents=True, exist_ok=True)
        npz_path, receipt_path = directory / (fold + ".npz"), directory / (fold + ".json")
        snapshot_family = "report_base" if method == "RGIC" else "native_1035"
        raw_manifest = OLD / "inputs" / snapshot_family / atlas / "manifest.json"
        need(raw_manifest.is_file(), "Existing raw cache manifest required")
        recipe = dict(version="original-report-atlas-roles-global-F-equal-demo4-v1",
                      classifier=classifier, atlas=atlas, method=method, fold=fold,
                      effective_window=effective, spec=resolved_spec, K=BUDGETS[atlas],
                      label_sha256=original.array_sha(meta["y"][all_ids]),
                      selection_ids_sha256=original.array_sha(meta["ids"][all_ids]),
                      role_sha256={r: original.array_sha(ix) for r, ix in roles.items()},
                      raw_manifest_sha256=sha(raw_manifest), module_sha256=sha(__file__),
                      original_data_source_sha256=sha(original.__file__),
                      phenotype_source_sha256=meta["demographics_source_sha256"],
                      preprocessing="original classifier-specific scaling; all-role F; report K; real four raw-unit covariates",
                      supervised_selection_scope="train_valid_test", outer_labels_used_for_feature_selection=True,
                      original_atlas_specific_roles=True, independent_test_generalization=False)
        if receipt_path.exists() and npz_path.exists():
            receipt = original.read_json(receipt_path)
            need(receipt["recipe"] == recipe and receipt["npz_sha256"] == sha(npz_path), "Prepared cache recipe/hash drift")
            with np.load(npz_path, allow_pickle=False) as handle:
                result = {key: handle[key] for key in handle.files}
            result["feature_metadata"] = receipt["feature_metadata"]
            return result

        strength, direction, detail = self.reader.candidates(atlas, method, max(1, window), fold)
        width = strength.shape[1]
        candidates = strength if direction is None else np.concatenate((strength, direction), axis=1)
        need(candidates.shape[0] == 1035 and np.isfinite(candidates).all(), "Nonfinite or malformed raw candidates")
        k = int(BUDGETS[atlas])
        need(candidates.shape[1] >= k, "Raw candidate pool is smaller than report K")
        upstream = None
        if method == "RGIC":
            with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                upstream_scores, _ = f_classif(np.asarray(candidates[all_ids], dtype=np.float32), meta["y"][all_ids])
            upstream_scores = np.nan_to_num(upstream_scores, nan=-np.inf,
                                           posinf=np.finfo(np.float32).max, neginf=-np.inf)
            upstream = np.argpartition(upstream_scores, -k)[-k:]
            upstream = upstream[np.argsort(upstream_scores[upstream])[::-1]]
            selector_inputs = np.asarray(candidates[:, upstream], dtype=np.float32)
        else:
            selector_inputs = candidates

        scaler = None
        scaler_fit = np.empty(0, dtype=np.int64)
        scaler_scope = "none"
        if classifier in ("fcn", "transformer", "gcn"):
            scaler_fit = all_ids if classifier == "gcn" else roles["train"]
            scaler_scope = "train_valid_test" if classifier == "gcn" else "train_only"
            scaler = StandardScaler().fit(selector_inputs[scaler_fit])
            selector_inputs = scaler.transform(selector_inputs).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            selector = SelectKBest(f_classif, k=k).fit(selector_inputs[all_ids], meta["y"][all_ids])
        local_selected = selector.get_support(indices=True).astype(np.int64)
        selected = local_selected if upstream is None else upstream[local_selected]
        need(len(selected) == k and len(np.unique(selected)) == k, "Selected report budget differs")
        demo, demo_detail = self.demographics(atlas, fold)
        result = {}
        for role, ix in roles.items():
            result["X_" + role] = np.asarray(selector_inputs[ix][:, local_selected], dtype=np.float32)
            result["y_" + role] = meta["y"][ix]
            result["y_" + role + "_ASD1"] = 1 - meta["y"][ix]
            result["demo_" + role] = demo[role]
            result["demos_" + role] = demo[role]
            result["ids_" + role] = meta["ids"][ix]
            result["sites_" + role] = meta["sites"][ix]
        metadata = dict(recipe=recipe, estimator=detail, selected_columns=selected.tolist(),
                        selected_column_ids=selected.tolist(), selected_edge_ids=(selected % width).tolist(),
                        selected_channel_ids=(selected >= width).astype(int).tolist(),
                        selected_k=k, n_scalar_selected=int((selected < width).sum()),
                        n_direction_selected=int((selected >= width).sum()),
                        selected_ic=int((selected < width).sum()), selected_direction=int((selected >= width).sum()),
                        direction_retained=bool(np.any(selected >= width)),
                        n_candidate_scalar=int(width), n_candidate_direction=0 if direction is None else int(direction.shape[1]),
                        upstream_RGIC_selected_columns=None if upstream is None else upstream.tolist(),
                        upstream_RGIC_selected_scores=None if upstream is None else finite_values(upstream_scores[upstream]),
                        upstream_RGIC_selection="raw_joint_all_role_F_argpartition_then_descending_F" if upstream is not None else "not_applied",
                        downstream_selected_input_columns=local_selected.tolist(),
                        downstream_selected_scores=finite_values(selector.scores_[local_selected]),
                        downstream_column_order="SelectKBest_support_preserves_input_order",
                        explicit_atlas_K_not_column_count_heuristic=True,
                        training_only=False, supervised_selection_fit_scope="train_valid_test",
                        feature_fit_ids=meta["ids"][all_ids].tolist(),
                        outer_labels_used_for_feature_selection=True, independent_test_generalization=False,
                        scaler_fit_scope=scaler_scope, scaler_fit_ids=meta["ids"][scaler_fit].tolist(),
                        scaler_mean_selected=None if scaler is None else scaler.mean_[local_selected].tolist(),
                        scaler_scale_selected=None if scaler is None else scaler.scale_[local_selected].tolist(),
                        svm_final_train_scaler="model_layer" if classifier == "svm" else "not_applicable",
                        demographics=demo_detail, source_root=str(OLD),
                        original_atlas_specific_roles=True, old_preselected_direction_opened=False,
                        internal_label_semantics=dict(ASD=0, TC=1), output_label_semantics=dict(ASD=1, TC=0))
        need(all(np.isfinite(value).all() for key, value in result.items()
                 if key.startswith(("X_", "demo_", "demos_"))), "Nonfinite prepared features/demographics")
        temporary = npz_path.with_name(npz_path.name + ".%d.partial" % os.getpid())
        with temporary.open("wb") as stream:
            np.savez(stream, **result)
        os.replace(temporary, npz_path)
        atomic(receipt_path, dict(status="complete", recipe=recipe, npz_sha256=sha(npz_path), feature_metadata=metadata))
        result["feature_metadata"] = metadata
        return result
