"""All-label joint preselection for the nine-method SVM/LR experiment.

Every method first ranks its COMPLETE native candidate pool using all 1035
train/validation/test labels.  A directional quota is allowed only inside the
already selected joint pool.  Feature scaling, direction beta, and classifiers
belong to the model layer.  The use of test labels is intentional and disclosed;
these are not independent held-out generalization estimates.
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


from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
PART4 = ROOT.parent
PREVIOUS_DATA = PART4 / "论文报告原设置_全标签_等权_四人口学_20261027/code/report_data_matched.py"
BROAD_FEATURES = PART4 / "RGIC广泛参数搜索_20261027/code/rgic_search_features.py"
sys.dont_write_bytecode = True


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_matched = _load_source("_joint_matched_raw_data", PREVIOUS_DATA)
_broad = _load_source("_joint_original_gate_components", BROAD_FEATURES)
np = _matched.np
f_classif = _matched.f_classif
ATLASES = _matched.ATLASES
METHODS = _matched.METHODS
INDEPENDENT = _matched.INDEPENDENT
BUDGETS = _matched.BUDGETS
ROLES = ("train", "valid", "test")
CORE_DEFAULTS = dict(gamma=.2, cap=.9, lambda_cost=1., percentile=95., alpha=1., delta=0.)
F_CHUNK_COLUMNS = 2048


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    # The report reader retains its stat-keyed memo and uses io_shared internally.
    return _matched.sha(path)


def _array_sha(values):
    return _matched.original.array_sha(np.asarray(values))


def _remember(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def normalize_params(params, atlas):
    """Accept either flat core fields or a nested ``core`` dictionary."""
    supplied = dict(params)
    nested = supplied.get("core", {})
    need(isinstance(nested, dict), "params.core must be a dictionary")
    core = dict(CORE_DEFAULTS, **nested)
    for name in CORE_DEFAULTS:
        if name in supplied:
            core[name] = supplied[name]
        core[name] = float(core[name])
        need(math.isfinite(core[name]), "Nonfinite RGIC core parameter: " + name)
    need(core["gamma"] >= 0 and core["lambda_cost"] >= 0 and core["delta"] >= 0,
         "gamma, lambda_cost and delta must be nonnegative")
    need(0 < core["cap"] <= 1 and 0 < core["percentile"] <= 100 and 0 <= core["alpha"] <= 1,
         "cap/percentile/alpha outside their natural domain")
    k_raw = supplied.get("K", BUDGETS[atlas])
    k = int(k_raw)
    need(k > 0 and float(k_raw) == k, "K must be a positive integer")
    readout = str(supplied.get("readout", supplied.get("mode", "joint_topk")))
    need(readout in ("joint_topk", "joint_then_quota"), "Unknown joint-preselection readout")
    quota = float(supplied.get("quota", .25))
    need(math.isfinite(quota) and 0 <= quota <= 1, "Direction quota must lie in [0,1]")
    factor = supplied.get("preselect_factor", 1.)
    if isinstance(factor, str) and factor.lower() == "all":
        factor = "all"
    else:
        factor = float(factor)
        need(math.isfinite(factor) and factor >= 1, "preselect_factor must be >=1 or 'all'")
    return dict(core=core, K=k, readout=readout, quota=quota, preselect_factor=factor)


def _clean_scores(scores):
    return np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-np.inf,
                         posinf=np.finfo(np.float32).max, neginf=-np.inf)


def _ranked_top(scores, count):
    """Descending F, with source-column order breaking exact ties.

    The deterministic tie rule makes appended constant-zero directional columns
    unable to change the alpha-zero endpoint's Pearson strength-column order.
    """
    need(0 < count <= len(scores), "Joint preselection count outside candidate pool")
    indices = np.arange(len(scores), dtype=np.int64)
    return indices[np.lexsort((indices, -scores))[:count]]


def select_joint_pool(scores, scalar_width, has_direction, params):
    """Select M jointly before any quota is considered; return original columns."""
    scores = _clean_scores(scores)
    k = params["K"]
    need(k <= len(scores), "K exceeds this method's complete candidate pool")
    if params["readout"] == "joint_topk":
        count = k
    elif params["preselect_factor"] == "all":
        count = len(scores)
    else:
        count = min(len(scores), max(k, int(math.ceil(params["preselect_factor"] * k))))
    pool = _ranked_top(scores, count)
    target_direction = int(round(k * params["quota"])) if has_direction else 0
    if params["readout"] == "joint_topk":
        selected = pool.copy()
    else:
        is_direction = pool >= scalar_width
        eligible = ~(is_direction & ~np.isfinite(scores[pool]))
        need(int(eligible.sum()) >= k,
             "Joint pool has fewer than K columns after excluding constant/undefined-F directions")
        strength_positions = np.flatnonzero(~is_direction)[:k - target_direction]
        direction_positions = np.flatnonzero(is_direction & eligible)[:target_direction]
        chosen = np.zeros(len(pool), dtype=bool)
        chosen[strength_positions] = True
        chosen[direction_positions] = True
        remaining = k - int(chosen.sum())
        if remaining:
            chosen[np.flatnonzero(~chosen & eligible)[:remaining]] = True
        # The final selected columns retain their order within the joint F pool.
        selected = pool[chosen]
    need(len(selected) == k and len(np.unique(selected)) == k and np.isin(selected, pool).all(),
         "Final quota readout bypassed or exceeded its jointly preselected pool")
    return selected, pool, target_direction


class FeatureEngine:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        need(self.root.parent == PART4.resolve(), "Outputs must remain in this fourth-part workspace")
        self.data = _matched.PaperData(root=self.root)
        self.reader = self.data.reader
        self._components = OrderedDict()
        self._scores = OrderedDict()

    def metadata(self):
        return self.data.metadata()

    def roles(self, atlas, fold):
        need(atlas in ATLASES and str(fold) in self.metadata()["splits"][atlas], "Unknown atlas/fold")
        return self.metadata()["splits"][atlas][str(fold)]

    def _gate(self, atlas, window, fold, core):
        key = (atlas, int(window), str(fold), core["delta"])
        if key in self._components:
            components = self._components[key]
            self._components.move_to_end(key)
        else:
            arrays = self.reader.base_arrays(atlas)
            train = self.roles(atlas, fold)["train"]
            prefix = "W%d_" % window
            gain = np.asarray(arrays[prefix + "gain"][train], dtype=np.float32)
            if core["delta"]:
                gain = np.maximum(gain - np.float32(core["delta"]), np.float32(0.)).astype(np.float32)
            components = _broad.gate_components(
                gain, arrays[prefix + "cost"][train], arrays[prefix + "peak_lag"][train],
                arrays[prefix + "peak_rho"][train])
            _remember(self._components, key, components, 12)
        eta = _broad.make_eta(components, core["lambda_cost"], core["percentile"])
        need(np.isfinite(eta).all() and np.all((eta >= 0) & (eta <= 1)), "Invalid group eta")
        raw = (components["reliability"] * _rgic_shared.abide_evidence(components, core["lambda_cost"])).astype(np.float32)
        return eta, dict(eta_sha256=_array_sha(eta), eta_mean=float(eta.mean()),
                         group_q_eta=float(np.percentile(raw, core["percentile"])),
                         gate_fit_scope="original_atlas_fold_train_only",
                         gate_fit_ids=self.metadata()["ids"][self.roles(atlas, fold)["train"]].tolist(),
                         delta_gain_policy="maximum(stored_zero_delta_gain-delta,0); recompute F/L/S and conditional cost")

    @staticmethod
    def _rgic_values(arrays, window, rows, edges, core, eta):
        ix = np.ix_(np.asarray(rows, dtype=np.int64), np.asarray(edges, dtype=np.int64))
        r0 = np.asarray(arrays["r0"][ix], dtype=np.float32)
        if core["alpha"] == 0:
            return _rgic_shared.abide_values(r0, None, None, None, core)
        prefix = "W%d_" % window
        rho = np.asarray(arrays[prefix + "peak_rho"][ix], dtype=np.float32)
        lag = np.asarray(arrays[prefix + "peak_lag"][ix])
        cached = arrays[prefix + "potential"][ix] if core["gamma"] == .2 and core["cap"] == .9 else None
        result = _rgic_shared.abide_values(r0, rho, lag, eta[np.asarray(edges, dtype=np.int64)], core, cached_potential=cached)
        need(all(np.isfinite(x).all() for x in result), "Nonfinite RGIC features")
        return result

    @staticmethod
    def _f(values, labels):
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            scores, _ = f_classif(np.asarray(values, dtype=np.float32), labels)
        return _clean_scores(scores)

    def prepare(self, atlas, window, fold, params, method="RGIC", include_test=False):
        need(atlas in ATLASES and method in METHODS and int(window) in (1, 2), "Unknown atlas/method/window")
        window, fold = int(window), str(fold)
        configuration = normalize_params(params, atlas)
        core = configuration["core"]
        meta, roles = self.metadata(), self.roles(atlas, fold)
        selection_ids = np.concatenate([roles[r] for r in ROLES])
        need(len(selection_ids) == 1035 and np.array_equal(np.sort(selection_ids), np.arange(1035)),
             "F selection must use every original train/valid/test identity exactly once")
        labels = meta["y"][selection_ids]
        if method == "RGIC":
            arrays = self.reader.base_arrays(atlas)
            if core["alpha"] == 0:
                # The intrinsic Pearson endpoint uses the SAME native Pearson
                # primitive, not merely another mathematically equivalent cache.
                arrays = dict(arrays, r0=self.reader.native_arrays(atlas)["r0"])
            width, has_direction = arrays["r0"].shape[1], True
            eta, detail = self._gate(atlas, window, fold, core)
            strength = direction = None
        else:
            strength, direction, detail = self.reader.candidates(atlas, method, window, fold)
            width, has_direction = strength.shape[1], direction is not None
            arrays = eta = None
        pool_width = width * (2 if has_direction else 1)
        if method == "RGIC" and core["alpha"] == 0:
            need(configuration["K"] <= width, "Intrinsic Pearson endpoint K exceeds the native Pearson pool")
        effective_window = 0 if method in INDEPENDENT else window
        score_key = (atlas, effective_window, fold, method,
                     tuple(core[name] for name in CORE_DEFAULTS) if method == "RGIC" else ())
        if score_key in self._scores:
            scores = self._scores[score_key]
            self._scores.move_to_end(score_key)
        else:
            scores = np.empty(pool_width, dtype=np.float32)
            for lo in range(0, width, F_CHUNK_COLUMNS):
                hi = min(width, lo + F_CHUNK_COLUMNS)
                if method == "RGIC":
                    scalar_chunk, direction_chunk = self._rgic_values(
                        arrays, window, selection_ids, np.arange(lo, hi), core, eta)
                else:
                    scalar_chunk = strength[selection_ids, lo:hi]
                    direction_chunk = None if direction is None else direction[selection_ids, lo:hi]
                scores[lo:hi] = self._f(scalar_chunk, labels)
                if has_direction:
                    scores[width + lo:width + hi] = self._f(direction_chunk, labels)
            need(not np.isnan(scores).any(), "Unresolved NaN F scores")
            _remember(self._scores, score_key, scores, 32)
        selected, pool, target_direction = select_joint_pool(scores, width, has_direction, configuration)
        direction_mask = selected >= width
        edges = selected % width
        unique_edges, inverse = np.unique(edges, return_inverse=True)
        demo, demo_detail = self.data.demographics(atlas, fold)
        result = dict(direction_mask=direction_mask.copy())
        output_roles = ROLES if include_test else ("train", "valid")
        for role in output_roles:
            rows = roles[role]
            if method == "RGIC":
                scalar_values, direction_values = self._rgic_values(arrays, window, rows, unique_edges, core, eta)
                values = np.where(direction_mask[None, :], direction_values[:, inverse], scalar_values[:, inverse])
            else:
                values = np.asarray(strength[np.ix_(rows, edges)], dtype=np.float32)
                if direction_mask.any():
                    values[:, direction_mask] = direction[np.ix_(rows, edges[direction_mask])]
            values = np.asarray(values, dtype=np.float32)
            need(values.shape == (len(rows), configuration["K"]) and np.isfinite(values).all(), "Invalid selected connection matrix")
            result["X_" + role] = values
            result["y_" + role] = meta["y"][rows]
            result["y_" + role + "_ASD1"] = 1 - meta["y"][rows]
            result["demos_" + role] = demo[role]
            result["demo_" + role] = demo[role]
            result["ids_" + role] = meta["ids"][rows]
            result["sites_" + role] = meta["sites"][rows]
        manifest = self.reader.inputs / ("report_base" if method == "RGIC" else "native_1035") / atlas / "manifest.json"
        recipe = dict(version="all-nine-complete-label-joint-pool-before-quota-v1", atlas=atlas,
                      method=method, fold=fold, effective_window=effective_window, params=configuration,
                      feature_module_sha256=sha(__file__), matched_raw_module_sha256=sha(PREVIOUS_DATA),
                      gate_component_module_sha256=sha(BROAD_FEATURES), raw_manifest_sha256=sha(manifest),
                      intrinsic_endpoint_native_manifest_sha256=(sha(self.reader.inputs / "native_1035" / atlas / "manifest.json")
                                                                  if method == "RGIC" and core["alpha"] == 0 else None),
                      raw_source_sha256=meta["source_sha256"],
                      all_F_label_sha256=_array_sha(labels), selection_ids_sha256=_array_sha(meta["ids"][selection_ids]),
                      role_sha256={r: _array_sha(ix) for r, ix in roles.items()})
        result["feature_metadata"] = dict(
            recipe=recipe, estimator=detail, params=configuration,
            full_joint_candidate_count=int(pool_width), n_candidate_scalar=int(width),
            n_candidate_direction=int(width if has_direction else 0), K=configuration["K"], M=int(len(pool)),
            joint_preselected_columns=pool.tolist(), selected_columns=selected.tolist(),
            selected_column_ids=selected.tolist(), selected_edge_ids=edges.tolist(),
            selected_channel_ids=direction_mask.astype(int).tolist(), direction_mask=direction_mask.tolist(),
            selected_scores=[float(v) if np.isfinite(v) else None for v in scores[selected]],
            n_scalar_selected=int((~direction_mask).sum()), n_direction_selected=int(direction_mask.sum()),
            target_direction_count=int(target_direction), direction_retained=bool(direction_mask.any()),
            joint_preselection_before_quota=True, final_columns_subset_of_joint_pool=True,
            joint_F_tie_rule="descending_F_then_ascending_original_column_index",
            quota_constant_direction_policy="exclude_minus_infinity_F_directions_then_fill_from_remaining_joint_pool",
            F_score_fit_scope="complete_train_valid_test_1035", supervised_selection_fit_scope="train_valid_test",
            feature_fit_ids=meta["ids"][selection_ids].tolist(), all_F_label_sha256=_array_sha(labels),
            outer_labels_used_for_feature_selection=True, independent_test_generalization=False,
            scaler_fit_scope="none_raw_connections_returned", beta_applied=False,
            beta_policy="model_layer_after_final_scaling_only", demographics=demo_detail,
            original_atlas_specific_roles=True, include_test=bool(include_test),
            raw_snapshot_only=True, old_selected_features_models_predictions_opened=False,
            internal_label_semantics=dict(ASD=0, TC=1), output_label_semantics=dict(ASD=1, TC=0),
            potential_policy="cached_original_potential_for_gamma0.2_cap0.9; otherwise_clipped_peak_compensation",
            alpha_zero_policy="intrinsic_RGIC_degeneration_same_native_Pearson_r0_zero_D_complete_joint_F_no_prediction_mixture",
            intrinsic_Pearson_endpoint=bool(method == "RGIC" and core["alpha"] == 0),
            prediction_fusion=False)
        return result
