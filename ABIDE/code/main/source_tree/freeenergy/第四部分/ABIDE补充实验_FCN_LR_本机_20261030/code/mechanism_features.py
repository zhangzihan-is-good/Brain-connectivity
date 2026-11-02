"""RGIC mechanism readouts, using the unchanged ABIDE primitives and protocol.

No fitting, hyperparameter search, prediction fusion, file writes or remote
execution occurs here. Inject the original joint_features module and an original
FeatureEngine whose output root is this NEW experiment directory. The original
reader may write its cohort metadata only to that supplied new root.

Random signs are independent PCG64 Rademacher draws for each canonical
subject-by-edge cell, shared across folds/roles/classifiers for a given seed,
atlas, window and ordered cohort. IC and |D| remain unchanged. Multiple random
seeds must be run/reported separately, not selected by classifier test results.
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
import copy
import hashlib
import json
from pathlib import Path

VERSION = "rgic-mechanism-readouts-v2"
CHANNELS = ("IC", "D", "IC+D")
DIRECTIONS = ("signed", "abs", "random_sign")
COMPONENT_REMOVALS = ("no_cost", "no_reliability", "no_compensation")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    return _io_shared.sha256_file(path, block_size=4*1024*1024)


def _components(value):
    if value == "full" or value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    _need(len(set(values)) == len(values) and set(values).issubset(COMPONENT_REMOVALS), "Unknown/duplicate component removal")
    return tuple(sorted(values))


class MechanismFeatures:
    """One instance per worker; output views match original FeatureEngine.prepare.

    prepare(..., selection='fixed') uses a signed/full-component anchor with the
    SAME channel family and K. It never pads a smaller mixed-anchor subset with
    zeros or claims that dropping columns kept K matched. A caller-supplied
    anchor can be a view or feature_metadata dictionary. An original native
    anchor is accepted only for its unmodified full-cohort IC+D/all-label scope.
    An engine instance must keep its active role/label context fixed; use a new
    base engine per cohort/permutation, as the native gate cache is role keyed.
    """
    def __init__(self, base_engine, joint_features_module):
        self.base, self.jf = base_engine, joint_features_module
        self.np = self.jf.np
        self._component_cache = OrderedDict()
        self._score_cache = OrderedDict()
        self._sign_cache = OrderedDict()
        self.source_sha256 = {
            "mechanism_features": _sha(__file__),
            "original_joint_features": _sha(self.jf.__file__),
            "original_gate_components": _sha(self.jf._broad.__file__),
            "original_matched_reader": _sha(self.jf._matched.__file__),
        }

    @staticmethod
    def _cache(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _context(self, atlas, fold, selection_scope):
        np = self.np
        _need(selection_scope in ("all", "train"), "selection_scope must be all or train")
        meta = self.base.metadata()
        n = len(meta["ids"])
        _need(len(meta["y"]) == n and len(meta["sites"]) == n, "Global metadata lengths differ")
        roles = {}
        for role in self.jf.ROLES:
            raw = np.asarray(self.base.roles(atlas, fold)[role])
            _need(raw.ndim == 1 and np.issubdtype(raw.dtype, np.integer), "Role indices must be one-dimensional global integers")
            roles[role] = raw.astype(np.int64, copy=False)
        active = np.concatenate([roles[r] for r in self.jf.ROLES])
        _need(len(roles["train"]) > 0 and len(active) == len(np.unique(active)) and
              np.all((active >= 0) & (active < n)), "Active roles overlap or exceed global raw rows")
        rows = active if selection_scope == "all" else roles["train"]
        _need(len(np.unique(meta["y"][rows])) >= 2, "F selection requires at least two label classes")
        return meta, roles, rows

    def _original_roles(self, atlas, fold, meta, roles):
        """An explicit byte-role identity check, never inferred from row count."""
        np = self.np
        if len(meta["ids"]) != 1035:
            return False
        original = self.base.reader.metadata()
        reference = original["splits"][atlas][str(fold)]
        return (np.array_equal(meta["ids"], original["ids"]) and
                np.array_equal(meta["y"], original["y"]) and
                all(np.array_equal(roles[r], reference[r]) for r in self.jf.ROLES) and
                len(np.concatenate([roles[r] for r in self.jf.ROLES])) == 1035)

    def _configuration(self, atlas, params, removals):
        original = self.jf.normalize_params(params, atlas)
        effective = copy.deepcopy(original)
        if "no_cost" in removals:
            effective["core"]["lambda_cost"] = 0.
        if "no_compensation" in removals:
            effective["core"]["gamma"] = 0.
        return original, effective

    def _arrays(self, atlas, core, meta):
        arrays = self.base.reader.base_arrays(atlas)
        if core["alpha"] == 0:
            arrays = dict(arrays, r0=self.base.reader.native_arrays(atlas)["r0"])
        _need(arrays["r0"].shape[0] == len(meta["ids"]), "Raw arrays must retain the global metadata row axis")
        return arrays

    def _gate(self, atlas, window, fold, core, removals):
        np = self.np
        if "no_reliability" not in removals:
            return self.base._gate(atlas, window, fold, core)
        train = np.asarray(self.base.roles(atlas, fold)["train"], dtype=np.int64)
        key = (atlas, int(window), str(fold), float(core["delta"]), self.jf._array_sha(train))
        if key not in self._component_cache:
            arrays = self.base.reader.base_arrays(atlas)
            prefix = "W%d_" % int(window)
            gain = np.asarray(arrays[prefix + "gain"][train], dtype=np.float32)
            if core["delta"]:
                gain = np.maximum(gain - np.float32(core["delta"]), np.float32(0.)).astype(np.float32)
            native = self.jf._broad.gate_components(gain, arrays[prefix + "cost"][train],
                                                  arrays[prefix + "peak_lag"][train], arrays[prefix + "peak_rho"][train])
            self._cache(self._component_cache, key, native, 12)
        native = self._component_cache[key]
        ablated = dict(native, reliability=np.ones_like(native["reliability"], dtype=np.float32))
        # Reuse original evidence, percentile normalization, clipping and dtype.
        # This removes F*L*S only. It does not set eta itself to one.
        eta = self.jf._broad.make_eta(ablated, core["lambda_cost"], core["percentile"])
        evidence = _rgic_shared.abide_evidence(native, core["lambda_cost"])
        _need(np.isfinite(eta).all() and np.all((eta >= 0) & (eta <= 1)), "Invalid ablated eta")
        return eta, dict(eta_sha256=self.jf._array_sha(eta), eta_mean=float(eta.mean()),
                         group_q_eta=float(np.percentile(evidence, core["percentile"])),
                         gate_fit_scope="active_train_only",
                         gate_fit_ids=self.base.metadata()["ids"][self.base.roles(atlas, fold)["train"]].tolist(),
                         removed_factor="frequency_times_lag_consistency_times_sign_consistency",
                         evidence_and_percentile_normalization_unchanged=True)

    def _signs(self, atlas, window, seed, meta, width):
        np = self.np
        _need(seed is not None and not isinstance(seed, bool) and int(seed) == seed and int(seed) >= 0,
              "random_sign requires an explicit nonnegative integer seed")
        cohort_hash = self.jf._array_sha(meta["ids"])
        key = (atlas, int(window), int(seed), cohort_hash, int(width))
        if key not in self._sign_cache:
            domain = dict(version=VERSION, intervention="subject_by_edge_Rademacher", atlas=atlas,
                          window=int(window), seed=int(seed), ordered_cohort_sha256=cohort_hash)
            stream_seed = int.from_bytes(hashlib.sha256(_canonical(domain).encode("utf-8")).digest()[:16], "little")
            rng = np.random.Generator(np.random.PCG64(stream_seed))
            signs = rng.integers(0, 2, size=(len(meta["ids"]), int(width)), dtype=np.int8)
            signs = signs * np.int8(2) - np.int8(1)
            self._cache(self._sign_cache, key, (signs, domain), 2)
        return self._sign_cache[key]

    def _values(self, arrays, window, rows, edges, core, eta, direction, signs):
        scalar, directional = self.base._rgic_values(arrays, int(window), rows, edges, core, eta)
        if direction == "abs":
            directional = self.np.abs(directional).astype(self.np.float32)
        elif direction == "random_sign":
            directional = (directional * signs[self.np.ix_(rows, edges)]).astype(self.np.float32)
        return scalar, directional

    def _scores(self, arrays, atlas, window, fold, configuration, removals, channels, direction, seed, eta, meta, signs,
                selection_scope, rows):
        np = self.np
        core, width = configuration["core"], int(arrays["r0"].shape[1])
        labels = meta["y"][rows]
        key = (atlas, int(window), str(fold), _canonical(core), removals, channels, direction,
               int(seed) if direction == "random_sign" else None, selection_scope,
               self.jf._array_sha(rows), self.jf._array_sha(labels), self.jf._array_sha(eta))
        if key in self._score_cache:
            self._score_cache.move_to_end(key)
            return self._score_cache[key]
        score_ic = np.empty(width, dtype=np.float32) if channels != "D" else None
        score_d = np.empty(width, dtype=np.float32) if channels != "IC" else None
        for lo in range(0, width, self.jf.F_CHUNK_COLUMNS):
            hi = min(width, lo + self.jf.F_CHUNK_COLUMNS)
            edges = np.arange(lo, hi)
            ic, d = self._values(arrays, window, rows, edges, core, eta, direction, signs)
            if score_ic is not None:
                score_ic[lo:hi] = self.base._f(ic, labels)
            if score_d is not None:
                score_d[lo:hi] = self.base._f(d, labels)
        scores = score_d if channels == "D" else score_ic if channels == "IC" else np.concatenate((score_ic, score_d))
        _need(not np.isnan(scores).any(), "Unresolved mechanism F scores")
        return self._cache(self._score_cache, key, scores, 24)

    def _anchor(self, anchor, atlas, window, fold, params, channels, include_test, selection_scope, selection_ids, roles):
        if anchor is None:
            anchor = self.prepare(atlas, window, fold, params, channels=channels, direction="signed",
                                  selection="rescreen", components="full", include_test=include_test, selection_scope=selection_scope)
        metadata = anchor.get("feature_metadata", anchor)
        recipe = metadata["recipe"]
        _need(recipe["atlas"] == atlas and str(recipe["fold"]) == str(fold) and int(recipe["effective_window"]) == int(window),
              "Fixed anchor atlas/window/fold differs")
        _need(metadata["params"] == params, "Fixed anchor must use the same original RGIC parameters and K")
        mechanism = metadata.get("mechanism", {})
        _need(not mechanism or (mechanism["channels"] == channels and mechanism["direction"] == "signed" and
                                not mechanism["component_removals"] and mechanism["selection"] == "rescreen"),
              "Anchor must be the signed full-component rescreen in the same channel family")
        if mechanism:
            _need(mechanism["selection_scope"] == selection_scope, "Fixed anchor selection scope differs")
        else:
            _need(channels == "IC+D" and selection_scope == "all" and
                  self._original_roles(atlas, fold, self.base.metadata(), roles),
                  "An original native anchor requires the unchanged full-cohort IC+D/all-label scope")
        _need(recipe["selection_ids_sha256"] == self.jf._array_sha(self.base.metadata()["ids"][selection_ids]) and
              recipe["all_F_label_sha256"] == self.jf._array_sha(self.base.metadata()["y"][selection_ids]) and
              recipe["role_sha256"] == {r: self.jf._array_sha(ix) for r, ix in roles.items()},
              "Fixed anchor active roles or selection labels differ")
        return metadata

    def prepare(self, atlas, window, fold, params, *, channels="IC+D", direction="signed", selection="rescreen",
                components="full", random_seed=None, anchor=None, include_test=False, selection_scope="all"):
        np = self.np
        window, fold = int(window), str(fold)
        _need(atlas in self.jf.ATLASES and window in (1, 2), "Unknown atlas/window")
        channels = {"IC-only": "IC", "D-only": "D"}.get(channels, channels)
        _need(channels in CHANNELS and direction in DIRECTIONS and selection in ("rescreen", "fixed"), "Unknown mechanism readout")
        removals = _components(components)
        _need(direction == "random_sign" or random_seed is None, "A random seed is meaningful only for random_sign")
        _need(selection == "fixed" or anchor is None, "An anchor may be supplied only for fixed selection")
        original_params, configuration = self._configuration(atlas, params, removals)
        meta, roles, selection_ids = self._context(atlas, fold, selection_scope)
        original_roles = self._original_roles(atlas, fold, meta, roles)
        # The complete, unmodified control is byte-identical to the original
        # FeatureEngine, rather than a new arithmetic reconstruction.
        passthrough = (selection_scope == "all" and original_roles and channels == "IC+D" and
                       direction == "signed" and not removals and selection == "rescreen")
        baseline = (self.base.prepare(atlas, window, fold, original_params, method="RGIC", include_test=include_test)
                    if passthrough else None)
        base_metadata = baseline["feature_metadata"] if baseline is not None else None
        arrays = self._arrays(atlas, configuration["core"], meta)
        width = int(arrays["r0"].shape[1])
        _need(original_params["K"] <= width * (2 if channels == "IC+D" else 1), "K exceeds the requested native channel pool")
        _need(configuration["core"]["alpha"] != 0 or original_params["K"] <= width,
              "Intrinsic Pearson endpoint K exceeds the native Pearson pool")
        signs = domain = None
        if direction == "random_sign":
            signs, domain = self._signs(atlas, window, random_seed, meta, width)
        eta, gate_detail = self._gate(atlas, window, fold, configuration["core"], removals)
        gate_detail = dict(gate_detail, gate_fit_scope="active_train_only")
        if passthrough:
            selected = np.asarray(base_metadata["selected_columns"], dtype=np.int64)
            pool = np.asarray(base_metadata["joint_preselected_columns"], dtype=np.int64)
            target_direction = base_metadata["target_direction_count"]
            selected_scores = base_metadata["selected_scores"]
            anchor_metadata = None
        elif selection == "fixed":
            anchor_metadata = self._anchor(anchor, atlas, window, fold, original_params, channels, include_test,
                                           selection_scope, selection_ids, roles)
            selected = np.asarray(anchor_metadata["selected_columns"], dtype=np.int64)
            pool = np.asarray(anchor_metadata["joint_preselected_columns"], dtype=np.int64)
            target_direction = anchor_metadata["target_direction_count"]
            selected_scores = anchor_metadata["selected_scores"]
        else:
            anchor_metadata = None
            scores = self._scores(arrays, atlas, window, fold, configuration, removals, channels, direction,
                                  random_seed, eta, meta, signs, selection_scope, selection_ids)
            scalar_width, has_direction = (width if channels != "D" else 0), channels != "IC"
            local, local_pool, target_direction = self.jf.select_joint_pool(scores, scalar_width, has_direction, original_params)
            selected_scores = [float(value) if np.isfinite(value) else None for value in scores[local]]
            selected = local + (width if channels == "D" else 0)
            pool = local_pool + (width if channels == "D" else 0)
        _need(len(selected) == original_params["K"] and len(np.unique(selected)) == len(selected) and
              np.all((selected >= 0) & (selected < 2 * width)) and np.isin(selected, pool).all(), "Invalid matched-K mechanism anchor/readout")
        mask, edges = selected >= width, selected % width
        _need(channels != "IC" or not mask.any(), "IC-only fixed anchor contains D; refusing unmatched-K truncation")
        _need(channels != "D" or mask.all(), "D-only fixed anchor contains IC; refusing unmatched-K truncation")
        unique_edges, inverse = np.unique(edges, return_inverse=True)
        result = dict(baseline) if baseline is not None else {}
        demo, demo_detail = self.base.data.demographics(atlas, fold)
        if not passthrough:
            for role in self.jf.ROLES if include_test else ("train", "valid"):
                ic, d = self._values(arrays, window, roles[role], unique_edges, configuration["core"], eta, direction, signs)
                values = np.where(mask[None, :], d[:, inverse], ic[:, inverse]).astype(np.float32)
                _need(values.shape == (len(roles[role]), original_params["K"]) and np.isfinite(values).all(), "Invalid mechanism feature matrix")
                result["X_" + role] = values
                result["y_" + role] = meta["y"][roles[role]]
                result["y_" + role + "_ASD1"] = 1 - meta["y"][roles[role]]
                result["demos_" + role] = demo[role]
                result["demo_" + role] = demo[role]
                result["ids_" + role] = meta["ids"][roles[role]]
                result["sites_" + role] = meta["sites"][roles[role]]
        result["direction_mask"] = mask.copy()
        labels = meta["y"][selection_ids]
        label_sha = self.jf._array_sha(labels)
        # Keep original file hashes and raw dtype arithmetic. Only active-role
        # identity and the explicit F scope differ for low-FD/LOSO/permutation.
        recipe = dict(version=VERSION, atlas=atlas, method="RGIC", fold=fold, effective_window=window,
                      params=original_params, feature_module_sha256=self.source_sha256["original_joint_features"],
                      matched_raw_module_sha256=self.source_sha256["original_matched_reader"],
                      gate_component_module_sha256=self.source_sha256["original_gate_components"],
                      raw_manifest_sha256=_sha(self.base.reader.inputs / "report_base" / atlas / "manifest.json"),
                      intrinsic_endpoint_native_manifest_sha256=(_sha(self.base.reader.inputs / "native_1035" / atlas / "manifest.json")
                                                                  if configuration["core"]["alpha"] == 0 else None),
                      raw_source_sha256=meta["source_sha256"], all_F_label_sha256=label_sha,
                      selection_ids_sha256=self.jf._array_sha(meta["ids"][selection_ids]),
                      role_sha256={r: self.jf._array_sha(ix) for r, ix in roles.items()},
                      selection_scope=selection_scope, F_fit_scope=selection_scope)
        metadata = copy.deepcopy(base_metadata) if base_metadata is not None else dict(
            joint_preselection_before_quota=True, final_columns_subset_of_joint_pool=True,
            joint_F_tie_rule="descending_F_then_ascending_original_column_index",
            quota_constant_direction_policy="exclude_minus_infinity_F_directions_then_fill_from_remaining_joint_pool",
            scaler_fit_scope="none_raw_connections_returned", beta_policy="model_layer_after_final_scaling_only",
            raw_snapshot_only=True, old_selected_features_models_predictions_opened=False,
            internal_label_semantics=dict(ASD=0, TC=1), output_label_semantics=dict(ASD=1, TC=0),
            potential_policy="cached_original_potential_for_gamma0.2_cap0.9; otherwise_clipped_peak_compensation",
            alpha_zero_policy="intrinsic_RGIC_degeneration_same_native_Pearson_r0_zero_D_complete_joint_F_no_prediction_mixture")
        metadata.update(K=original_params["K"], M=int(len(pool)), params=original_params,
                        full_joint_candidate_count=width * (2 if channels == "IC+D" else 1),
                        n_candidate_scalar=width if channels != "D" else 0, n_candidate_direction=width if channels != "IC" else 0,
                        selected_columns=selected.tolist(), selected_column_ids=selected.tolist(), joint_preselected_columns=pool.tolist(),
                        selected_edge_ids=edges.tolist(), selected_channel_ids=mask.astype(int).tolist(), direction_mask=mask.tolist(),
                        selected_scores=selected_scores, n_scalar_selected=int((~mask).sum()), n_direction_selected=int(mask.sum()),
                        target_direction_count=int(target_direction), direction_retained=bool(mask.any()), estimator=gate_detail,
                        prediction_fusion=False, beta_applied=False, demographics=demo_detail,
                        F_score_fit_scope=("complete_active_train_valid_test" if selection_scope == "all" else "active_train_only"),
                        supervised_selection_fit_scope=("train_valid_test" if selection_scope == "all" else "train"),
                        feature_fit_ids=meta["ids"][selection_ids].tolist(), all_F_label_sha256=label_sha,
                        outer_labels_used_for_feature_selection=bool(selection_scope == "all" and len(roles["test"])),
                        independent_test_generalization=False,
                        independent_generalization_not_asserted_by_feature_module=True,
                        original_atlas_specific_roles=original_roles, include_test=bool(include_test),
                        intrinsic_Pearson_endpoint=bool(configuration["core"]["alpha"] == 0 and channels == "IC+D"),
                        active_cohort_n=int(sum(len(ix) for ix in roles.values())), global_raw_cohort_n=int(len(meta["ids"])))
        mechanism = dict(version=VERSION, channels=channels, direction=direction, selection=selection,
                         selection_scope=selection_scope,
                         component_removals=list(removals), effective_params=configuration,
                         random_seed=int(random_seed) if direction == "random_sign" else None,
                         random_sign_domain=domain, random_sign_unit="independent_subject_by_native_edge" if domain else None,
                         signs_shared_across_folds_roles_classifiers=bool(domain),
                         randomization_precedes_F_rescreen=selection == "rescreen" and bool(domain),
                         fixed_original_columns=selection == "fixed", fixed_anchor_sha256=(hashlib.sha256(_canonical(anchor_metadata).encode()).hexdigest() if anchor_metadata else None),
                         total_connection_K=original_params["K"],
                         cross_condition_K_matching="caller_must_use_identical_K_no_larger_than_each_compared_pool",
                         unchanged_original_control_passthrough=passthrough,
                         quota_applies_to_channel_composition=bool(channels == "IC+D"),
                         selected_scores_scope="signed_full_component_anchor_F" if selection == "fixed" else "current_transformed_F",
                         no_compensation_definition="gamma=0; retain peak-lag search and original magnitude cap",
                         no_cost_definition="lambda_cost=0; retain group gain and reliability/normalization",
                         no_reliability_definition="replace F*L*S factor by1 before original make_eta percentile normalization",
                         original_D_definition="sign(peak_lag)*abs(IC-r0)",
                         demographics_not_part_of_connection_K=True, downstream_beta_and_scaling_not_applied=True)
        metadata["mechanism"] = mechanism
        metadata["recipe"] = dict(recipe, mechanism=mechanism, mechanism_source_sha256=self.source_sha256)
        result["feature_metadata"] = metadata
        return result

    def pool(self, atlas, window, fold, core=None, components="full"):
        """Return signed native (IC, D, audit) on the FULL raw reader row axis.

        Active train identities alone fit the reliability gate. No F ranking,
        quota, beta, demographics, scaling or classification is performed. The
        caller must index rows in that reader's metadata order (1035 for the
        original reader, 200 for RawReader), and predeclare matched-K comparisons.
        """
        np = self.np
        window, fold = int(window), str(fold)
        _need(atlas in self.jf.ATLASES and window in (1, 2), "Unknown atlas/window")
        removals = _components(components)
        original, effective = self._configuration(atlas, {"core": dict(core or {}), "K": 1}, removals)
        meta, roles, _ = self._context(atlas, fold, "train")
        arrays = self._arrays(atlas, effective["core"], meta)
        rows = np.arange(len(meta["ids"]), dtype=np.int64)
        width = int(arrays["r0"].shape[1])
        eta, detail = self._gate(atlas, window, fold, effective["core"], removals)
        detail = dict(detail, gate_fit_scope="active_train_only")
        scalar = np.empty((len(rows), width), dtype=np.float32)
        direction = np.empty_like(scalar)
        for lo in range(0, width, self.jf.F_CHUNK_COLUMNS):
            hi = min(width, lo + self.jf.F_CHUNK_COLUMNS)
            scalar[:, lo:hi], direction[:, lo:hi] = self.base._rgic_values(
                arrays, window, rows, np.arange(lo, hi), effective["core"], eta)
        _need(np.isfinite(scalar).all() and np.isfinite(direction).all(), "Invalid full mechanism pool")
        audit = dict(version=VERSION, atlas=atlas, window=window, fold=fold,
                     row_axis="global_raw_metadata_order", global_row_indices=rows.tolist(),
                     global_ids_sha256=self.jf._array_sha(meta["ids"]),
                     raw_source_sha256=meta["source_sha256"],
                     raw_manifest_sha256=_sha(self.base.reader.inputs / "report_base" / atlas / "manifest.json"),
                     intrinsic_endpoint_native_manifest_sha256=(_sha(self.base.reader.inputs / "native_1035" / atlas / "manifest.json")
                                                                 if effective["core"]["alpha"] == 0 else None),
                     active_role_sha256={r: self.jf._array_sha(ix) for r, ix in roles.items()},
                     original_core=original["core"], effective_core=effective["core"],
                     component_removals=list(removals), estimator=detail,
                     scalar_shape=list(scalar.shape), direction_shape=list(direction.shape),
                     source_sha256=self.source_sha256, original_D_definition="sign(peak_lag)*abs(IC-r0)",
                     no_F_selection_no_beta_no_demographics=True, prediction_fusion=False)
        return scalar, direction, audit
