"""Data-only controls for the local ABIDE FCN/LR supplementary pilot.

No model training, source patching, output writing, or remote execution occurs
here.  The caller supplies the complete native candidate matrices BEFORE F
preselection, and applies its frozen all-label F screen and K readout afterward.
Subset roles returned here require a newly fitted subset gate/F screen; slicing
already selected full-cohort features is a different descriptive experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


K_GRID = (250, 500, 1000, 2000)
ROLES = ("train", "valid", "test")
DEMOGRAPHIC_COLUMNS = ("SEX", "AGE_AT_SCAN", "HANDEDNESS_CATEGORY", "FIQ")
FD_COLUMN = "func_mean_fd"
LOW_MOTION_MAX_FD_MM = 0.2


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(value):
    def numeric_json(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError("Non-JSON audit value: " + type(item).__name__)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False, default=numeric_json)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _array_sha(value):
    array = np.ascontiguousarray(value)
    return _identity(dict(shape=list(array.shape), dtype=str(array.dtype),
                          bytes_sha256=hashlib.sha256(array.tobytes()).hexdigest()))


def _seed(seed, *parts):
    _need(isinstance(seed, (int, np.integer)) and int(seed) >= 0, "seed must be a nonnegative integer")
    return int(_identity([int(seed), *parts])[:8], 16)


def _ids(subject_ids, n):
    ids = [str(v) for v in (range(n) if subject_ids is None else subject_ids)]
    _need(len(ids) == n and len(set(ids)) == n, "Subject identities must be unique and aligned")
    return np.asarray(ids, dtype=str)


def _labels(labels, n):
    labels = np.asarray(labels)
    _need(labels.shape == (n,) and np.isin(labels, (0, 1)).all(), "Labels must be aligned binary ASD0/TC1 values")
    return labels.astype(np.int64, copy=False)


def validate_roles(roles, n_samples, require_cover=False):
    """Return disjoint integer roles without changing any subject's role."""
    _need(set(roles) == set(ROLES), "Exactly train/valid/test roles are required")
    result = {}
    for role in ROLES:
        raw = np.asarray(roles[role])
        _need(raw.ndim == 1 and (raw.size == 0 or np.issubdtype(raw.dtype, np.integer)), "Role rows must be one-dimensional integers")
        rows = raw.astype(np.int64, copy=True)
        _need(np.all((rows >= 0) & (rows < n_samples)) and len(np.unique(rows)) == len(rows), "Invalid or duplicate role row")
        result[role] = rows
    combined = np.concatenate([result[r] for r in ROLES])
    _need(len(np.unique(combined)) == len(combined), "Roles overlap")
    if require_cover:
        _need(len(combined) == n_samples and np.array_equal(np.sort(combined), np.arange(n_samples)), "Roles do not cover the cohort exactly")
    return result


def load_mean_fd(phenotype_csv, subject_ids):
    """Read existing PCP mean-FD metadata aligned by FILE_ID; never fill missing FD."""
    path = Path(phenotype_csv)
    ids = _ids(subject_ids, len(subject_ids))
    wanted, index = set(ids.tolist()), {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _need(reader.fieldnames is not None and {"FILE_ID", FD_COLUMN}.issubset(reader.fieldnames), "Phenotype FILE_ID/func_mean_fd columns missing")
        for row in reader:
            sid = row["FILE_ID"].strip()
            if sid in wanted:
                _need(sid not in index, "Duplicate phenotype FILE_ID: " + sid)
                index[sid] = row[FD_COLUMN]
    _need(set(index) == wanted, "A cohort FILE_ID is absent from the phenotype source")
    result = np.full(len(ids), np.nan, dtype=np.float64)
    for i, sid in enumerate(ids):
        try:
            value = float(index[sid])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value >= 0:
            result[i] = value
    return result, dict(column=FD_COLUMN, units="mm", source=str(path.resolve()),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        n=len(ids), n_missing=int((~np.isfinite(result)).sum()),
                        imputation_for_subset_membership=False)


def low_motion_mask(mean_fd):
    """Use the pre-existing finite 0<=func_mean_fd<=0.2 mm rule, inclusive."""
    fd = np.asarray(mean_fd, dtype=np.float64)
    _need(fd.ndim == 1, "FD must be a subject-aligned vector")
    valid = np.isfinite(fd) & (fd >= 0)
    mask = valid & (fd <= LOW_MOTION_MAX_FD_MM)
    return mask, dict(column=FD_COLUMN, units="mm", rule="finite and 0 <= func_mean_fd <= 0.2",
                      threshold_fixed_before_results=True, n_total=len(fd), n_retained=int(mask.sum()),
                      n_missing_or_invalid=int((~valid).sum()), n_above_threshold=int((valid & ~mask).sum()),
                      mask_sha256=_array_sha(mask))


def describe_population(rows, labels=None, sites=None):
    rows = np.asarray(rows)
    _need(rows.ndim == 1 and (rows.size == 0 or np.issubdtype(rows.dtype, np.integer)), "Population rows must be integer indices")
    rows = rows.astype(np.int64, copy=False)
    _need(np.all(rows >= 0) and len(np.unique(rows)) == len(rows), "Population rows must be unique nonnegative indices")
    description = dict(n=int(len(rows)))
    if labels is not None:
        y = np.asarray(labels)
        _need(rows.size == 0 or rows.max() < len(y), "Population/labels misaligned")
        _labels(y, len(y))
        description.update(n_ASD=int((y[rows] == 0).sum()), n_TC=int((y[rows] == 1).sum()),
                           both_classes=bool(len(np.unique(y[rows])) == 2))
    if sites is not None:
        site = np.asarray(sites).astype(str)
        _need(rows.size == 0 or rows.max() < len(site), "Population/sites misaligned")
        description["site_counts"] = {s: int((site[rows] == s).sum()) for s in sorted(set(site[rows].tolist()))}
        if labels is not None:
            description["site_by_diagnosis"] = {
                s: dict(n_ASD=int(((site[rows] == s) & (y[rows] == 0)).sum()),
                        n_TC=int(((site[rows] == s) & (y[rows] == 1)).sum()))
                for s in description["site_counts"]}
    return description


def filter_roles(roles, eligible, labels=None, sites=None):
    """Filter fixed roles; the caller must refit preprocessing/screening on them."""
    mask = np.asarray(eligible)
    _need(mask.ndim == 1 and mask.dtype == np.bool_, "Eligibility must be a boolean vector")
    checked = validate_roles(roles, len(mask))
    filtered = {role: rows[mask[rows]] for role, rows in checked.items()}
    _need(len(filtered["train"]) > 0, "Subset has no training subjects")
    audit = dict(original_roles_preserved=True, subjects_moved_between_roles=False,
                 subset_requires_new_gate_and_full_subset_F_screen=True,
                 eligible_mask_sha256=_array_sha(mask),
                 roles={r: describe_population(filtered[r], labels, sites) for r in ROLES},
                 role_rows_sha256={r: _array_sha(filtered[r]) for r in ROLES})
    return filtered, audit


def capacity_candidates(strength, direction, roles, seed, mode="shuffled_direction",
                        subject_ids=None, strata=None):
    """Return (strength, pseudo_direction, audit) BEFORE joint 2E F screening.

    No labels are accepted.  ``shuffled_direction`` independently permutes every
    D column within each role (and optionally site), preserving its marginal
    distribution while breaking subject correspondence. ``gaussian_null`` draws
    independent noise using ONLY training D moments (unit scale if D is absent).
    The same whole 2E candidate pool and K/readout rule as RGIC must then be used.
    Neither control is a fitted prediction fusion or a direction estimator.
    """
    scalar = np.asarray(strength, dtype=np.float32)
    _need(scalar.ndim == 2 and scalar.shape[1] > 0 and np.isfinite(scalar).all(), "Strength must be finite N by E")
    n, edges = scalar.shape
    role_rows = validate_roles(roles, n)
    _need(len(role_rows["train"]) > 0, "Training rows are required")
    ids = _ids(subject_ids, n)
    _need(mode in ("shuffled_direction", "gaussian_null"), "Unknown label-blind capacity control")
    source = None if direction is None else np.asarray(direction, dtype=np.float32)
    _need(source is None or (source.shape == scalar.shape and np.isfinite(source).all()), "Direction must match complete strength pool")
    _need(mode != "shuffled_direction" or source is not None, "Shuffled-D requires the full native direction matrix")
    groups = np.repeat("all", n) if strata is None else np.asarray(strata).astype(str)
    _need(groups.shape == (n,), "Randomization strata misaligned")
    pseudo = np.zeros_like(scalar)
    fallback_columns, cells, fixed_permutation_positions = 0, 0, 0
    if mode == "gaussian_null":
        if source is None:
            center, scale = np.zeros(edges), np.ones(edges)
        else:
            train = np.asarray(source[role_rows["train"]], dtype=np.float64)
            center, scale = train.mean(0), train.std(0, ddof=0)
            good = scale > 0
            fallback = float(np.median(scale[good])) if good.any() else 1.0
            fallback_columns = int((~good).sum())
            scale = np.where(good, scale, fallback)
    for role in ROLES:
        for group in sorted(set(groups[role_rows[role]].tolist())):
            rows = role_rows[role][groups[role_rows[role]] == group]
            rows = rows[np.argsort(ids[rows], kind="mergesort")]
            for edge in range(edges):
                random = np.random.RandomState(_seed(seed, "capacity", mode, role, group, edge))
                if mode == "shuffled_direction":
                    perm = random.permutation(len(rows))
                    pseudo[rows, edge] = source[rows[perm], edge]
                    fixed_permutation_positions += int((perm == np.arange(len(rows))).sum())
                else:
                    pseudo[rows, edge] = random.normal(center[edge], scale[edge], len(rows)).astype(np.float32)
                cells += len(rows)
    _need(np.isfinite(pseudo).all(), "Pseudo-direction contains nonfinite values")
    return scalar.copy(), pseudo, dict(control=mode, seed=int(seed), labels_used=False,
        original_strength_sha256=_array_sha(scalar), pseudo_direction_sha256=_array_sha(pseudo),
        n_scalar_candidates=edges, n_pseudo_direction_candidates=edges, full_joint_candidate_count=2 * edges,
        prescreen_scope="caller must apply same complete-label F screen to ALL 2E candidates before K readout",
        role_boundaries_preserved=True, strata="site_or_caller_strata" if strata is not None else "role_only",
        gaussian_moments_fit_scope="train_only" if mode == "gaussian_null" else None,
        gaussian_constant_source_scale_fallback_columns=fallback_columns,
        shuffled_fixed_position_fraction=(float(fixed_permutation_positions / cells) if cells and mode == "shuffled_direction" else None),
        outside_active_roles_filled_zero=True, interpretation="capacity/null-feature control; pseudo channel is not inferred direction")


def build_confound_design(demographics, sites, roles, mean_fd=None,
                          include_four=True, include_site=True, include_fd=False):
    """Train-fitted nuisance design, ordered original four demographics plus site/FD.

    Original hand/sex numerical codes are retained; this is a linear sensitivity
    model, not a causal correction. Missing numeric values use training means,
    as in the original four-demographic preparation. FD<0 is missing. Test-only
    site levels receive the reference encoding and are explicitly counted.
    """
    demo = np.asarray(demographics, dtype=np.float64)
    _need(demo.ndim == 2 and demo.shape[1] == 4, "Demographics must follow SEX,AGE_AT_SCAN,HANDEDNESS_CATEGORY,FIQ")
    # Match the original report_data_matched.numeric sentinel convention.
    demo = np.where(np.isfinite(demo) & (demo > -999), demo, np.nan)
    n = len(demo)
    role_rows = validate_roles(roles, n)
    train = role_rows["train"]
    _need(len(train) > 0, "Nuisance fitting requires training rows")
    columns, blocks, numeric_detail = ["intercept"], [np.ones((n, 1))], {}
    numeric = [(name, demo[:, i]) for i, name in enumerate(DEMOGRAPHIC_COLUMNS)] if include_four else []
    if include_fd:
        fd = np.asarray(mean_fd, dtype=np.float64)
        _need(fd.shape == (n,), "FD covariate must be aligned")
        numeric.append((FD_COLUMN, np.where(fd >= 0, fd, np.nan)))
    for name, raw in numeric:
        finite = np.isfinite(raw)
        observed = raw[train][finite[train]]
        fill = float(observed.mean()) if len(observed) else 0.0
        imputed = np.where(finite, raw, fill)
        center, scale = float(imputed[train].mean()), float(imputed[train].std(ddof=0))
        scale = scale if scale > 0 else 1.0
        blocks.append(((imputed - center) / scale)[:, None])
        columns.append(name)
        numeric_detail[name] = dict(training_mean_fill=fill, training_center=center, training_scale=scale,
                                    n_missing=int((~finite).sum()), all_missing_train=not bool(len(observed)))
    site_detail = None
    if include_site:
        site = np.asarray(sites).astype(str)
        _need(site.shape == (n,), "Site covariate must be aligned")
        levels = sorted(set(site[train].tolist()))
        for level in levels[1:]:
            columns.append("site=" + level)
            blocks.append((site == level).astype(float)[:, None])
        site_detail = dict(training_levels=levels, reference=levels[0],
                           unseen_rows_by_role={r: int((~np.isin(site[ix], levels)).sum()) for r, ix in role_rows.items()},
                           unseen_site_encoding="reference/all-zero indicator; no test-only level is fitted")
    design = np.hstack(blocks)
    _need(np.isfinite(design).all(), "Nuisance design is nonfinite")
    return design, dict(columns=columns, fit_scope="train_only", training_rows_sha256=_array_sha(train),
                        numeric=numeric_detail, site=site_detail, design_sha256=_array_sha(design),
                        labels_used=False, causal_adjustment_claimed=False)


def residualize_connections(values, confound_design, train_rows):
    """OLS nuisance coefficients fit on TRAIN only, applied to every role."""
    x, z = np.asarray(values, dtype=np.float64), np.asarray(confound_design, dtype=np.float64)
    train = np.asarray(train_rows)
    _need(x.ndim == z.ndim == 2 and len(x) == len(z) and np.isfinite(x).all() and np.isfinite(z).all(), "Finite aligned connection/design matrices required")
    _need(train.ndim == 1 and np.issubdtype(train.dtype, np.integer) and len(train) > 0 and
          len(np.unique(train)) == len(train) and np.all((train >= 0) & (train < len(x))), "Invalid nuisance training rows")
    coefficients, _, rank, singular = np.linalg.lstsq(z[train], x[train], rcond=None)
    residual = (x - z @ coefficients).astype(np.float32)
    _need(np.isfinite(residual).all(), "Residualized connection matrix is nonfinite")
    return residual, dict(fit_scope="train_only", method="numpy.linalg.lstsq rcond=None",
                          rank=int(rank), n_design_columns=z.shape[1], n_training_rows=len(train),
                          coefficients_sha256=_array_sha(coefficients), training_rows_sha256=_array_sha(train),
                          residual_sha256=_array_sha(residual), labels_used=False,
                          rank_deficient=bool(rank < z.shape[1]),
                          apply_before_new_full_population_F_screen=True)


def label_permutation(labels, seed, sites=None, subject_ids=None, mode="within_site"):
    """One cohort-level permutation shared across folds/classifiers/methods.

    Within-site permutations preserve site prevalence. The function does not use
    role/fold labels and does not silently redraw an unhelpful permutation.
    Original data labels are not mutated. All-label F selection must be repeated
    with these SAME permuted labels, including the outer test labels.
    """
    original = np.asarray(labels)
    _need(original.ndim == 1, "Permutation labels must be a vector")
    y = _labels(original, len(original))
    ids = _ids(subject_ids, len(y))
    _need(mode in ("within_site", "global"), "Permutation mode must be within_site or global")
    if mode == "within_site":
        _need(sites is not None, "Within-site permutation requires aligned site identities")
        groups = np.asarray(sites).astype(str)
        _need(groups.shape == y.shape, "Permutation sites misaligned")
    else:
        groups = np.repeat("all", len(y))
    permuted, permutation = y.copy(), np.arange(len(y), dtype=np.int64)
    detail = []
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(ids[rows], kind="mergesort")]
        source = rows[np.random.RandomState(_seed(seed, "labels", mode, group)).permutation(len(rows))]
        permuted[rows], permutation[rows] = y[source], source
        detail.append(dict(stratum=group, n=len(rows), n_ASD=int((y[rows] == 0).sum()),
                           n_TC=int((y[rows] == 1).sum()), mixed_labels=bool(len(np.unique(y[rows])) == 2)))
    return permuted, dict(seed=int(seed), mode=mode, original_label_sha256=_array_sha(y),
        permuted_label_sha256=_array_sha(permuted), subject_ids_sha256=_identity(ids.tolist()),
        permutation_index=permutation.tolist(), strata=detail,
        n_changed_labels=int((permuted != y).sum()),
        share_same_permutation_across_all_folds_methods_classifiers=True,
        redo_joint_F_with_permuted_train_valid_test_labels=True,
        labels_used_for_grouping=False, original_labels_mutated=False,
        interpretation="null for the disclosed all-label procedure, not an independent-held-out-generalization test")


def permutation_contract(observed, permuted):
    """Require equal actual budgets; distinguish fixed-param and full-search nulls.

    The caller must supply the actual run contract for original and permuted
    labels. This helper cannot certify that a search was actually executed.
    """
    required = ("classifier", "scope", "folds", "feature_prescreen", "selection_rule", "parameter_policy",
                "search_budget", "epochs", "seed_policy")
    _need(all(k in observed and k in permuted for k in required), "Permutation budget contract is incomplete")
    _need(all(observed[k] == permuted[k] for k in required), "Observed and permuted-label procedures/budgets differ")
    _need(observed["feature_prescreen"] == "recomputed_with_current_train_valid_test_labels", "Permutation must recompute the full-label F screen")
    policy = observed["parameter_policy"]
    _need(policy in ("locked_parameters_refit", "same_budget_reselect_and_refit"), "Unknown permutation parameter policy")
    if policy == "same_budget_reselect_and_refit":
        _need(isinstance(observed["search_budget"], dict) and observed["search_budget"] and
              all(isinstance(v, (int, np.integer)) and int(v) >= 0 for v in observed["search_budget"].values()) and
              any(int(v) > 0 for v in observed["search_budget"].values()), "Full-search permutation needs an explicit positive budget")
    return dict(matched_contract_sha256=_identity({k: observed[k] for k in required}),
                full_reselected_pipeline=policy == "same_budget_reselect_and_refit",
                experiment_label=("equal-budget reselected full-pipeline permutation" if policy == "same_budget_reselect_and_refit"
                                  else "locked-parameter all-label re-screen-and-refit permutation pilot"),
                execution_verified=False,
                cannot_label_fixed_parameter_pilot_as_full_reselected_search=True)


def permutation_pvalue(observed, null_values, alternative="greater", null_center=None):
    """Finite Monte Carlo (+1)/(B+1); never discard failed/nonfinite null runs."""
    value, null = float(observed), np.asarray(null_values, dtype=np.float64)
    _need(np.isfinite(value) and null.ndim == 1 and len(null) > 0 and np.isfinite(null).all(), "All predeclared permutation runs must be finite; report failures separately")
    _need(alternative in ("greater", "less", "two-sided"), "Unknown permutation alternative")
    if alternative == "greater":
        exceed = int((null >= value).sum())
    elif alternative == "less":
        exceed = int((null <= value).sum())
    else:
        _need(null_center is not None and np.isfinite(float(null_center)), "A two-sided test requires a predeclared finite null center")
        exceed = int((np.abs(null - float(null_center)) >= abs(value - float(null_center))).sum())
    return dict(pvalue=float((exceed + 1) / (len(null) + 1)), n_permutations=int(len(null)),
                n_at_least_as_extreme=exceed, alternative=alternative,
                minimum_attainable_pvalue=float(1 / (len(null) + 1)),
                method="Monte Carlo permutation p=(1+extreme)/(B+1)",
                predeclared_null_center=float(null_center) if alternative == "two-sided" else None)
