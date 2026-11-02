"""Deterministic parameter proposals; this module never fits or scores models.

All feature proposals retain the complete-label joint-preselection requirement.
Historical records supply PARAMETER seeds only, never selected columns, models,
probabilities, or test scores.  Direction beta is strictly positive and belongs
to the model layer, after its final feature scaling.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
PART4 = ROOT.parent
HISTORICAL = PART4 / "RGIC广泛参数搜索_20261027/adaptive_stage2/selection/real/svm"
ATLASES = ("aal", "cc200", "ez")
BUDGETS = dict(aal=1000, cc200=3000, ez=1000)
SCALAR_POOLS = dict(aal=6670, cc200=19900, ez=6670)
DEFAULT_CORE = dict(gamma=.2, cap=.9, lambda_cost=1., percentile=95., alpha=1., delta=0.)
CORE_AXES = dict(
    gamma=[0., .005, .02, .05, .1, .2, .4, .8, 1.6, 3.2, 6.4, 12.8],
    cap=[.2, .4, .6, .8, .9, .95, .99, .995, .999, 1.],
    lambda_cost=[0., .001, .01, .03, .1, .3, 1., 3., 10., 20., 40.],
    percentile=[.625, 2.5, 5., 25., 50., 65., 75., 85., 90., 95., 99., 99.5, 100.],
    alpha=[0., .0001, .0003, .001, .003, .01, .03, .1, .3, .6, .8, .95, .99, 1.],
    delta=[0., .005, .01, .03, .06, .1],
)
QUOTAS = [0., .05, .1, .15, .25, .4, .5, .75, 1.]
PRESELECT_FACTORS = [1., 2., 4., "all"]
BETAS = [.01, .03, .1, .3, 1., 3., 10., 30.]
GAMMA_FACTORS = [.01, .1, 1., 10., 100.]
SCALERS = ["standard", "robust", "none"]
DEMOGRAPHICS = ["four", "sex_age", "none"]
VERSION = "joint-first-svm-lr-deterministic-space-v1"


def _check_unit(atlas, window=None, fold=None):
    if atlas not in ATLASES:
        raise ValueError("Unknown atlas")
    if window is not None and int(window) not in (1, 2):
        raise ValueError("Unknown lag window")
    if fold is not None and int(fold) not in range(10):
        raise ValueError("Unknown original fold")


def _key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def canonical_candidate(candidate):
    """Stable JSON key; the caller adds kind/defaults before requesting a key."""
    return _key(candidate)


def candidate_id(candidate):
    return hashlib.sha256(canonical_candidate(candidate).encode("utf-8")).hexdigest()[:24]


def anchor_sources(atlas, window, fold):
    _check_unit(atlas, window, fold)
    path = HISTORICAL / atlas / ("W%d" % int(window)) / str(int(fold)) / "selection.json"
    return ([] if not path.is_file() else
            [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  parameter_fields_only=["chosen_pure.spec", "chosen_scalar_pure.spec"])])


def _feature(core, atlas, *, k=None, readout="joint_topk", quota=.25, factor=1.):
    return dict(core=dict(DEFAULT_CORE, **core), K=BUDGETS[atlas] if k is None else int(k),
                readout=readout, quota=float(quota), preselect_factor=factor)


def _candidate(features, *, beta=1.):
    return dict(features=copy.deepcopy(features),
                inputs=dict(scaler="standard", demographics="four"),
                model=dict(C=1., class_weight=None, beta=float(beta)))


def _unique(values):
    seen = set()
    result = []
    for value in values:
        identity = _key(value)
        if identity not in seen:
            seen.add(identity)
            result.append(value)
    return result


def anchors(atlas, window, fold):
    """Three mandatory anchors, then same-fold historical pure parameter seeds."""
    _check_unit(atlas, window, fold)
    result = [
        _candidate(_feature({}, atlas)),
        _candidate(_feature({}, atlas, readout="joint_then_quota", quota=.25, factor="all")),
        _candidate(_feature(dict(alpha=0.), atlas)),
    ]
    path = HISTORICAL / atlas / ("W%d" % int(window)) / str(int(fold)) / "selection.json"
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        for version in ("chosen_pure", "chosen_scalar_pure"):
            chosen = record.get(version)
            if not isinstance(chosen, dict) or not isinstance(chosen.get("spec"), dict):
                continue
            # Deliberately never read validation or test score fields here.
            spec = chosen["spec"]
            core = {name: float(spec.get(name, DEFAULT_CORE[name])) for name in DEFAULT_CORE}
            if spec.get("readout") == "scalar":
                features = _feature(core, atlas, readout="joint_then_quota", quota=0., factor="all")
            else:
                features = _feature(core, atlas)
            beta = float(spec.get("beta", 1.))
            result.append(_candidate(features, beta=beta if beta > 0 else 1.))
    return _unique(result)


def _k_axis(atlas):
    base, maximum = BUDGETS[atlas], SCALAR_POOLS[atlas]
    return sorted({max(1, min(maximum, int(round(base * multiplier))))
                   for multiplier in (.25, .5, 1., 1.5, 2., 4.)} | {maximum})


def proxy_candidates(atlas, window, fold, count=512, seed=None):
    """Yield ``count`` deterministic stratified-random feature proposals.

    This stdlib implementation does not silently change between Sobol and random
    sampling according to optional installed packages.  Mandatory anchors and
    axis boundaries precede randomized coverage.  No model fits occur here.
    """
    _check_unit(atlas, window, fold)
    count = int(count)
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return
    if seed is None:
        seed = 20261027 + ATLASES.index(atlas) * 1000 + int(window) * 100 + int(fold)
    rng = random.Random(int(seed))
    forced = [entry["features"] for entry in anchors(atlas, window, fold)]
    for name, values in CORE_AXES.items():
        for boundary in (values[0], values[-1]):
            forced.append(_feature({name: boundary}, atlas))
    for boundary in (_k_axis(atlas)[0], _k_axis(atlas)[-1]):
        forced.append(_feature({}, atlas, k=boundary))
    for quota in (0., .25, 1.):
        for factor in PRESELECT_FACTORS:
            forced.append(_feature({}, atlas, readout="joint_then_quota", quota=quota, factor=factor))
    seen = set()
    emitted = 0
    for features in _unique(forced):
        if emitted >= count:
            return
        identity = _key(features)
        seen.add(identity)
        emitted += 1
        yield copy.deepcopy(features)
    dimensions = list(CORE_AXES.values()) + [_k_axis(atlas), ["joint_topk", "joint_then_quota"],
                                            QUOTAS, PRESELECT_FACTORS]
    names = list(CORE_AXES)
    strata = []
    for unused in dimensions:
        shuffled = list(range(max(count, 1)))
        rng.shuffle(shuffled)
        strata.append(shuffled)
    draw = 0
    while emitted < count:
        picks = []
        for dim, values in enumerate(dimensions):
            if draw < count:
                u = (strata[dim][draw] + rng.random()) / count
            else:
                u = rng.random()
            picks.append(values[min(len(values) - 1, int(u * len(values)))])
        draw += 1
        core = dict(zip(names, picks[:len(names)]))
        k, readout, quota, factor = picks[len(names):]
        if readout == "joint_topk":
            quota, factor = .25, 1.  # Canonicalize inactive readout parameters.
        features = _feature(core, atlas, k=k, readout=readout, quota=quota, factor=factor)
        identity = _key(features)
        if identity in seen:
            continue
        seen.add(identity)
        emitted += 1
        yield features


def model_grid(kind):
    """70 RBF-SVM configurations or 34 LR configurations; beta starts at 1."""
    kind = str(kind).lower()
    if kind in ("svm", "rbf-svm", "rbf_svm"):
        return [dict(C=10. ** exponent, gamma_factor=factor, class_weight=weight, beta=1.)
                for exponent in range(-3, 4) for factor in GAMMA_FACTORS for weight in (None, "balanced")]
    if kind in ("lr", "logreg", "logistic", "logistic_regression"):
        return [dict(C=10. ** exponent, penalty=penalty, class_weight=weight, beta=1.)
                for penalty, exponents in (("l2", range(-6, 4)), ("l1", range(-4, 3)))
                for exponent in exponents for weight in (None, "balanced")]
    raise ValueError("Only SVM and logistic regression are authorized")


def _normalize_candidate(candidate, atlas):
    result = copy.deepcopy(candidate)
    features = result.setdefault("features", {})
    core = dict(DEFAULT_CORE, **features.get("core", {}))
    for name in DEFAULT_CORE:
        core[name] = float(features.pop(name, core[name]))
        if not math.isfinite(core[name]):
            raise ValueError("Nonfinite core parameter")
    features["core"] = core
    features["K"] = int(features.get("K", BUDGETS[atlas]))
    features.setdefault("readout", "joint_topk")
    features.setdefault("quota", .25)
    features.setdefault("preselect_factor", 1.)
    if not 1 <= features["K"] <= SCALAR_POOLS[atlas]:
        raise ValueError("K outside matched native scalar pool")
    if not (core["gamma"] >= 0 and core["lambda_cost"] >= 0 and core["delta"] >= 0 and
            0 < core["cap"] <= 1 and 0 < core["percentile"] <= 100 and 0 <= core["alpha"] <= 1):
        raise ValueError("Core parameter outside natural domain")
    if features["readout"] not in ("joint_topk", "joint_then_quota") or not 0 <= features["quota"] <= 1:
        raise ValueError("Only joint-first readouts are allowed")
    inputs = result.setdefault("inputs", {})
    inputs.setdefault("scaler", "standard")
    inputs.setdefault("demographics", "four")
    if inputs["scaler"] not in SCALERS or inputs["demographics"] not in DEMOGRAPHICS:
        raise ValueError("Unknown scaler/demographics proposal")
    model = result.setdefault("model", {})
    model.setdefault("C", 1.)
    model.setdefault("class_weight", None)
    model.setdefault("beta", 1.)
    if not all(math.isfinite(float(model[key])) and float(model[key]) > 0 for key in ("C", "beta")):
        raise ValueError("C and beta must be finite and positive")
    return result


def _initial_state(atlas):
    axes = copy.deepcopy(CORE_AXES)
    axes.update(K=_k_axis(atlas), quota=list(QUOTAS), beta=list(BETAS),
                gamma_factor=list(GAMMA_FACTORS),
                model_C_svm=[10. ** i for i in range(-3, 4)],
                model_C_lr_l2=[10. ** i for i in range(-6, 4)],
                model_C_lr_l1=[10. ** i for i in range(-4, 3)])
    return dict(version=VERSION, atlas=atlas, round=0, axes=axes, expanded_bounds=[],
                natural_bounds=dict(gamma=[0, None], cap=[0, 1], lambda_cost=[0, None],
                                    percentile=[0, 100], alpha=[0, 1], delta=[0, None],
                                    K=[1, SCALAR_POOLS[atlas]], quota=[0, 1], beta=[0, None]))


def _numeric_neighbors(axis, current, *, lower=None, upper=None, lower_open=False,
                       multiplier=3., logarithmic=True, integer=False):
    values = sorted(set(float(v) for v in axis) | {float(current)})
    events = []
    proposals = []
    for side, edge in (("lower", values[0]), ("upper", values[-1])):
        if current != edge:
            continue
        hard = ((side == "lower" and lower is not None and current <= lower) or
                (side == "upper" and upper is not None and current >= upper))
        if hard:
            events.append(dict(side=side, event="hard_boundary", value=current))
            continue
        if side == "lower":
            extra = current / multiplier
            if lower is not None:
                extra = max(lower, extra)
        else:
            extra = current * multiplier if current > 0 else 1.
            if upper is not None:
                extra = min(upper, extra)
        if integer:
            extra = int(math.floor(extra)) if side == "lower" else int(math.ceil(extra))
        valid = math.isfinite(extra) and extra != current
        if lower is not None:
            valid = valid and (extra > lower if lower_open else extra >= lower)
        if upper is not None:
            valid = valid and extra <= upper
        if valid:
            values.append(float(extra))
            values.sort()
            events.append(dict(side=side, event="soft_boundary_expanded", old_boundary=current,
                               new_boundary=extra, multiplier=multiplier))
        else:
            events.append(dict(side=side, event="numerical_or_integer_boundary", value=current))
    position = values.index(float(current))
    for neighbor_position in (position - 1, position + 1):
        if 0 <= neighbor_position < len(values):
            neighbor = values[neighbor_position]
            midpoint = (math.exp((math.log(current) + math.log(neighbor)) / 2.)
                        if logarithmic and current > 0 and neighbor > 0
                        else (current + neighbor) / 2.)
            for proposed in (neighbor, midpoint):
                if integer:
                    proposed = int(round(proposed))
                if math.isfinite(proposed) and proposed != current and (lower is None or (proposed > lower if lower_open else proposed >= lower)):
                    if upper is None or proposed <= upper:
                        proposals.append(proposed)
    return sorted(set(proposals)), sorted(set(values)), events


def neighbors(candidate, atlas, space_state=None):
    """Coordinate neighbors plus explicit boundary-expansion audit information.

    The returned state is a copy.  Reusing it within the same search unit retains
    every expanded numeric bound; it must not be shared between unrelated units.
    Approximately 50 proposals are returned, before model/feature deduplication.
    """
    _check_unit(atlas)
    current = _normalize_candidate(candidate, atlas)
    state = _initial_state(atlas) if space_state is None else copy.deepcopy(space_state)
    if state.get("version") != VERSION or state.get("atlas") != atlas:
        raise ValueError("Search-space state belongs to another schema/atlas")
    state["round"] += 1
    proposals, log = [], []

    def numeric(name, value, setter, **bounds):
        options, expanded_axis, events = _numeric_neighbors(state["axes"][name], value, **bounds)
        state["axes"][name] = expanded_axis
        for event in events:
            item = dict(round=state["round"], axis=name, **event)
            log.append(item)
            if event["event"] == "soft_boundary_expanded":
                state["expanded_bounds"].append(item)
        for option in options:
            proposal = copy.deepcopy(current)
            setter(proposal, option)
            proposals.append(proposal)

    for name in DEFAULT_CORE:
        lower, upper = 0., None
        lower_open = name in ("cap", "percentile")
        if name in ("alpha", "cap"):
            upper = 1.
        if name == "percentile":
            upper = 100.
        numeric(name, current["features"]["core"][name],
                lambda c, v, n=name: c["features"]["core"].__setitem__(n, v),
                lower=lower, upper=upper, lower_open=lower_open,
                logarithmic=name in ("gamma", "lambda_cost", "alpha", "delta"))
    numeric("K", current["features"]["K"], lambda c, v: c["features"].__setitem__("K", int(v)),
            lower=1, upper=SCALAR_POOLS[atlas], integer=True, logarithmic=False)
    if current["features"]["readout"] == "joint_then_quota":
        numeric("quota", current["features"]["quota"], lambda c, v: c["features"].__setitem__("quota", v),
                lower=0., upper=1., logarithmic=False)
    for factor in PRESELECT_FACTORS:
        if factor != current["features"]["preselect_factor"]:
            proposal = copy.deepcopy(current)
            proposal["features"].update(readout="joint_then_quota", preselect_factor=factor)
            proposals.append(proposal)
    proposal = copy.deepcopy(current)
    if current["features"]["readout"] == "joint_topk":
        proposal["features"].update(readout="joint_then_quota", preselect_factor="all")
    else:
        proposal["features"].update(readout="joint_topk", preselect_factor=1., quota=.25)
    proposals.append(proposal)
    for field, values in (("scaler", SCALERS), ("demographics", DEMOGRAPHICS)):
        for value in values:
            if value != current["inputs"][field]:
                proposal = copy.deepcopy(current)
                proposal["inputs"][field] = value
                proposals.append(proposal)
    model = current["model"]
    declared_kind = str(model.get("kind", current.get("kind", ""))).lower()
    is_lr = declared_kind in ("lr", "logreg", "logistic", "logistic_regression") or (
        not declared_kind and "penalty" in model and "gamma_factor" not in model)
    penalty = model.get("penalty", "l2")
    c_axis = "model_C_lr_" + penalty if is_lr else "model_C_svm"
    if c_axis not in state["axes"]:
        raise ValueError("Unsupported logistic penalty")
    numeric(c_axis, float(model["C"]), lambda c, v: c["model"].__setitem__("C", v),
            lower=0., lower_open=True, multiplier=10.)
    numeric("beta", float(model["beta"]), lambda c, v: c["model"].__setitem__("beta", v),
            lower=0., lower_open=True)
    if not is_lr:
        numeric("gamma_factor", float(model.get("gamma_factor", 1.)),
                lambda c, v: c["model"].__setitem__("gamma_factor", v),
                lower=0., lower_open=True, multiplier=10.)
    else:
        proposal = copy.deepcopy(current)
        proposal["model"]["penalty"] = "l1" if penalty == "l2" else "l2"
        proposals.append(proposal)
    proposal = copy.deepcopy(current)
    proposal["model"]["class_weight"] = None if model["class_weight"] == "balanced" else "balanced"
    proposals.append(proposal)
    result = [proposal for proposal in _unique(proposals) if _key(proposal) != _key(current)]
    log.append(dict(round=state["round"], event="coordinate_proposals", n_candidates=len(result),
                    joint_first_only=True, beta_strictly_positive=True))
    return result, log, state
