"""Small CPU classifiers for the joint-label-screened SVM/LR experiment.

Feature selection is external to this module.  This module fits its optional
input scaler on training rows only and applies direction beta AFTER scaling.
The SVM is deliberately uncalibrated: its score is an ASD-oriented decision
margin, while prediction always retains the native ASD=0 / TC=1 labels.
No held-out labels are accepted by a fitted estimator or input transform.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
import time
import warnings

# The bundled Windows Python can need these before importing NumPy / sklearn.
_dll_handles = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for _directory in (r"D:\miniconda\Library\bin", r"D:\miniconda\DLLs"):
        if os.path.isdir(_directory):
            _dll_handles.append(os.add_dll_directory(_directory))
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

FORMAT = "joint-svm-logistic-model-v1"
TRANSFORM_FORMAT = "train-only-input-transform-v1"
DEMO_NAMES = ("SEX", "AGE_AT_SCAN", "HANDEDNESS_CATEGORY", "FIQ")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(values):
    """Hash a numeric array including dtype and shape, never an object buffer."""
    values = np.ascontiguousarray(values)
    _need(not values.dtype.hasobject, "Object arrays cannot be hashed")
    h = hashlib.sha256()
    h.update(str(values.dtype).encode("ascii"))
    h.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    h.update(memoryview(values).cast("B"))
    return h.hexdigest()


def _matrix(values, name, width=None, allow_missing=False):
    values = np.asarray(values, dtype=np.float64, order="C")
    _need(values.ndim == 2 and values.shape[0] > 0, name + " must be a nonempty 2D matrix")
    if width is not None:
        _need(values.shape[1] == width, name + " has a different feature width")
    if not allow_missing:
        _need(np.isfinite(values).all(), name + " contains nonfinite values")
    return values


def _get_demo(view, role):
    for key in ("demos_" + role, "demo_" + role):
        if key in view:
            return view[key]
    return None


def _demo_mode(mode):
    mode = str(mode).lower()
    if mode in ("sex_age", "age_sex"):
        return "sex_age"
    _need(mode in ("four", "none"), "demographics must be four, sex_age/age_sex, or none")
    return mode


def _demographic_matrix(values, rows, mode, source_width=None):
    if mode == "none":
        return np.empty((rows, 0), dtype=np.float64), 0
    _need(values is not None, "Requested demographic inputs are missing")
    values = _matrix(values, "demographics", allow_missing=True)
    _need(values.shape[0] == rows, "Demographic and connection row counts differ")
    if source_width is not None:
        _need(values.shape[1] == source_width, "Demographic source width changed after fit")
    _need(values.shape[1] == 4 if mode == "four" else values.shape[1] in (2, 4),
          "Demographics must be ordered SEX, AGE, HAND, FIQ (sex_age accepts the first two)")
    count = 4 if mode == "four" else 2
    return values[:, :count].copy(), int(values.shape[1])


def prepare_inputs(view, settings):
    """Fit a train-only transform and return a/z/(optional x), mask, transform.

    ``view`` uses X_train, X_valid, optional X_test, demos_train / demos_valid /
    demos_test (demo_* are accepted aliases), and an optional direction_mask.
    The demographic column order is SEX, AGE, HAND, FIQ.  ``settings`` contains
    scaler='standard'|'robust'|'none' and demographics='four'|'sex_age'|'none'.
    ``age_sex`` is an alias for sex_age; it does NOT reverse the column order.
    transform is entirely JSON-serializable and can be stored in a model bundle.
    """
    a_raw = _matrix(view["X_train"], "X_train")
    _need(a_raw.shape[1] > 0, "No connection features were supplied")
    mode = _demo_mode(settings.get("demographics", "four"))
    scaler_kind = str(settings.get("scaler", "standard")).lower()
    _need(scaler_kind in ("standard", "robust", "none"), "Unknown scaler")
    demo, source_width = _demographic_matrix(_get_demo(view, "train"), len(a_raw), mode)
    if demo.shape[1]:
        observed = np.isfinite(demo)
        counts = observed.sum(axis=0)
        fills = np.divide(np.where(observed, demo, 0.).sum(axis=0), counts,
                          out=np.zeros(demo.shape[1], dtype=np.float64), where=counts > 0)
        demo = np.where(observed, demo, fills)
    else:
        fills = np.empty(0, dtype=np.float64)
        counts = np.empty(0, dtype=np.int64)
    merged = np.ascontiguousarray(np.concatenate((a_raw, demo), axis=1))
    with threadpool_limits(limits=1):
        if scaler_kind == "standard":
            scaler = StandardScaler().fit(merged)
            center, scale = scaler.mean_, scaler.scale_
        elif scaler_kind == "robust":
            scaler = RobustScaler(quantile_range=(25., 75.), unit_variance=False).fit(merged)
            center, scale = scaler.center_, scaler.scale_
        else:
            center, scale = np.zeros(merged.shape[1]), np.ones(merged.shape[1])
    raw_mask = np.asarray(view.get("direction_mask", np.zeros(a_raw.shape[1], dtype=bool)), dtype=bool)
    _need(raw_mask.shape == (a_raw.shape[1],), "direction_mask does not match connection width")
    mask = np.concatenate((raw_mask, np.zeros(demo.shape[1], dtype=bool)))
    transform = dict(
        format=TRANSFORM_FORMAT, fit_scope="train", n_connection_features=int(a_raw.shape[1]),
        n_output_features=int(merged.shape[1]), demographics=mode,
        demographic_source_width=source_width, demographic_columns=list(DEMO_NAMES[:demo.shape[1]]),
        demographic_indices=list(range(demo.shape[1])), demographic_fill_values=fills.tolist(),
        demographic_training_observed_counts=counts.tolist(), all_missing_demographic_fill=0.,
        scaler=scaler_kind, center=np.asarray(center).tolist(), scale=np.asarray(scale).tolist(),
        robust_quantile_range=([25., 75.] if scaler_kind == "robust" else None),
        direction_mask=mask.tolist(), beta_applied=False,
        beta_policy="multiply_direction_columns_after_final_scaling_in_estimator_only",
        input_training_sha256=array_sha(merged),
    )
    result = dict(a=apply_transform(a_raw, _get_demo(view, "train"), transform),
                  z=apply_transform(view["X_valid"], _get_demo(view, "valid"), transform),
                  direction_mask=mask, transform=transform)
    if "X_test" in view:
        result["x"] = apply_transform(view["X_test"], _get_demo(view, "test"), transform)
    return result


def apply_transform(Xraw, demos, transform):
    """Apply a saved train-only input transform; never fit or apply beta here."""
    _need(transform.get("format") == TRANSFORM_FORMAT, "Unsupported input-transform format")
    values = _matrix(Xraw, "connections", int(transform["n_connection_features"]))
    mode = _demo_mode(transform["demographics"])
    demographic, _ = _demographic_matrix(demos, len(values), mode,
                                         int(transform["demographic_source_width"]))
    fills = np.asarray(transform["demographic_fill_values"], dtype=np.float64)
    if demographic.shape[1]:
        _need(fills.shape == (demographic.shape[1],), "Invalid demographic fill values")
        demographic = np.where(np.isfinite(demographic), demographic, fills)
    merged = np.concatenate((values, demographic), axis=1)
    center = np.asarray(transform["center"], dtype=np.float64)
    scale = np.asarray(transform["scale"], dtype=np.float64)
    _need(center.shape == scale.shape == (merged.shape[1],), "Invalid input-transform dimensions")
    _need(np.isfinite(center).all() and np.isfinite(scale).all() and np.all(scale > 0),
          "Invalid scaler center or scale")
    result = np.ascontiguousarray((merged - center) / scale, dtype=np.float64)
    _need(np.isfinite(result).all(), "Transformed inputs contain nonfinite values")
    return result


class FitFailure(RuntimeError):
    """A candidate that must be logged as failed, never selected as converged."""
    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata


def _canonical_config(config):
    supplied = dict(config)
    kind = str(supplied.get("kind", "svm")).lower()
    kind = {"lr": "logistic", "logreg": "logistic", "logistic_regression": "logistic",
            "rbf": "svm", "rbf_svm": "svm"}.get(kind, kind)
    _need(kind in ("svm", "logistic"), "Only svm and logistic classifiers are supported")
    C = float(supplied.get("C", 1.))
    beta = float(supplied.get("beta", 1.))
    factor = float(supplied.get("gamma_factor", 1.))
    _need(math.isfinite(C) and C > 0, "C must be finite and positive")
    _need(math.isfinite(beta) and beta >= 0, "beta must be finite and nonnegative")
    _need(math.isfinite(factor) and factor > 0, "gamma_factor must be finite and positive")
    class_weight = supplied.get("class_weight", None)
    _need(class_weight is None or class_weight == "balanced" or isinstance(class_weight, dict),
          "class_weight must be None, balanced, or a dictionary")
    if isinstance(class_weight, dict):
        class_weight = {int(k): float(v) for k, v in class_weight.items()}
        _need(set(class_weight).issubset({0, 1}) and all(math.isfinite(v) and v > 0 for v in class_weight.values()),
              "Invalid binary class weights")
    penalty = str(supplied.get("penalty", "l2")).lower()
    _need(penalty in ("l1", "l2"), "Logistic penalty must be l1 or l2")
    return dict(kind=kind, C=C, gamma_factor=factor, class_weight=class_weight,
                penalty=penalty, beta=beta)


def _squared_distances(left, right):
    if left.shape[1] == 0:
        return np.zeros((len(left), len(right)), dtype=np.float64)
    values = euclidean_distances(left, right, squared=True)
    np.maximum(values, 0., out=values)
    _need(np.isfinite(values).all(), "Squared distances overflowed")
    return values


def _oriented_outputs(estimator, values, positive_label, with_probability):
    classes = np.asarray(estimator.classes_)
    _need(classes.shape == (2,) and np.array_equal(classes, [0, 1]), "Native classes must be ASD0 / TC1")
    margin = np.asarray(estimator.decision_function(values), dtype=np.float64).reshape(-1)
    output = dict(score=margin if positive_label == int(classes[1]) else -margin,
                  prediction=np.asarray(estimator.predict(values), dtype=np.int64))
    if with_probability:
        index = int(np.flatnonzero(classes == positive_label)[0])
        output["probability"] = np.asarray(estimator.predict_proba(values)[:, index], dtype=np.float64)
    _need(np.isfinite(output["score"]).all(), "Nonfinite decision margins")
    return output


class PreparedEvaluator:
    """Reuse prepared train/validation inputs and RBF component distances.

    fit(config) returns model (a persistable bundle), score, prediction,
    metadata, and (for logistic only) positive-label probability.  ``z`` is the
    validation feature matrix, never validation labels.  No test data are hashed
    or inspected here.  Cached matrices are bounded to one weighted input pair
    and one kernel pair, in addition to the two channel-distance pairs.
    """
    def __init__(self, a, y, z, seed=42, positive_label=0, direction_mask=None):
        self.a = _matrix(a, "a").copy(order="C")
        self.z = _matrix(z, "z", self.a.shape[1]).copy(order="C")
        _need(self.a.shape[1] > 0, "No prepared features")
        raw_y = np.asarray(y)
        _need(raw_y.ndim == 1 and len(raw_y) == len(self.a), "Training y has an invalid shape")
        _need(np.isfinite(raw_y).all() and np.array_equal(np.unique(raw_y), [0, 1]),
              "Training y must contain both native binary labels ASD0 and TC1")
        self.y = raw_y.astype(np.int64, copy=True)
        _need(positive_label in (0, 1), "positive_label must be 0 or 1")
        self.positive_label = int(positive_label)
        self.seed = int(seed)
        self.direction_mask = np.asarray(direction_mask if direction_mask is not None else
                                         np.zeros(self.a.shape[1], dtype=bool), dtype=bool).copy()
        _need(self.direction_mask.shape == (self.a.shape[1],), "Prepared direction_mask width mismatch")
        self.a.setflags(write=False)
        self.z.setflags(write=False)
        self.y.setflags(write=False)
        self.direction_mask.setflags(write=False)
        self._input_hash = dict(train=array_sha(self.a), valid=array_sha(self.z),
                                train_labels=array_sha(self.y), direction_mask=array_sha(self.direction_mask))
        self._distance_cache = None
        self._weighted_cache = None
        self._kernel_cache = None

    def _weighted(self, beta):
        if self._weighted_cache is None or self._weighted_cache[0] != beta:
            weights = np.where(self.direction_mask, beta, 1.)
            a = np.ascontiguousarray(self.a * weights)
            z = np.ascontiguousarray(self.z * weights)
            _need(np.isfinite(a).all() and np.isfinite(z).all(), "Direction beta overflowed inputs")
            self._weighted_cache = (beta, a, z)
        return self._weighted_cache[1:]

    def _distances(self):
        if self._distance_cache is None:
            scalar = ~self.direction_mask
            dsa = _squared_distances(self.a[:, scalar], self.a[:, scalar])
            dsz = _squared_distances(self.z[:, scalar], self.a[:, scalar])
            dda = _squared_distances(self.a[:, self.direction_mask], self.a[:, self.direction_mask])
            ddz = _squared_distances(self.z[:, self.direction_mask], self.a[:, self.direction_mask])
            np.fill_diagonal(dsa, 0.)
            np.fill_diagonal(dda, 0.)
            self._distance_cache = (dsa, dsz, dda, ddz)
        return self._distance_cache

    def _kernels(self, beta, gamma):
        key = (beta, gamma)
        if self._kernel_cache is None or self._kernel_cache[0] != key:
            dsa, dsz, dda, ddz = self._distances()
            train = np.exp(-gamma * (dsa + beta * beta * dda))
            valid = np.exp(-gamma * (dsz + beta * beta * ddz))
            np.fill_diagonal(train, 1.)
            self._kernel_cache = (key, train, valid)
        return self._kernel_cache[1:]

    def fit(self, config):
        settings = _canonical_config(config)
        started = time.perf_counter()
        attempts = []
        with threadpool_limits(limits=1):
            weighted_a, weighted_z = self._weighted(settings["beta"])
            metadata = dict(
                format=FORMAT, config=settings, supplied_config=dict(config), seed=self.seed,
                positive_label=self.positive_label, native_label_semantics={"ASD": 0, "TC": 1},
                score="positive_label_oriented_decision_margin", prediction="native_estimator_predict",
                probability_calibration=False, n_train=len(self.a), n_valid=len(self.z),
                n_features=self.a.shape[1], n_direction_features=int(self.direction_mask.sum()),
                beta_applied_after_scaler=True, feature_sha256=self._input_hash.copy(),
                numerical_threads=1, numpy_version=np.__version__, sklearn_version=sklearn.__version__)
            if settings["kind"] == "svm":
                variance = float(np.var(weighted_a))
                _need(math.isfinite(variance), "Weighted training variance overflowed")
                denominator = float(weighted_a.shape[1] * variance)
                gamma = settings["gamma_factor"] / denominator if denominator > 0 else settings["gamma_factor"]
                _need(math.isfinite(gamma) and gamma > 0, "Resolved RBF gamma is invalid")
                train, valid = self._kernels(settings["beta"], gamma)
                estimator = SVC(C=settings["C"], kernel="precomputed", probability=False,
                                class_weight=settings["class_weight"], cache_size=64,
                                tol=1e-3, max_iter=-1, random_state=self.seed)
                estimator.fit(train, self.y)
                output = _oriented_outputs(estimator, valid, self.positive_label, False)
                metadata.update(solver="libsvm_precomputed_rbf", cache_size_mb=64,
                                gamma_factor=settings["gamma_factor"], resolved_gamma=gamma,
                                weighted_train_variance=variance, gamma_denominator=denominator,
                                zero_variance_gamma_fallback=(denominator == 0),
                                gamma_definition="gamma_factor/(p*Var(betaweighted_training_A)); zeroVar: gamma_factor",
                                fit_status=int(estimator.fit_status_), n_iter=estimator.n_iter_.tolist(),
                                support_vectors=int(len(estimator.support_)))
                if estimator.fit_status_ != 0:
                    raise FitFailure("SVM did not converge", metadata)
                bundle = dict(format=FORMAT, kind="svm", estimator=estimator,
                              train_weighted_A=weighted_a,
                              train_direction_unweighted=np.ascontiguousarray(self.a[:, self.direction_mask]),
                              resolved_gamma=gamma)
            else:
                estimator = None
                for limit in (5000, 20000):
                    candidate = LogisticRegression(
                        C=settings["C"], penalty=settings["penalty"], solver="liblinear",
                        dual=False, class_weight=settings["class_weight"], tol=1e-4,
                        max_iter=limit, random_state=self.seed, fit_intercept=True,
                        intercept_scaling=1.)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        candidate.fit(weighted_a, self.y)
                    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
                    reached_limit = bool(np.any(candidate.n_iter_ >= limit))
                    attempts.append(dict(max_iter=limit, n_iter=candidate.n_iter_.tolist(),
                                         convergence_warnings=convergence,
                                         other_warnings=[str(item.message) for item in caught
                                                         if not issubclass(item.category, ConvergenceWarning)],
                                         converged=not convergence and not reached_limit))
                    if not convergence and not reached_limit:
                        estimator = candidate
                        break
                metadata.update(solver="liblinear", tolerance=1e-4, fit_intercept=True,
                                intercept_scaling=1., attempts=attempts, retries=len(attempts) - 1,
                                resolved_gamma=None, gamma_factor_ignored_for_logistic=True)
                if estimator is None:
                    metadata.update(fit_status="failed_convergence", elapsed_seconds=time.perf_counter() - started)
                    raise FitFailure("Logistic regression did not converge after the 20000-iteration retry", metadata)
                output = _oriented_outputs(estimator, weighted_z, self.positive_label, True)
                metadata.update(fit_status=0, n_iter=estimator.n_iter_.tolist())
                bundle = dict(format=FORMAT, kind="logistic", estimator=estimator)
        metadata.update(elapsed_seconds=time.perf_counter() - started,
                        classes=estimator.classes_.astype(int).tolist())
        bundle.update(config=settings, beta=settings["beta"], direction_mask=self.direction_mask.copy(),
                      positive_label=self.positive_label, n_features=int(self.a.shape[1]),
                      metadata=metadata)
        output.update(model=bundle, metadata=metadata)
        return output


def predict_model(bundle, transformedX):
    """Return {score, prediction, optional probability} from unweighted prepared X.

    Caller applies bundle['transform'] with apply_transform first, when present.
    This function alone applies beta.  Its SVM uses the same sum of strength and
    beta-squared direction distances as the search evaluator, including beta=0.
    """
    _need(bundle.get("format") == FORMAT, "Unsupported model bundle")
    values = _matrix(transformedX, "transformedX", int(bundle["n_features"]))
    mask = np.asarray(bundle["direction_mask"], dtype=bool)
    _need(mask.shape == (values.shape[1],), "Saved direction mask width mismatch")
    beta = float(bundle["beta"])
    with threadpool_limits(limits=1):
        if bundle["kind"] == "svm":
            train = _matrix(bundle["train_weighted_A"], "saved weighted training A", values.shape[1])
            train_d = _matrix(bundle["train_direction_unweighted"], "saved unweighted training direction",
                              int(mask.sum()))
            ds = _squared_distances(values[:, ~mask], train[:, ~mask])
            dd = _squared_distances(values[:, mask], train_d)
            kernel = np.exp(-float(bundle["resolved_gamma"]) * (ds + beta * beta * dd))
            return _oriented_outputs(bundle["estimator"], kernel, int(bundle["positive_label"]), False)
        _need(bundle["kind"] == "logistic", "Unknown model kind in saved bundle")
        weighted = np.ascontiguousarray(values * np.where(mask, beta, 1.))
        _need(np.isfinite(weighted).all(), "Saved direction beta overflowed inputs")
        return _oriented_outputs(bundle["estimator"], weighted, int(bundle["positive_label"]), True)


def save_model(bundle, path):
    """Atomically persist a final selected bundle (not every search candidate)."""
    _need(bundle.get("format") == FORMAT, "save_model expects fit(config)['model']")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=path.name + ".", suffix=".tmp",
                                         dir=str(path.parent), delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(bundle, handle, protocol=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(path)


def load_model(path):
    """Read a trusted local selected-model pickle produced by save_model."""
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    _need(isinstance(bundle, dict) and bundle.get("format") == FORMAT, "Unsupported saved model")
    _need(bundle.get("kind") in ("svm", "logistic"), "Unknown saved model kind")
    return bundle
