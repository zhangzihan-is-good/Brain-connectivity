"""Reliability-gated lagged connectivity on preprocessed regional time series.

This module implements the common RGIC equations.  It does not preprocess
images, fit diagnostic classifiers, choose hyperparameters, or infer causation.
All edge arrays use the canonical upper triangle ``i < j``.  A positive lag
compares ``x_i(t)`` with ``x_j(t + lag)`` and indicates that i precedes j.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RGICConfig:
    """Explicit numerical choices for RGIC.

    ``lags`` are always nonzero integer sampling offsets, even when
    ``lag_unit='seconds'``.  In seconds mode only temporal cost is converted
    using ``tr``; FLS lag consistency still uses exact signed sample offsets.
    Consequently one fit assumes a common sampling interval in seconds mode.
    ``lambda_`` and ``gamma`` have inverse units matching temporal cost.
    """

    lags: tuple[int, ...] = (-1, 1)
    lag_unit: str = "samples"
    tr: float | None = None
    lambda_: float = 1.0
    gamma: float = 0.2
    delta: float = 0.0
    cap: float = 0.9
    quantile: float = 0.95
    epsilon: float = 1e-8
    tie_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        try:
            values = tuple(self.lags)
        except TypeError as exc:
            raise ValueError("lags must be a sequence of signed integer offsets") from exc
        if not values or any(
            isinstance(d, (bool, np.bool_)) or not isinstance(d, Integral) or d == 0
            for d in values
        ):
            raise ValueError("lags must contain nonzero signed integers")
        if len(set(values)) != len(values):
            raise ValueError("lags must not contain duplicates")
        if any(-d not in values for d in values):
            raise ValueError("lags must contain the matching negative/positive offset for every lag")
        object.__setattr__(self, "lags", tuple(sorted(map(int, values), key=lambda d: (abs(d), d))))
        if self.lag_unit not in ("samples", "seconds"):
            raise ValueError("lag_unit must be 'samples' or 'seconds'")
        for name in ("lambda_", "gamma", "delta", "cap", "quantile", "epsilon", "tie_tolerance"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a finite number, not bool")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if min(self.lambda_, self.gamma, self.delta, self.tie_tolerance) < 0:
            raise ValueError("lambda_, gamma, delta and tie_tolerance must be nonnegative")
        if not 0 < self.cap <= 1:
            raise ValueError("cap must lie in (0, 1]")
        if not 0 < self.quantile <= 1:
            raise ValueError("quantile must lie in (0, 1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.tr is not None:
            if isinstance(self.tr, (bool, np.bool_)):
                raise ValueError("tr must be a positive finite sampling interval")
            try:
                tr = float(self.tr)
            except (TypeError, ValueError) as exc:
                raise ValueError("tr must be a positive finite sampling interval") from exc
            if not np.isfinite(tr) or tr <= 0:
                raise ValueError("tr must be a positive finite sampling interval")
            object.__setattr__(self, "tr", tr)
        if self.lag_unit == "seconds" and self.tr is None:
            raise ValueError("tr is required when lag_unit='seconds'")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable configuration values."""
        values = asdict(self)
        values["lags"] = list(self.lags)
        return values


@dataclass
class RGICResult:
    """One target record's connectivity and intermediate edge arrays.

    Arrays follow ``pairs``, the canonical upper triangle ``i < j``.
    ``opposite_tie`` flags peaks tied across positive and negative offsets;
    the deterministic representative lag is retained for the core equations.
    """

    n_regions: int
    pairs: np.ndarray
    r0: np.ndarray
    rstar: np.ndarray
    lag: np.ndarray
    cost: np.ndarray
    gain: np.ndarray
    potential: np.ndarray
    evidence: np.ndarray
    eta: np.ndarray
    ic: np.ndarray
    strength: np.ndarray
    F: np.ndarray
    L: np.ndarray
    S: np.ndarray
    reliability: np.ndarray
    metadata: dict[str, Any]
    opposite_tie: np.ndarray

    def matrix(self, name: str = "ic", diagonal: float | None = None) -> np.ndarray:
        """Expand an edge field to an R x R matrix.

        ``ic``, ``r0`` and ``strength`` default to diagonal one. Other fields
        default to zero. Only the lag matrix is antisymmetric.
        """
        if name not in (
            "r0", "rstar", "lag", "cost", "gain", "potential", "evidence",
            "eta", "ic", "strength", "F", "L", "S", "reliability", "opposite_tie",
        ):
            raise ValueError(f"Unknown result field: {name!r}")
        values = getattr(self, name)
        antisymmetric = name == "lag"
        if diagonal is None:
            diagonal = 1.0 if name in ("r0", "ic", "strength") else 0.0
        if not np.isfinite(diagonal):
            raise ValueError("diagonal must be finite")
        if antisymmetric and diagonal != 0:
            raise ValueError("An antisymmetric lag matrix must have diagonal zero")
        output = np.zeros((self.n_regions, self.n_regions), dtype=np.float64)
        i, j = self.pairs.T
        output[i, j] = values
        output[j, i] = -values if antisymmetric else values
        np.fill_diagonal(output, diagonal)
        return output


def _as_records(records: Any, *, max_lag: int, n_regions: int | None = None) -> list[np.ndarray]:
    if isinstance(records, np.ndarray):
        if records.ndim == 2:
            raw = [records]
        elif records.ndim == 3:
            raw = list(records)
        else:
            raise ValueError("Expected a (time, ROI) record, (record, time, ROI) array, or list of 2D records")
    else:
        try:
            raw = list(records)
        except TypeError as exc:
            raise ValueError("records must contain numeric time-by-ROI arrays") from exc
        if raw and not all(np.ndim(x) == 2 for x in raw):
            try:
                as_array = np.asarray(raw)
            except ValueError as exc:
                raise ValueError("Each record must have shape (time, ROI)") from exc
            if as_array.ndim == 2:
                raw = [as_array]
    if not raw:
        raise ValueError("At least one record is required")
    valid = []
    for index, record in enumerate(raw):
        x = np.asarray(record)
        if x.ndim != 2:
            raise ValueError(f"Record {index} must have shape (time, ROI)")
        if x.dtype.kind not in "buif" or np.iscomplexobj(x):
            raise ValueError(f"Record {index} must contain real numeric values")
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] - max_lag < 3:
            raise ValueError(f"Record {index} has too few time points: every lag needs at least three overlapping samples")
        if x.shape[1] < 2:
            raise ValueError("At least two ROIs are required")
        if n_regions is None:
            n_regions = x.shape[1]
        if x.shape[1] != n_regions:
            raise ValueError(f"Record {index} has {x.shape[1]} ROIs, expected {n_regions}")
        if not np.isfinite(x).all():
            raise ValueError(f"Record {index} contains NaN or infinity; clean or exclude invalid input explicitly")
        if np.any(np.all(x == x[0], axis=0)):
            raise ValueError(f"Record {index} contains a constant ROI; correlations are undefined")
        valid.append(x)
    return valid


def _labels(roi_names: Sequence[str] | None, n_regions: int) -> tuple[str, ...] | None:
    if roi_names is None:
        return None
    if isinstance(roi_names, str):
        raise ValueError("roi_names must be a sequence of unique labels, not one string")
    names = tuple(roi_names)
    if len(names) != n_regions:
        raise ValueError(f"Expected {n_regions} ROI names, got {len(names)}")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("ROI names must be nonempty unique strings")
    return names


def _unit_columns(x: np.ndarray, context: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        centered = x - np.mean(x, axis=0)
    if not np.isfinite(centered).all():
        raise ValueError(f"Numerical range is too large for stable correlation ({context})")
    scale = np.max(np.abs(centered), axis=0)
    if np.any(scale == 0):
        raise ValueError(f"A constant ROI occurs within overlapping samples ({context})")
    centered = centered / scale
    norm = np.sqrt(np.sum(centered * centered, axis=0))
    return centered / norm


def _record_stats(x: np.ndarray, config: RGICConfig, pairs: np.ndarray) -> dict[str, np.ndarray]:
    i, j = pairs.T
    zero = _unit_columns(x, "zero lag")
    r0 = np.clip((zero.T @ zero)[i, j], -1, 1)
    correlations = []
    for lag in config.lags:
        if lag > 0:
            left, right = x[:-lag], x[lag:]
        else:
            left, right = x[-lag:], x[:lag]
        a = _unit_columns(left, f"lag {lag}, first signal")
        b = _unit_columns(right, f"lag {lag}, second signal")
        correlations.append(np.clip((a.T @ b)[i, j], -1, 1))
    rho = np.asarray(correlations)
    absolute = np.abs(rho)
    maximum = absolute.max(axis=0)
    candidates = maximum[None, :] - absolute <= config.tie_tolerance
    # Config order is smaller absolute lag first, then negative signed lag.
    selected = np.argmax(candidates, axis=0)
    rstar = rho[selected, np.arange(len(pairs))]
    lag_values = np.asarray(config.lags, dtype=np.int64)
    lag = lag_values[selected]
    positive = lag_values > 0
    opposite_tie = candidates[positive].any(axis=0) & candidates[~positive].any(axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        cost = np.abs(lag).astype(np.float64) * (config.tr if config.lag_unit == "seconds" else 1.0)
    if not np.isfinite(cost).all():
        raise ValueError("The chosen lags and TR produce a nonfinite temporal cost")
    return {
        "r0": r0,
        "rstar": rstar,
        "lag": lag,
        "cost": cost,
        "gain": np.maximum(np.abs(rstar) - np.abs(r0) - config.delta, 0.0),
        "opposite_tie": opposite_tie,
    }


def _potential(rstar: np.ndarray, cost: np.ndarray, config: RGICConfig) -> np.ndarray:
    absolute = np.abs(rstar)
    output = np.zeros_like(absolute)
    nonzero = absolute > 0
    # Apply the cap in log space so large gamma values cannot overflow exp.
    with np.errstate(over="ignore", invalid="ignore"):
        log_amplitude = np.log(absolute[nonzero]) + config.gamma * cost[nonzero]
        output[nonzero] = np.exp(np.minimum(log_amplitude, np.log(config.cap)))
    return np.sign(rstar) * output


def _gate(gain: np.ndarray, cost: np.ndarray, reliability: np.ndarray, config: RGICConfig) -> tuple[np.ndarray, np.ndarray, float]:
    with np.errstate(over="ignore", under="ignore"):
        evidence = reliability * gain * np.exp(-config.lambda_ * cost)
    # The normalization set always includes every undirected edge, including zeros.
    q = float(np.quantile(evidence, config.quantile, method="linear"))
    eta = np.clip(evidence / (q + config.epsilon), 0.0, 1.0)
    return evidence, eta, q


class RGIC:
    """Fit reference-only FLS reliability and transform target recordings.

    No labels are used.  References contribute equal weight per record.
    ``transform`` always returns a list, including for a single 2D input.
    Supplying ROI labels in both calls permits exact order verification.
    Without labels, consistent ROI order is the caller's responsibility.
    """

    def __init__(self, config: RGICConfig | None = None):
        self.config = config if config is not None else RGICConfig()
        if not isinstance(self.config, RGICConfig):
            raise TypeError("config must be an RGICConfig instance")

    def fit(self, reference_records: Any, *, roi_names: Sequence[str] | None = None) -> RGIC:
        """Estimate F, L, S and R exclusively from the given references.

        Valid reference records have strictly positive gain.  Opposite-lag
        ties retain the deterministic representative peak in FLS and are
        flagged in each result. A single reference is
        accepted mathematically but cannot establish repeated consistency.
        """
        config = self.config
        records = _as_records(reference_records, max_lag=max(map(abs, config.lags)))
        n_regions = records[0].shape[1]
        names = _labels(roi_names, n_regions)
        pairs = np.column_stack(np.triu_indices(n_regions, 1))
        m = np.zeros(len(pairs), dtype=np.int64)
        sign_sum = np.zeros(len(pairs), dtype=np.float64)
        lag_counts = np.zeros((len(config.lags), len(pairs)), dtype=np.int64)
        for x in records:
            stats = _record_stats(x, config, pairs)
            valid = stats["gain"] > 0
            m += valid
            sign_sum += np.where(valid, np.sign(stats["rstar"]), 0.0)
            for index, lag in enumerate(config.lags):
                lag_counts[index] += valid & (stats["lag"] == lag)
        F = m / len(records)
        L = np.divide(lag_counts.max(axis=0), m, out=np.zeros_like(F), where=m > 0)
        S = np.divide(np.abs(sign_sum), m, out=np.zeros_like(F), where=m > 0)
        self.F_, self.L_, self.S_ = F, L, S
        self.reliability_ = F * L * S
        self.n_regions_ = n_regions
        self.n_reference_ = len(records)
        self.roi_names_ = names
        self.config_ = config
        return self

    def _check_fitted(self) -> None:
        required = ("F_", "L_", "S_", "reliability_", "n_regions_", "n_reference_", "roi_names_", "config_")
        if any(not hasattr(self, name) for name in required):
            raise ValueError("Call fit(reference_records) before transform(records)")
        if self.config != self.config_:
            raise ValueError("Configuration changed after fit; refit the references before transforming")
        expected = self.n_regions_ * (self.n_regions_ - 1) // 2
        for name in ("F_", "L_", "S_", "reliability_"):
            array = np.asarray(getattr(self, name))
            if array.shape != (expected,) or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                raise ValueError(f"Invalid fitted state: {name}")
        if not np.allclose(self.reliability_, self.F_ * self.L_ * self.S_, rtol=1e-12, atol=1e-15):
            raise ValueError("Invalid fitted state: reliability_ must equal F_ * L_ * S_")

    def transform(self, records: Any, *, roi_names: Sequence[str] | None = None) -> list[RGICResult]:
        """Calculate target-record connectivity without updating reference R."""
        self._check_fitted()
        config = self.config_
        targets = _as_records(records, max_lag=max(map(abs, config.lags)), n_regions=self.n_regions_)
        names = _labels(roi_names, self.n_regions_)
        order_checked = names is not None and self.roi_names_ is not None
        if order_checked and names != self.roi_names_:
            raise ValueError("Target ROI labels/order do not match the fitted references")
        pairs = np.column_stack(np.triu_indices(self.n_regions_, 1))
        results = []
        for x in targets:
            stats = _record_stats(x, config, pairs)
            potential = _potential(stats["rstar"], stats["cost"], config)
            evidence, eta, q = _gate(stats["gain"], stats["cost"], self.reliability_, config)
            ic = (1.0 - eta) * stats["r0"] + eta * potential
            results.append(RGICResult(
                n_regions=self.n_regions_, pairs=pairs.copy(), r0=stats["r0"], rstar=stats["rstar"], lag=stats["lag"],
                cost=stats["cost"], gain=stats["gain"], potential=potential, evidence=evidence, eta=eta,
                ic=ic, strength=np.abs(ic), F=self.F_.copy(), L=self.L_.copy(), S=self.S_.copy(),
                reliability=self.reliability_.copy(),
                metadata={
                    "config": config.to_dict(), "aggregation": "per_record", "normalization_scope": "current_record_undirected_edges",
                    "normalization_quantile_value": q, "n_reference": self.n_reference_, "n_timepoints": int(x.shape[0]),
                    "roi_names": list(self.roi_names_ or names) if (self.roi_names_ or names) is not None else None,
                    "roi_order_checked": order_checked, "lag_definition": "positive lag: first ROI precedes second ROI",
                    "lag_array_unit": "samples", "lag_consistency_unit": "signed_sample_offset",
                }, opposite_tie=stats["opposite_tie"],
            ))
        return results

