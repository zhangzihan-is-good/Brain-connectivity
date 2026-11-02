"""Original ABIDE lag-search primitives used by the sensitivity analyses."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

import sys as _shared_sys
from pathlib import Path as _SharedPath
_shared_root = next(p for p in _SharedPath(__file__).resolve().parents if (p / "rgic_shared.py").is_file())
if str(_shared_root) not in _shared_sys.path:
    _shared_sys.path.insert(0, str(_shared_root))
import rgic_shared as _rgic_shared

DEFAULT_GAMMA = 0.2
DEFAULT_DELTA = 0.0
DEFAULT_C_MAX = 0.90


@dataclass
class SubjectRGCBase:
    pearson: np.ndarray
    peak_rho: Dict[int, np.ndarray]
    peak_lag: Dict[int, np.ndarray]
    potential: Dict[int, np.ndarray]
    gain: Dict[int, np.ndarray]
    cost: Dict[int, np.ndarray]


def standardize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-12)
    return np.nan_to_num(x)


def vector_pairs(n_roi: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(1, n_roi) for j in range(i)]


def corr_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag > 0:
        xs = x[lag:]
        ys = y[:-lag]
    elif lag < 0:
        lag_abs = -lag
        xs = x[:-lag_abs]
        ys = y[lag_abs:]
    else:
        xs = x
        ys = y

    if len(xs) < 2:
        return 0.0

    sx = xs.std()
    sy = ys.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0

    return float(np.mean((xs - xs.mean()) * (ys - ys.mean())) / (sx * sy))


def compute_subject_rgc_base(
    roi_by_time: np.ndarray,
    windows: Sequence[int] = (1, 2),
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    c_max: float = DEFAULT_C_MAX,
) -> SubjectRGCBase:
    x = standardize_rows(roi_by_time)
    n_roi = x.shape[0]
    pairs = vector_pairs(n_roi)

    pearson = np.zeros(len(pairs), dtype=np.float32)
    peak_rho = {int(w): np.zeros(len(pairs), dtype=np.float32) for w in windows}
    peak_lag = {int(w): np.zeros(len(pairs), dtype=np.int16) for w in windows}
    potential = {int(w): np.zeros(len(pairs), dtype=np.float32) for w in windows}
    gain = {int(w): np.zeros(len(pairs), dtype=np.float32) for w in windows}
    cost = {int(w): np.zeros(len(pairs), dtype=np.float32) for w in windows}

    for edge_idx, (i, j) in enumerate(pairs):
        xi = x[i]
        xj = x[j]
        rho0 = corr_at_lag(xi, xj, 0)
        pearson[edge_idx] = rho0

        for window in windows:
            lags = [lag for lag in range(-int(window), int(window) + 1) if lag != 0]
            lag_corrs = np.array([corr_at_lag(xi, xj, lag) for lag in lags], dtype=np.float64)
            best_idx = int(np.abs(lag_corrs).argmax())
            best_lag = int(lags[best_idx])
            best_rho = float(lag_corrs[best_idx])

            p_val = _rgic_shared.abide_matrix_potential(best_rho, best_lag, gamma, c_max)

            peak_rho[int(window)][edge_idx] = best_rho
            peak_lag[int(window)][edge_idx] = best_lag
            potential[int(window)][edge_idx] = p_val
            gain[int(window)][edge_idx] = max(abs(best_rho) - abs(rho0) - float(delta), 0.0)
            cost[int(window)][edge_idx] = abs(best_lag)

    return SubjectRGCBase(
        pearson=pearson,
        peak_rho=peak_rho,
        peak_lag=peak_lag,
        potential=potential,
        gain=gain,
        cost=cost,
    )

