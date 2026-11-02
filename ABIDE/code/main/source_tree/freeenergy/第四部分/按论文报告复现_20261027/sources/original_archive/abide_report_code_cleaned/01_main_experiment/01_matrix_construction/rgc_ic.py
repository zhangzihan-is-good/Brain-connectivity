#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 代码功能：RGC-IC 核心公式模块，集中实现 lag 搜索、gain-cost evidence、reliability、eta gate 和 IC 融合。
# 函数功能：
# - SubjectRGCBase：类定义。
# - EtaTemplate：类定义。
# - decode_if_bytes：模块级函数。
# - ensure_dir：模块级函数。
# - standardize_rows：模块级函数。
# - load_roi_timeseries：模块级函数。
# - vector_pairs：模块级函数。
# - corr_at_lag：模块级函数。
# - compute_subject_rgc_base：模块级函数。
# - write_subject_rgc_base：模块级函数。
# - build_rgc_base_hdf5：模块级函数。
# - read_ids：模块级函数。
# - stack_patient_dataset：模块级函数。
# - labels_for_ids：模块级函数。
# - pheno_for_ids：模块级函数。
# - _mode_frequency：模块级函数。
# - estimate_eta_template：模块级函数。
# - make_rgc_features：模块级函数。
# - build_fold_feature_sets：模块级函数。


from __future__ import annotations

# Shared RGIC execution: IC commit/profile provenance is verified at entry.
import sys as _rgic_sys
from pathlib import Path as _RGICPath
_RGIC_MERGE_ROOT = next(p for p in _RGICPath(__file__).resolve().parents if (p / 'rgic_shared.py').is_file())
if str(_RGIC_MERGE_ROOT) not in _rgic_sys.path:
    _rgic_sys.path.insert(0, str(_RGIC_MERGE_ROOT))
import rgic_shared as _rgic_shared


import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from utils import format_config


DERIVATIVE_PATHS = {
    "aal": "cpac/filt_global/rois_aal/{subject}_rois_aal.1D",
    "cc200": "cpac/filt_global/rois_cc200/{subject}_rois_cc200.1D",
    "dosenbach160": "cpac/filt_global/rois_dosenbach160/{subject}_rois_dosenbach160.1D",
    "ez": "cpac/filt_global/rois_ez/{subject}_rois_ez.1D",
    "ho": "cpac/filt_global/rois_ho/{subject}_rois_ho.1D",
    "tt": "cpac/filt_global/rois_tt/{subject}_rois_tt.1D",
}


@dataclass
class SubjectRGCBase:
    pearson: np.ndarray
    peak_rho: Dict[int, np.ndarray]
    peak_lag: Dict[int, np.ndarray]
    potential: Dict[int, np.ndarray]
    gain: Dict[int, np.ndarray]
    cost: Dict[int, np.ndarray]


@dataclass
class EtaTemplate:
    eta: np.ndarray
    raw_eta: np.ndarray
    reliability: np.ndarray
    frequency: np.ndarray
    lag_consistency: np.ndarray
    sign_consistency: np.ndarray
    mean_gain: np.ndarray
    mean_cost: np.ndarray
    q_eta: float


DEFAULT_GAMMA = 0.2
DEFAULT_LAMBDA_COST = 1.0
DEFAULT_DELTA = 0.0
DEFAULT_C_MAX = 0.90
DEFAULT_EPS = 1e-8
DEFAULT_ETA_PERCENTILE = 95.0


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def standardize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-12)
    return np.nan_to_num(x)


def load_roi_timeseries(file_path: str) -> np.ndarray:
    df = pd.read_csv(file_path, sep="\t", header=0)
    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    roi_cols = ["#" + str(idx) for idx in sorted(int(col[1:]) for col in df.keys().tolist())]
    roi_by_time = np.nan_to_num(df[roi_cols].to_numpy().T)
    return standardize_rows(roi_by_time)


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


def write_subject_rgc_base(
    output_h5: h5py.File,
    pid: str,
    derivative: str,
    base: SubjectRGCBase,
    attrs: Dict,
    overwrite: bool = False,
) -> None:
    patients = output_h5.require_group("patients")
    patient_group = patients.require_group(pid)
    for key, value in attrs.items():
        patient_group.attrs[key] = value

    if derivative in patient_group:
        if not overwrite:
            return
        del patient_group[derivative]

    derivative_group = patient_group.require_group(derivative)
    derivative_group.create_dataset("pearson", data=base.pearson)

    for window in sorted(base.potential.keys()):
        window_group = derivative_group.require_group("w{}".format(window))
        window_group.create_dataset("peak_rho", data=base.peak_rho[window])
        window_group.create_dataset("peak_lag", data=base.peak_lag[window])
        window_group.create_dataset("potential", data=base.potential[window])
        window_group.create_dataset("gain", data=base.gain[window])
        window_group.create_dataset("cost", data=base.cost[window])


def build_rgc_base_hdf5(
    pheno,
    derivatives: Sequence[str],
    output_path: str,
    functionals_root: str = "./data/functionals",
    windows: Sequence[int] = (1, 2),
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    c_max: float = DEFAULT_C_MAX,
    overwrite: bool = False,
) -> str:
    ensure_dir(os.path.dirname(output_path))
    with h5py.File(output_path, "a") as h5:
        h5.attrs["c_max"] = float(c_max)
        h5.attrs["gamma"] = float(gamma)
        h5.attrs["delta"] = float(delta)
        h5.attrs["windows"] = ",".join(str(int(w)) for w in windows)

        file_ids = pheno["FILE_ID"].tolist()
        for derivative in derivatives:
            if derivative not in DERIVATIVE_PATHS:
                raise ValueError("Unsupported derivative: {}".format(derivative))

            template = os.path.join(functionals_root, DERIVATIVE_PATHS[derivative])
            for idx, pid in enumerate(file_ids, start=1):
                patient_group = h5.require_group("patients").require_group(pid)
                if derivative in patient_group and not overwrite:
                    continue

                file_path = format_config(template, {"subject": pid})
                if not os.path.exists(file_path):
                    raise FileNotFoundError(file_path)

                roi_by_time = load_roi_timeseries(file_path)
                base = compute_subject_rgc_base(
                    roi_by_time,
                    windows=windows,
                    gamma=gamma,
                    delta=delta,
                    c_max=c_max,
                )
                record = pheno[pheno["FILE_ID"] == pid].iloc[0]
                attrs = {
                    "id": record["FILE_ID"],
                    "y": int(record["DX_GROUP"]),
                    "site": record["SITE_ID"],
                    "sex": int(record["SEX"]),
                    "age": float(record["AGE"]),
                }
                write_subject_rgc_base(
                    output_h5=h5,
                    pid=pid,
                    derivative=derivative,
                    base=base,
                    attrs=attrs,
                    overwrite=overwrite,
                )
                print("Stored {} {} ({}/{})".format(derivative, pid, idx, len(file_ids)))

    return output_path


def read_ids(dataset) -> List[str]:
    return [decode_if_bytes(item) for item in dataset]


def stack_patient_dataset(
    base_h5: h5py.File,
    ids: Sequence[str],
    derivative: str,
    dataset_name: str,
    window: int | None = None,
) -> np.ndarray:
    values = []
    for pid in ids:
        derivative_group = base_h5["patients"][pid][derivative]
        if window is None:
            values.append(np.asarray(derivative_group[dataset_name], dtype=np.float32))
        else:
            values.append(np.asarray(derivative_group["w{}".format(int(window))][dataset_name], dtype=np.float32))
    return np.asarray(values, dtype=np.float32)


def labels_for_ids(base_h5: h5py.File, ids: Sequence[str]) -> np.ndarray:
    return np.asarray([int(base_h5["patients"][pid].attrs["y"]) for pid in ids], dtype=np.int64)


def pheno_for_ids(base_h5: h5py.File, ids: Sequence[str]) -> np.ndarray:
    pheno = []
    for pid in ids:
        attrs = base_h5["patients"][pid].attrs
        pheno.append([float(attrs["sex"]), float(attrs["age"])])
    return np.asarray(pheno, dtype=np.float32)


def _mode_frequency(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max() / values.size)


def estimate_eta_template(
    gain_train, cost_train, peak_lag_train, peak_rho_train,
    lambda_cost=DEFAULT_LAMBDA_COST, eta_percentile=DEFAULT_ETA_PERCENTILE, eps=DEFAULT_EPS,
):
    values = _rgic_shared.abide_group_eta(gain_train, cost_train, peak_lag_train, peak_rho_train,
                                         lambda_cost, eta_percentile, eps)
    return EtaTemplate(**values)


def make_rgc_features(pearson, potential, eta):
    return _rgic_shared.abide_matrix_blend(pearson, potential, eta)


def build_fold_feature_sets(
    base_h5: h5py.File,
    fold_group,
    derivative: str,
    feature_kind: str,
    lambda_cost: float = DEFAULT_LAMBDA_COST,
    eta_percentile: float = DEFAULT_ETA_PERCENTILE,
    eps: float = DEFAULT_EPS,
    include_pheno: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, EtaTemplate | None]:
    train_ids = read_ids(fold_group["train"])
    valid_ids = read_ids(fold_group["valid"])
    test_ids = read_ids(fold_group["test"])

    y_train = labels_for_ids(base_h5, train_ids)
    y_valid = labels_for_ids(base_h5, valid_ids)
    y_test = labels_for_ids(base_h5, test_ids)

    if feature_kind == "pearson":
        x_train = stack_patient_dataset(base_h5, train_ids, derivative, "pearson")
        x_valid = stack_patient_dataset(base_h5, valid_ids, derivative, "pearson")
        x_test = stack_patient_dataset(base_h5, test_ids, derivative, "pearson")
        eta_template = None
    elif feature_kind in {"w1-rgc", "w2-rgc"}:
        window = int(feature_kind[1])
        train_pearson = stack_patient_dataset(base_h5, train_ids, derivative, "pearson")
        valid_pearson = stack_patient_dataset(base_h5, valid_ids, derivative, "pearson")
        test_pearson = stack_patient_dataset(base_h5, test_ids, derivative, "pearson")

        train_potential = stack_patient_dataset(base_h5, train_ids, derivative, "potential", window=window)
        valid_potential = stack_patient_dataset(base_h5, valid_ids, derivative, "potential", window=window)
        test_potential = stack_patient_dataset(base_h5, test_ids, derivative, "potential", window=window)

        gain_train = stack_patient_dataset(base_h5, train_ids, derivative, "gain", window=window)
        cost_train = stack_patient_dataset(base_h5, train_ids, derivative, "cost", window=window)
        peak_lag_train = stack_patient_dataset(base_h5, train_ids, derivative, "peak_lag", window=window)
        peak_rho_train = stack_patient_dataset(base_h5, train_ids, derivative, "peak_rho", window=window)

        eta_template = estimate_eta_template(
            gain_train=gain_train,
            cost_train=cost_train,
            peak_lag_train=peak_lag_train,
            peak_rho_train=peak_rho_train,
            lambda_cost=lambda_cost,
            eta_percentile=eta_percentile,
            eps=eps,
        )
        x_train = make_rgc_features(train_pearson, train_potential, eta_template.eta)
        x_valid = make_rgc_features(valid_pearson, valid_potential, eta_template.eta)
        x_test = make_rgc_features(test_pearson, test_potential, eta_template.eta)
    else:
        raise ValueError("Unsupported feature kind: {}".format(feature_kind))

    if include_pheno:
        x_train = np.concatenate([x_train, pheno_for_ids(base_h5, train_ids)], axis=1)
        x_valid = np.concatenate([x_valid, pheno_for_ids(base_h5, valid_ids)], axis=1)
        x_test = np.concatenate([x_test, pheno_for_ids(base_h5, test_ids)], axis=1)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, eta_template
