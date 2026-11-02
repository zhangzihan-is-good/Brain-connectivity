"""Reproduce only the simulations displayed in the revised Fig. 1.

No input data, parameter reselection, output calibration or supplementary analyses.
Run: python experiment.py --workers 4 --output results
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

import core
import estimators
from rgic_shared import simulation_estimate

FULL = "RGIC"
DEFAULT = "RGIC original"
REMOVALS = ["No reliability", "No gain", "No cost", "No compensation"]
DIRECTIONAL = ["RGIC", "Lag peak", "Gaussian_TE", "VAR_Granger", "PSI"]
STRENGTH = ["RGIC", "Lag peak", "Pearson", "Gaussian_TE", "VAR_Granger", "PSI",
            "Coherence", "GCMI", "PhiID_UniqueTransfer"]
FIXED = {"physical": {"lam": 2.0, "gamma": 0.45}, "W2": {"lam": 2.25, "gamma": 0.55}}
LAGGED = ["stable", "lag_variable", "sign_variable", "sporadic"]
NULL = ["sync", "sporadic", "independent"]
GROUP = ["cell", "field", "family", "relation", "repeat"]
TARGET = slice(core.N_REFERENCE, None)


def _baseline_targets(data, tr):
    """All methods receive the same three target records and lag-search limit."""
    x, y = [data[k][TARGET] for k in ("x", "y")]
    shape = x.shape[:2]
    result = estimators.compute_baselines_batch(
        x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), tr,
        max_delay=int(np.floor(2.2 / tr + 1e-10)))
    return {m: {k: np.asarray(v[k]).reshape(shape) for k in ("strength", "direction")}
            for m, v in result.items()}


def _predictions(stats, reliability, tr, baseline=True):
    result = {m: core.estimate(stats, reliability, m, tr, **FIXED["physical"])
              for m in [FULL] + REMOVALS + (["Pearson", "Lag peak"] if baseline else [])}
    if baseline:
        result[DEFAULT] = core.estimate(stats, reliability, FULL, tr, 1.0, 0.2,
                                        seconds_parameters=False)
    return {m: {k: v[k][TARGET] for k in ("strength", "direction")}
            for m, v in result.items()}


def _world(task):
    cell, split, repeat, purpose, scenario = task
    variant = 20261021 if purpose == "strength" else 0
    if scenario == "main":
        data = core.generate(cell, split, repeat, variant=variant)
    else:
        from interference import generate_interference
        data = generate_interference(cell, split, repeat, purpose)
    tr = core.FIELDS[cell["field"]]["tr"]
    stats = core.candidates(data, tr, "physical")
    reliability = core.reliability(stats)
    prediction = _predictions(stats, reliability, tr, baseline=scenario == "main")
    if scenario == "main":
        prediction.update(_baseline_targets(data, tr))
    common = dict(cell=cell["id"], field=cell["field"], family=cell["family"],
                  relation=cell["relation"], repeat=repeat, scenario=scenario)
    present = data["present"][TARGET]
    if purpose == "strength":
        rows = []
        for record in range(core.N_TARGET):
            mask = present[record]
            if mask.sum() < 4:
                continue
            for method, p in prediction.items():
                rows.append(common | dict(record=record, method=method,
                    mae=float(np.abs(p["strength"][record, mask] - data["kappa"][mask]).mean()),
                    n_present=int(mask.sum())))
        return pd.DataFrame(rows), None
    nr, ne = present.shape
    truth_direction = np.where(present, np.sign(data["actual_d"][TARGET]), 0).astype(int)
    metadata = common | dict(record=np.repeat(np.arange(nr), ne),
        edge=np.tile(np.arange(ne), nr), present=present.ravel(),
        true_direction=truth_direction.ravel(),
        template_direction=np.tile(np.sign(data["template_d"]).astype(int), nr),
        persistent_direction=cell["relation"] in ("stable", "sign_variable"))
    frames = [pd.DataFrame(metadata | dict(method=method, direction=p["direction"].ravel()))
              for method, p in prediction.items()
              if method in DIRECTIONAL + REMOVALS + [DEFAULT]]
    development = None
    if split == 0 and scenario == "main":
        development = dict(cell=cell, repeat=repeat, truth=data["template_d"], windows={})
        for window in ("physical", "W2"):
            s = stats if window == "physical" else core.candidates(data, tr, window)
            development["windows"][window] = dict(stats=s, R=core.reliability(s))
    return pd.concat(frames, ignore_index=True), development


def _run(pool, jobs, label):
    frames, development = [], []
    for i, (frame, dev) in enumerate(pool.map(_world, jobs), 1):
        frames.append(frame)
        if dev is not None:
            development.append(dev)
        if i % 36 == 0 or i == len(jobs):
            print(f"{label}: {i}/{len(jobs)} worlds", flush=True)
    return pd.concat(frames, ignore_index=True), development


def _thresholds(development):
    null = development[development.relation.isin(["sync", "independent"])]
    keys = ["scenario", "field", "family", "method"]
    return (null.groupby(keys, sort=False).direction.apply(
        lambda x: float(np.quantile(np.abs(x), 0.95))).rename("direction_threshold").reset_index())


def _direction_metrics(test, thresholds):
    keys = ["scenario", "field", "family", "method"]
    joined = test.merge(thresholds, on=keys, how="left", validate="many_to_one")
    assert joined.direction_threshold.notna().all()
    rows = []
    for key, g in joined.groupby(["scenario"] + GROUP + ["method"], sort=False):
        active = g.present & (g.true_direction != 0)
        absent = g.true_direction == 0
        called = np.abs(g.direction) > g.direction_threshold
        correct = np.sign(g.direction) == g.true_direction
        rows.append(dict(zip(["scenario"] + GROUP + ["method"], key)) | dict(
            current_correct_detection=float((called & correct)[active].mean()) if active.any() else np.nan,
            null_false_alarm=float(called[absent].mean()) if absent.any() else np.nan))
    blocks = []
    keys = ["scenario", "field", "family", "repeat", "method"]
    for key, g in test.groupby(keys, sort=False):
        assert len(g) == len(core.RELATIONS) * core.N_PAIRS * core.N_TARGET
        direction = g.direction.to_numpy()
        score = np.stack([np.maximum(direction, 0), np.maximum(-direction, 0)], axis=-1)
        truth = np.stack([g.persistent_direction & (g.template_direction > 0),
                          g.persistent_direction & (g.template_direction < 0)], axis=-1)
        blocks.append(dict(zip(keys, key)) | dict(
            persistent_ap=float(average_precision_score(truth.ravel(), score.ravel()))))
    return pd.DataFrame(rows), pd.DataFrame(blocks)


def _grid(development):
    """Evaluate the published grid; the fixed configurations are never reselected."""
    rows = []
    # Stack worlds so the continuous gate is evaluated in one array operation.
    for window in ("physical", "W2"):
        r0 = np.stack([w["windows"][window]["stats"]["r0"][TARGET] for w in development])
        peak = np.stack([w["windows"][window]["stats"]["peak"][TARGET] for w in development])
        tau = np.stack([w["windows"][window]["stats"]["tau"][TARGET] for w in development])
        gain = np.stack([w["windows"][window]["stats"]["G"][TARGET] for w in development])
        rel = np.stack([w["windows"][window]["R"][TARGET] for w in development])
        tr = np.asarray([core.FIELDS[w["cell"]["field"]]["tr"] for w in development])[:, None, None]
        cost = np.abs(tau) * tr
        labels, block_ids = [], {}
        for i, w in enumerate(development):
            truth = np.zeros((core.N_TARGET, core.N_PAIRS, 2), dtype=bool)
            if w["cell"]["relation"] in ("stable", "sign_variable"):
                truth[..., 0] = w["truth"][None] > 0
                truth[..., 1] = w["truth"][None] < 0
            labels.append(truth)
            key = w["cell"]["field"], w["cell"]["family"], w["repeat"]
            block_ids.setdefault(key, []).append(i)
        labels = np.asarray(labels)
        blocks = [(ids, labels[ids].ravel()) for ids in block_ids.values()]
        for lam in core.LAMBDAS:
            for gamma in core.GAMMAS:
                stats=dict(r0=r0,peak=peak,tau=tau,G=gain,C=np.abs(tau))
                direction=simulation_estimate(stats,rel,FULL,tr,lam,gamma)['direction']
                scores = np.stack([np.maximum(direction, 0), np.maximum(-direction, 0)], axis=-1)
                ap = np.mean([average_precision_score(y, scores[ids].ravel()) for ids, y in blocks])
                rows.append(dict(window=window, method=FULL, lam=lam, gamma=gamma,
                                 development_ap=float(ap)))
        print(f"Development grid: {window}, {len(core.LAMBDAS)} x {len(core.GAMMAS)}", flush=True)
    return pd.DataFrame(rows)


def interval(values, endpoint, scenario="main"):
    values = np.asarray(values, dtype=float)
    assert len(values) == core.N_TEST and np.isfinite(values).all()
    seed, draws = (20261022, 10000) if endpoint == "mae" else (20261004, 10000 if scenario == "interference" else 5000)
    ids = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    low, high = np.quantile(values[ids].mean(axis=1), [.025, .975])
    return float(values.mean()), float(low), float(high)


def _summaries(direction_cells, direction_blocks, strength_cells):
    endpoints = [("current_correct_detection", direction_cells[direction_cells.relation.isin(LAGGED)]),
                 ("null_false_alarm", direction_cells[direction_cells.relation.isin(NULL)]),
                 ("persistent_ap", direction_blocks), ("mae", strength_cells)]
    summaries, repeats, removals, parameter_changes = [], [], [], []
    for endpoint, frame in endpoints:
        for scenario, part in frame.groupby("scenario", sort=False):
            means = part.groupby(["repeat", "method"], sort=False)[endpoint].mean().unstack().reindex(range(10))
            assert means.notna().all().all(), (scenario, endpoint)
            for method in means:
                values = means[method].to_numpy()
                mean, low, high = interval(values, endpoint, scenario)
                summaries.append(dict(scenario=scenario, endpoint=endpoint, method=method,
                                      mean=mean, ci_low=low, ci_high=high))
                repeats.extend(dict(scenario=scenario, endpoint=endpoint, method=method,
                                    repeat=r, value=float(v)) for r, v in enumerate(values))
            for method in REMOVALS:
                delta = means[method] - means[FULL] if endpoint in ("mae", "null_false_alarm") else means[FULL] - means[method]
                mean, low, high = interval(delta, endpoint, scenario)
                removals.append(dict(scenario=scenario, endpoint=endpoint, method=method,
                    mean=mean, ci_low=low, ci_high=high, positive_means="removal worsens performance"))
                repeats.extend(dict(scenario=scenario, endpoint="removal_" + endpoint, method=method,
                                    repeat=r, value=float(v)) for r, v in enumerate(delta))
            if DEFAULT in means:
                delta = means[DEFAULT] - means[FULL] if endpoint in ("mae", "null_false_alarm") else means[FULL] - means[DEFAULT]
                mean, low, high = interval(delta, endpoint, scenario)
                parameter_changes.append(dict(scenario=scenario, endpoint=endpoint,
                    mean=mean, ci_low=low, ci_high=high, positive_means="fixed physical-time settings improve performance"))
    return (pd.DataFrame(summaries), pd.DataFrame(repeats), pd.DataFrame(removals),
            pd.DataFrame(parameter_changes))


def figure_inputs(out, tables):
    """Small, generated tables consumed by the figure and PowerPoint renderer."""
    cells = tables["direction_cells"]
    blocks = tables["direction_blocks"]
    strength = tables["strength_cells"]
    cells[cells.scenario == "main"].assign(window="physical", track="native").to_csv(
        out / "cell_metrics.csv", index=False)
    blocks[blocks.scenario == "main"].assign(window="physical", track="native").to_csv(
        out / "block_metrics.csv", index=False)
    strength[(strength.scenario == "main") & strength.method.isin(STRENGTH)].to_csv(
        out / "native_world_MAE.csv", index=False)
    configuration = {DEFAULT: "Sample-unit defaults", FULL: "Fixed physical-time"}
    strength[(strength.scenario == "main") & strength.method.isin(configuration)].assign(
        configuration=lambda q: q.method.map(configuration)).to_csv(
            out / "parameter_strength_world_MAE.csv", index=False)
    cells[(cells.scenario == "main") & cells.method.isin(configuration)].assign(
        configuration=lambda q: q.method.map(configuration)).to_csv(
            out / "parameter_detection_world_metrics.csv", index=False)
    endpoints = {"removal_current_correct_detection": "a_detection",
                 "removal_persistent_ap": "b_native_ap", "removal_mae": "mae"}
    r = tables["repeat_metrics"]
    r = r[r.endpoint.isin(endpoints)].rename(columns={"value": "difference"}).copy()
    r["setting"] = r.scenario.map({"main": "Original", "interference": "Shared interference"})
    r["endpoint"] = r.endpoint.map(endpoints)
    r.to_csv(out / "ablation_repeat_differences.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    scenarios = ["main", "interference"]
    thresholds, cells, blocks, records, development = [], [], [], [], []
    with threadpool_limits(1), ProcessPoolExecutor(max_workers=args.workers,
            initializer=threadpool_limits, initargs=(1,)) as pool:
        for scenario in scenarios:
            dev, dev_stats = _run(pool, [(c, 0, r, "direction", scenario)
                for c in core.cells() for r in range(core.N_DEV)], f"{scenario} development")
            if scenario == "main":
                development = dev_stats
            threshold = _thresholds(dev)
            test, _ = _run(pool, [(c, 1, r, "direction", scenario)
                for c in core.cells() for r in range(core.N_TEST)], f"{scenario} direction test")
            cell, block = _direction_metrics(test, threshold)
            native, _ = _run(pool, [(c, 1, r, "strength", scenario)
                for c in core.cells() if c["relation"] != "independent"
                for r in range(core.N_TEST)], f"{scenario} strength test")
            thresholds.append(threshold); cells.append(cell); blocks.append(block); records.append(native)
    thresholds, cells, blocks, records = [pd.concat(v, ignore_index=True)
                                        for v in (thresholds, cells, blocks, records)]
    strength = records.groupby(["scenario"] + GROUP + ["method"], sort=False).mae.mean().reset_index()
    summary, repeats, ablations, comparison = _summaries(cells, blocks, strength)
    tables = dict(direction_thresholds=thresholds, direction_cells=cells, direction_blocks=blocks,
                  strength_records=records, strength_cells=strength, summary=summary,
                  repeat_metrics=repeats, ablation_summary=ablations, parameter_comparison=comparison)
    tables["development_grid"] = _grid(development)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    figure_inputs(out, tables)
    print(summary.to_string(index=False), flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"Complete: {time.time()-started:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
