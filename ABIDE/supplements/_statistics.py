"""Exact selected summary function bodies; no historical HTML/plot/CLI.
Extracted from the hash-bound original summary_report.py without changing numerical definitions.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib,json,math
import numpy as np

EXPECTED_METHODS = {"RGIC", "Pearson", "Lag_peak", "Gaussian_TE", "VAR_Granger", "PSI",
                    "PhiID_unique_transfer", "Coherence", "GCMI"}

GROUP_FIELDS = ("family", "condition", "classifier", "atlas", "window", "seed", "method", "permutation_id")

METRICS = ("ACC", "AUROC", "balanced_ACC", "sensitivity", "specificity")

LIMITATION = ("采用完整训练＋验证＋测试标签联合预筛的结果不属于独立测试泛化；训练标签预筛的LOSO须按记录单列，"
              "也不等同于新的外部队列。区间仅描述已拟合OOF预测在当前样本中的条件不确定性，不包含重新训练、"
              "选参、预筛或回顾性设计的不确定性；不据此声称方向的因果贡献。")

BOOTSTRAP_METHOD = ("paired two-stage bootstrap: resample sites as clusters; within each sampled site, "
                    "resample subjects within fixed outer-fold×diagnosis cells; both methods share every draw; "
                    "fixed fitted OOF predictions, no retraining; descriptive conditional 95% interval")

def need(value, message):
    if not value:
        raise ValueError(message)

def canonical(value):
    return json.dumps(clean_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stable_seed(seed, value):
    return int(hashlib.sha256((str(seed) + ":" + canonical(value)).encode("utf8")).hexdigest()[:8], 16)

def weighted_metrics(y, prediction, score, weights=None):
    """ASD0 sensitivity, TC1 specificity, ASD-oriented AUROC including ties."""
    y, pred, score = np.asarray(y), np.asarray(prediction), np.asarray(score, dtype=float)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    need(y.ndim == pred.ndim == score.ndim == w.ndim == 1 and len(y) == len(pred) == len(score) == len(w), "Metric arrays are misaligned")
    need(np.isin(y, (0, 1)).all() and np.isin(pred, (0, 1)).all() and np.isfinite(score).all() and np.isfinite(w).all() and np.all(w >= 0), "Invalid binary labels, score, or bootstrap weight")
    total = float(w.sum())
    if total == 0:
        return {m: None for m in METRICS}
    asd, tc = y == 0, y == 1
    n_asd, n_tc = float(w[asd].sum()), float(w[tc].sum())
    sensitivity = float(w[asd & (pred == 0)].sum() / n_asd) if n_asd else None
    specificity = float(w[tc & (pred == 1)].sum() / n_tc) if n_tc else None
    auc = None
    if n_asd and n_tc:
        order = np.argsort(score, kind="mergesort")
        ordered_score = score[order]
        starts = np.r_[0, np.flatnonzero(ordered_score[1:] != ordered_score[:-1]) + 1]
        positive = np.add.reduceat((w * asd)[order], starts)
        negative = np.add.reduceat((w * tc)[order], starts)
        before = np.cumsum(negative) - negative
        auc = float(np.dot(positive, before + .5 * negative) / (n_asd * n_tc))
    return dict(ACC=float(w[pred == y].sum() / total), AUROC=auc,
                balanced_ACC=(sensitivity + specificity) / 2 if n_asd and n_tc else None,
                sensitivity=sensitivity, specificity=specificity)

def mean_defined(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None

def group_identity(record):
    return tuple(record[k] for k in GROUP_FIELDS)

def key_dict(key):
    return dict(zip(GROUP_FIELDS, key))

def normalize_record(raw, path, records_root):
    need(isinstance(raw, dict), "Record must be a JSON object")
    record = dict(raw)
    record["permutation_id"] = int(record.get("permutation_id", 0))
    need(record["permutation_id"] >= 0, "Permutation index must be nonnegative")
    record["window"] = int(record["window"])
    record["seed"] = int(record["seed"])
    for name in ("family", "condition", "classifier", "atlas", "method"):
        need(isinstance(record.get(name), str) and record[name], "Missing string field " + name)
    record["fold"] = str(record["fold"])
    record["expected_folds"] = int(record.get("expected_folds", 10))
    record["expected_test_n"] = int(record["expected_test_n"])
    need(record["expected_folds"] > 0 and record["expected_test_n"] > 0, "Expected fold/subject counts must be positive")
    record["status"] = record.get("status", "complete")
    need(record["status"] in ("complete", "failed", "unavailable", "pending", "running"), "Unknown record status")
    record["protocol_sha"] = record.get("protocol_sha", record.get("protocol_sha256"))
    need(isinstance(record["protocol_sha"], str) and len(record["protocol_sha"]) == 64 and
         all(c in "0123456789abcdefABCDEF" for c in record["protocol_sha"]), "Missing SHA256 protocol identity")
    need(record.get("parameter_source") is not None, "Parameter provenance is required")
    record["_path"] = str(path.relative_to(records_root))
    record["record_sha256"] = file_sha(path)
    if record["status"] != "complete":
        record["reason"] = str(record.get("reason", record.get("error", "No reason supplied")))
        return record
    need(record.get("label_semantics", "ASD0_TC1") == "ASD0_TC1" and
         record.get("score_semantics", "higher_ASD") == "higher_ASD", "Unknown label/score orientation")
    arrays = {name: np.asarray(record[name]) for name in
              ("ids_test", "sites_test", "y_test", "prediction_test", "score_test")}
    n = len(arrays["ids_test"])
    need(all(a.ndim == 1 and len(a) == n for a in arrays.values()), "Test arrays have different shapes")
    ids = [str(v) for v in arrays["ids_test"]]
    need(len(set(ids)) == len(ids), "Duplicate test identity inside fold")
    calculated = weighted_metrics(arrays["y_test"], arrays["prediction_test"], arrays["score_test"])
    supplied = record.get("test", {})
    need(isinstance(supplied, dict), "test must contain test-role metrics, not a validation scalar")
    for metric in METRICS:
        if metric in supplied and supplied[metric] is not None and calculated[metric] is not None:
            need(abs(float(supplied[metric]) - calculated[metric]) <= 1e-7,
                 "Provided test metric differs from actual test predictions: " + metric)
    record["test"] = calculated
    record["ids_test"] = ids
    record["sites_test"] = arrays["sites_test"].astype(str).tolist()
    record["y_test"] = arrays["y_test"].astype(int).tolist()
    record["prediction_test"] = arrays["prediction_test"].astype(int).tolist()
    record["score_test"] = arrays["score_test"].astype(float).tolist()
    return record

def aggregate_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[group_identity(record)].append(record)
    aggregates = []
    for key, all_rows in sorted(grouped.items(), key=lambda item: canonical(item[0])):
        rows = [row for row in all_rows if row["status"] == "complete"]
        first = all_rows[0]
        errors = []
        expected = int(first["expected_folds"])
        expected_ids = [str(x) for x in first.get("expected_fold_ids", range(expected))]
        if len(expected_ids) != expected or len(set(expected_ids)) != expected:
            errors.append("Invalid expected_fold_ids")
        if len({canonical([r["expected_folds"], r["expected_test_n"], r.get("expected_fold_ids", expected_ids), r["protocol_sha"]]) for r in all_rows}) != 1:
            errors.append("Inconsistent expected coverage or protocol identity")
        screen_scopes = sorted({str(r.get("feature_metadata", {}).get("recipe", {}).get("F_fit_scope",
                                    "all" if r.get("full_test_labels_used_in_F") is True else
                                    "train" if r.get("full_test_labels_used_in_F") is False else "unspecified")) for r in all_rows})
        if len(screen_scopes) != 1:
            errors.append("Feature-screen label scope differs across folds")
        for field in ("reference_family", "reference_condition", "reference_method", "reference_seed", "parameter_policy"):
            if len({canonical(r.get(field)) for r in all_rows}) != 1:
                errors.append(field + " differs across folds")
        fold_ids = [r["fold"] for r in rows]
        if len(set(fold_ids)) != len(fold_ids):
            errors.append("Duplicate COMPLETE fold records; no latest/best record was selected")
        if set(fold_ids) != set(expected_ids):
            errors.append("Missing/unexpected fold identities")
        flattened = []
        for r in rows:
            flattened.extend(dict(id=sid, site=site, y=y, pred=pred, score=score, fold=r["fold"])
                             for sid, site, y, pred, score in zip(r["ids_test"], r["sites_test"], r["y_test"], r["prediction_test"], r["score_test"]))
        if len({r["id"] for r in flattened}) != len(flattened):
            errors.append("OOF identities duplicated across folds")
        if len(flattened) != first["expected_test_n"]:
            errors.append("OOF subject total differs from expected_test_n")
        # An explicit subject-list hash is optional, but differing lists cannot
        # hide behind equal counts when the caller supplies one.
        population_sha = hashlib.sha256(canonical(sorted(r["id"] for r in flattened)).encode("utf8")).hexdigest()
        declared = {r.get("expected_test_ids_sha256") for r in all_rows if r.get("expected_test_ids_sha256")}
        if declared and declared != {population_sha}:
            errors.append("Expected OOF population identity hash differs")
        summary = dict(**key_dict(key), status="complete" if not errors else "incomplete",
                       complete_folds=len(set(fold_ids)), expected_folds=expected,
                       n_test=len(flattened), expected_test_n=first["expected_test_n"],
                       errors=errors, failed_or_unavailable_records=sum(r["status"] != "complete" for r in all_rows),
                       protocol_sha=first["protocol_sha"], test_population_sha256=population_sha,
                       F_fit_scopes=screen_scopes,
                       fold_record_sha256={r["_path"]: r["record_sha256"] for r in all_rows},
                       parameter_sources=[json.loads(x) for x in sorted({canonical(r["parameter_source"]) for r in all_rows})],
                       _rows=rows, _subjects=flattened, _first=first)
        for metric in METRICS:
            values = [r["test"][metric] for r in rows]
            defined = [v for v in values if v is not None]
            summary["mean_fold_" + metric] = mean_defined(values)
            summary["sd_fold_" + metric] = float(np.std(defined, ddof=1)) if len(defined) > 1 else None
            summary["n_defined_folds_" + metric] = len(defined)
        if flattened:
            summary["pooled"] = weighted_metrics([r["y"] for r in flattened], [r["pred"] for r in flattened], [r["score"] for r in flattened])
        else:
            summary["pooled"] = {m: None for m in METRICS}
        summary["mean_fit_seconds"] = mean_defined([r.get("fit_seconds") for r in rows])
        summary["mean_K"] = mean_defined([r.get("feature_metadata", {}).get("K") for r in rows])
        summary["mean_direction_features"] = mean_defined([r.get("feature_metadata", {}).get("n_direction_selected",
                                                                 r.get("feature_metadata", {}).get("n_direction")) for r in rows])
        aggregates.append(summary)
    # Rank only complete groups with the same task, never pool different
    # conditions, seeds, populations, classifiers, or null permutations.
    tasks = defaultdict(list)
    for row in aggregates:
        if row["status"] == "complete":
            tasks[tuple(row[k] for k in GROUP_FIELDS if k != "method")].append(row)
    for task in tasks.values():
        full_nine = {r["method"] for r in task} == EXPECTED_METHODS
        for r in task:
            r["n_compared_methods"] = len(task)
            r["complete_nine_method_ranking"] = full_nine
            for metric in ("ACC", "AUROC"):
                value = r["mean_fold_" + metric]
                r["rank_" + metric] = (1 + sum(o["mean_fold_" + metric] is not None and o["mean_fold_" + metric] > value for o in task)) if value is not None else None
    return aggregates

def bootstrap_weights(subjects, count, seed, cache):
    signature = [[s["id"], s["site"], s["fold"], s["y"]] for s in subjects]
    key = canonical([signature, count, seed])
    if key in cache:
        return cache[key]
    rng = np.random.RandomState(stable_seed(seed, signature))
    sites = sorted({s["site"] for s in subjects})
    cells = defaultdict(list)
    for i, row in enumerate(subjects):
        cells[(row["site"], row["fold"], row["y"])].append(i)
    by_site = {site: [np.asarray(ix, dtype=int) for (s, _, _), ix in cells.items() if s == site] for site in sites}
    result = np.zeros((count, len(subjects)), dtype=np.int32)
    for b in range(count):
        multiplicities = rng.multinomial(len(sites), np.repeat(1 / len(sites), len(sites)))
        for site, times in zip(sites, multiplicities):
            for _ in range(int(times)):
                for ix in by_site[site]:
                    draws = rng.multinomial(len(ix), np.repeat(1 / len(ix), len(ix)))
                    result[b, ix] += draws
    cache[key] = result
    return result

def paired_comparison(reference, target, bootstrap, bootstrap_seed, weight_cache):
    rm = {r["id"]: r for r in reference["_subjects"]}
    tm = {r["id"]: r for r in target["_subjects"]}
    common = sorted(set(rm) & set(tm))
    need(common, "No shared test identities")
    need(all((rm[s]["site"], rm[s]["fold"], rm[s]["y"]) == (tm[s]["site"], tm[s]["fold"], tm[s]["y"]) for s in common),
         "Paired identities have different test site, fold, or labels")
    subjects = [rm[s] for s in common]
    y = np.asarray([rm[s]["y"] for s in common])
    rp, rs = np.asarray([rm[s]["pred"] for s in common]), np.asarray([rm[s]["score"] for s in common])
    tp, ts = np.asarray([tm[s]["pred"] for s in common]), np.asarray([tm[s]["score"] for s in common])
    folds = {fold: np.asarray([i for i, row in enumerate(subjects) if row["fold"] == fold], dtype=int)
             for fold in sorted({row["fold"] for row in subjects})}
    fold_delta, defined_folds = {m: [] for m in METRICS}, {m: [] for m in METRICS}
    for fold, ix in folds.items():
        r, t = weighted_metrics(y[ix], rp[ix], rs[ix]), weighted_metrics(y[ix], tp[ix], ts[ix])
        for metric in METRICS:
            if r[metric] is not None and t[metric] is not None:
                fold_delta[metric].append(r[metric] - t[metric])
                defined_folds[metric].append(fold)
    weights = bootstrap_weights(subjects, bootstrap, bootstrap_seed, weight_cache)
    distributions = {m: [] for m in METRICS}
    for w in weights:
        metrics = {}
        for fold, ix in folds.items():
            metrics[fold] = (weighted_metrics(y[ix], rp[ix], rs[ix], w[ix]), weighted_metrics(y[ix], tp[ix], ts[ix], w[ix]))
        for metric in METRICS:
            wanted = defined_folds[metric]
            if not wanted:
                continue
            values = [(metrics[f][0][metric], metrics[f][1][metric]) for f in wanted]
            if all(a is not None and b is not None for a, b in values):
                distributions[metric].append(float(np.mean([a - b for a, b in values])))
    result = dict(reference={k: reference[k] for k in GROUP_FIELDS}, target={k: target[k] for k in GROUP_FIELDS},
                  orientation="reference_minus_target", n_common=len(common), n_reference_only=len(set(rm) - set(tm)),
                  n_target_only=len(set(tm) - set(rm)), shared_population="intersection of OOF identities",
                  shared_population_sha256=hashlib.sha256(canonical(common).encode("utf8")).hexdigest(),
                  same_test_population=set(rm) == set(tm), bootstrap_method=BOOTSTRAP_METHOD,
                  bootstrap_requested=bootstrap, bootstrap_seed=bootstrap_seed, n_sites=len({r["site"] for r in subjects}),
                  conditional_on_fitted_models=True, independent_generalization_claimed=False,
                  significance_claimed=False)
    for metric in METRICS:
        samples = distributions[metric]
        result["mean_fold_delta_" + metric] = mean_defined(fold_delta[metric])
        result["fold_deltas_" + metric] = dict(zip(defined_folds[metric], fold_delta[metric]))
        result["n_defined_folds_" + metric] = len(defined_folds[metric])
        result["conditional_CI95_" + metric] = np.quantile(samples, [.025, .975]).tolist() if len(samples) >= 2 else None
        result["bootstrap_valid_" + metric] = len(samples)
        result["bootstrap_undefined_" + metric] = bootstrap - len(samples)
    return result
