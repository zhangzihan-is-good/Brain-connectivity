"""Compute fixed-setting HCP coefficients and network-pair predictions.

Fit Pearson, reliability descriptors and full RGIC using the supplied folds.
Network-pair metrics use global out-of-fold predictions without subgroup fits.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import _shared
from rgic_shared import fuse_cached
from shared_paths import workspace_root
from io_shared import write_csv as shared_write_csv

ROOT = Path(__file__).resolve().parent
INPUTS = workspace_root() / 'dataset' / 'reproduction_inputs' / 'HCP'
FEATURES = ['distance', 'abs_pearson', 'mean_gain', 'mean_cost', 'freq',
            'lag_consistency', 'sign_consistency', 'reliability',
            'gain_reliability', 'gain_cost_evidence', 'raw_eta', 'eta',
            'q_gain', 'q_cost', 'q_evidence', 'abs_ic', 'abs_gain_ic']


def write_csv(path, rows):
    # Keep the historical NA spelling while sharing the CSV serializer.
    values = [{k: 'NA' if isinstance(v, float) and not np.isfinite(v)
               else v for k, v in row.items()} for row in rows]
    shared_write_csv(path, values)


def fit_fold(name, repeat, fold, X, y, assignments):
    train = np.flatnonzero(assignments[repeat] != fold)
    test = np.flatnonzero(assignments[repeat] == fold)
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        C=1, class_weight='balanced', solver='lbfgs', tol=1e-4, max_iter=3000))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        clf.fit(X[train], y[train])
    if any(issubclass(q.category, ConvergenceWarning) for q in caught):
        raise RuntimeError('Convergence warning: incomplete result rejected')
    score = clf.predict_proba(X[test])[:, 1]
    scaler, model = clf.steps[0][1], clf.steps[1][1]
    rows = [dict(model='RGIC', group='RGIC full', window='W1', repeat=repeat,
                 fold=fold, feature=feature, coefficient=float(model.coef_[0, i]),
                 scaler_mean=float(scaler.mean_[i]), scaler_scale=float(scaler.scale_[i]),
                 intercept=float(model.intercept_[0]), n_train=len(train))
            for i, feature in enumerate(FEATURES)] if name == 'RGIC' else []
    return name, repeat, fold, test, score, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=INPUTS / 'appendix_inputs.npz')
    parser.add_argument('--results', type=Path, default=ROOT / 'output' / 'appendix')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files if key != 'metadata_json'}
    f = {key: np.asarray(value, dtype=np.float64) for key, value in data.items()
         if np.shape(value) == (79800,)}
    ic = fuse_cached(f['r0'], f['eta'], f['potential'])
    f.update(abs_pearson=np.abs(f['r0']),
             gain_reliability=f['mean_gain'] * f['reliability'],
             abs_ic=np.abs(ic), abs_gain_ic=np.maximum(np.abs(ic) - np.abs(f['r0']), 0))
    X = np.column_stack([f[key] for key in FEATURES])
    y = data['sc_support'].astype(np.int8)
    assignments = data['cv_folds']
    order = ['Pearson', 'Reliability parts', 'RGIC']
    matrices = {'Pearson': np.column_stack([f['distance'], f['abs_pearson']]),
                'Reliability parts': np.column_stack([f[k] for k in
                    ['distance', 'abs_pearson', 'freq', 'lag_consistency', 'sign_consistency', 'reliability']]),
                'RGIC': X}
    oof = np.full((3, 20, len(y)), np.nan)
    network = data['network_labels']
    values = np.unique(network)
    assert np.array_equal(values, np.arange(7))
    fitted = {}
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=min(max(args.workers, 1), 6)) as pool:
        jobs = [pool.submit(fit_fold, name, r, k, matrices[name], y, assignments)
                for name in order for r in range(20) for k in range(5)]
        for job in as_completed(jobs):
            name, r, k, test, score, rows = job.result()
            oof[order.index(name), r, test] = score
            fitted[name, r, k] = rows
    assert np.isfinite(oof).all()
    np.save(out / 'diagnostic_OOF.npy', oof)
    coefficient_rows = [row for key in sorted(fitted) for row in fitted[key]]
    write_csv(out / 'coefficient_folds.csv', coefficient_rows)
    coefficient_summary = []
    for feature in FEATURES:
        coefficients = np.array([r['coefficient'] for r in coefficient_rows if r['feature'] == feature])
        coefficient_summary.append(dict(model='RGIC', group='RGIC full', window='W1', feature=feature,
            coefficient_mean=float(coefficients.mean()), coefficient_sd=float(coefficients.std(ddof=1)),
            absolute_coefficient_mean=float(np.abs(coefficients).mean()),
            coefficient_min=float(coefficients.min()), coefficient_max=float(coefficients.max()),
            positive_fraction=float(np.mean(coefficients > 0)), n_folds=100, n_cv_repeats=20))
    write_csv(out / 'coefficient_summary.csv', coefficient_summary)
    pair_rows, pair_summary = [], []
    ei, ej = data['edge_i'], data['edge_j']
    ni, nj = np.minimum(network[ei], network[ej]), np.maximum(network[ei], network[ej])
    baseline = oof[order.index('Pearson')]
    for i in values:
        for j in values[values >= i]:
            mask = (ni == i) & (nj == j)
            n, n_positive = int(mask.sum()), int(y[mask].sum())
            valid = 0 < n_positive < n
            for model in ['RGIC', 'Reliability parts']:
                local = []
                for repeat in range(20):
                    if valid:
                        a = float(average_precision_score(y[mask], oof[order.index(model), repeat, mask]))
                        b = float(average_precision_score(y[mask], baseline[repeat, mask]))
                        c = float(roc_auc_score(y[mask], oof[order.index(model), repeat, mask]))
                        d = float(roc_auc_score(y[mask], baseline[repeat, mask]))
                    else:
                        a = b = c = d = float('nan')
                    row = dict(model=model, group='RGIC full' if model == 'RGIC' else model,
                        window='W1', network_i=f'N{int(i)+1}', network_j=f'N{int(j)+1}',
                        network_index_i=int(i), network_index_j=int(j), repeat=repeat,
                        n_edges=n, n_positive=n_positive, n_negative=n-n_positive,
                        valid=valid, ap=a, pearson_ap=b, roc_auc=c, pearson_roc_auc=d,
                        delta_ap_pp=100*(a-b), delta_roc_auc_pp=100*(c-d))
                    pair_rows.append(row)
                    local.append(row)
                line = {key: value for key, value in local[0].items()
                        if key not in ('repeat', 'ap', 'pearson_ap', 'roc_auc', 'pearson_roc_auc', 'delta_ap_pp', 'delta_roc_auc_pp')}
                line['n_cv_repeats'] = 20
                for key in ['ap', 'pearson_ap', 'roc_auc', 'pearson_roc_auc', 'delta_ap_pp', 'delta_roc_auc_pp']:
                    v = np.array([r[key] for r in local])
                    line[key+'_mean'] = float(v.mean())
                    line[key+'_sd'] = float(v.std(ddof=1))
                pair_summary.append(line)
    write_csv(out / 'network_pair_results.csv', pair_rows)
    write_csv(out / 'network_pair_summary.csv', pair_summary)
    print(f'HCP appendix: 300 fits and all network-pair statistics saved to {out}', flush=True)


if __name__ == '__main__':
    main()
