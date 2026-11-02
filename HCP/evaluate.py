"""Reproduce the fixed-configuration HCP experiments in panels b--e.

Inputs are cached group-level edge features and a weighted anatomical matrix.
No parameter search or raw-BOLD feature re-estimation is performed.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from reporting_utils import safe_gain_ratio
import _shared
from rgic_shared import fuse_cached
from shared_paths import workspace_root
from io_shared import write_csv, write_json

ROOT = Path(__file__).resolve().parent
INPUTS = workspace_root() / 'dataset' / 'reproduction_inputs' / 'HCP'
SEED, NODE_SEED = 2026101809, 2026102101
METHODS = ['Pearson', 'RGIC', 'Lag peak', 'Coherence', 'GCMI', 'Gaussian TE',
           'VAR Granger', 'PSI', 'PhiID unique transfer']
FULL = ['mean_gain', 'mean_cost', 'freq', 'lag_consistency', 'sign_consistency',
        'reliability', 'gain_reliability', 'gain_cost_evidence', 'raw_eta', 'eta',
        'q_gain', 'q_cost', 'q_evidence', 'abs_ic', 'abs_gain_ic']
GROUPS = {
    'Pearson': [], 'Gain': ['mean_gain'],
    'Cost-discounted gain': ['gain_cost_evidence'],
    'Reliability parts': ['freq', 'lag_consistency', 'sign_consistency', 'reliability'],
    'Final-IC features': ['abs_ic', 'abs_gain_ic', 'abs_delta_ic', 'signed_delta_ic'],
    'RGIC': FULL,
}
NATIVE = {'Lag peak': 'native_lag_peak', 'Gaussian TE': 'native_gaussian_te',
          'VAR Granger': 'native_var_granger', 'Coherence': 'native_coherence',
          'PSI': 'native_psi', 'GCMI': 'native_gcmi', 'PhiID unique transfer': 'native_phiid'}
CONTROLS = ['Observed', 'Random', 'Global shuffle', 'Distance-bin shuffle']


def scores(y, prediction):
    return {'ap': float(average_precision_score(y, prediction)),
            'roc_auc': float(roc_auc_score(y, prediction))}


def load_inputs(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files if key != 'metadata_json'}
        metadata = json.loads(str(archive['metadata_json']))
    f = {key: np.asarray(value, dtype=np.float64) for key, value in data.items()
         if np.shape(value) == (79800,)}
    f['abs_pearson'] = abs(f['r0'])
    f['gain_reliability'] = f['mean_gain'] * f['reliability']
    ic = fuse_cached(f['r0'], f['eta'], f['potential'])
    f.update(abs_ic=abs(ic), abs_gain_ic=np.maximum(abs(ic) - abs(f['r0']), 0),
             abs_delta_ic=abs(ic - f['r0']), signed_delta_ic=ic - f['r0'])
    models = {}
    for name, extra in GROUPS.items():
        columns = ['distance', 'abs_pearson'] + extra
        models[name] = {'X': np.column_stack([f[key] for key in columns]), 'columns': columns}
    for name, field in NATIVE.items():
        columns = ['distance', 'abs_pearson', field]
        models[name] = {'X': np.column_stack([f[key] for key in columns]), 'columns': columns}
    y, folds = data['sc_support'].astype(np.int8), data['cv_folds']
    ei, ej = np.triu_indices(400, 1)
    assert y.shape == (79800,) and y.sum() == 5059 and folds.shape == (20, 79800)
    assert np.array_equal(ei, data['edge_i']) and np.array_equal(ej, data['edge_j'])
    assert set(np.unique(folds)) == set(range(5)) and len(models) == 13
    assert all(np.isfinite(model['X']).all() for model in models.values())
    config = metadata['fixed_configuration']
    assert config['W'] == 1 and config['lambda'] == config['gamma'] == 1
    strata = {}
    for label, values in [('Distance', f['distance']), ('Pearson', f['abs_pearson'])]:
        bins = np.searchsorted(np.quantile(values, [.2, .4, .6, .8]), values, side='right')
        strata.update({f'{label} Q{i+1}': bins == i for i in range(5)})
    network = data['network_labels']
    assert network.shape == (400,)
    strata['Within network'] = network[ei] == network[ej]
    strata['Between networks'] = ~strata['Within network']
    return models, y, folds, strata, network, metadata


def predict(X, y, train, test):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1, class_weight='balanced', solver='lbfgs', tol=1e-4, max_iter=3000))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X[train], y[train])
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('A classifier did not converge.')
    prediction = model.predict_proba(X[test])[:, 1]
    assert np.isfinite(prediction).all()
    return prediction


def edge_model(item, y, folds):
    name, spec = item
    oof = np.full(folds.shape, np.nan)
    rows = []
    for repeat in range(20):
        for fold in range(5):
            test = np.flatnonzero(folds[repeat] == fold)
            train = np.flatnonzero(folds[repeat] != fold)
            prediction = predict(spec['X'], y, train, test)
            oof[repeat, test] = prediction
            rows.append(dict(model=name, group='RGIC full' if name == 'RGIC' else name,
                             repeat=repeat, fold=fold, n_features=spec['X'].shape[1],
                             **scores(y[test], prediction)))
    assert np.isfinite(oof).all()
    print(f'Edge CV: {name}', flush=True)
    return name, rows, oof


def add_paired(rows, baseline, index, metrics=('ap', 'roc_auc'), pp=False):
    for row in rows:
        base = baseline[tuple(row[key] for key in index)]
        for metric in metrics:
            field = f'delta_{metric}_pp' if pp else f'delta_{metric}_vs_pearson'
            row[field] = (100 if pp else 1) * (row[metric] - base[metric])


def control_model(control, X, y, folds, base):
    bins = np.searchsorted(np.quantile(X[:, 0], np.linspace(0, 1, 21)[1:-1]),
                           X[:, 0], side='right')
    rows = []
    for repeat in range(20):
        if control == 'Random':
            extra = np.random.default_rng(np.random.SeedSequence([SEED, repeat, 3])).normal(
                size=(len(y), 15))
        else:
            kind = 1 if control == 'Global shuffle' else 2
            rng = np.random.default_rng(np.random.SeedSequence([SEED, repeat, kind]))
            permutation = rng.permutation(len(y)) if kind == 1 else np.arange(len(y))
            if kind == 2:
                for b in range(20):
                    ids = np.flatnonzero(bins == b)
                    permutation[ids] = rng.permutation(ids)
                assert np.array_equal(bins, bins[permutation])
            extra = X[permutation, 2:]
        changed = np.column_stack((X[:, :2], extra))
        for fold in range(5):
            test = np.flatnonzero(folds[repeat] == fold)
            train = np.flatnonzero(folds[repeat] != fold)
            rows.append(dict(control=control, repeat=repeat, fold=fold,
                             **scores(y[test], predict(changed, y, train, test))))
    add_paired(rows, base, ('repeat', 'fold'), pp=True)
    return rows


def node_splits(network):
    ei, ej = np.triu_indices(400, 1)
    splits = []
    for repeat in range(20):
        splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=NODE_SEED + repeat)
        for fold, (train_nodes, test_nodes) in enumerate(splitter.split(np.zeros((400, 1)), network)):
            test_mask = np.zeros(400, dtype=bool)
            test_mask[test_nodes] = True
            train = np.flatnonzero(~test_mask[ei] & ~test_mask[ej])
            test = np.flatnonzero(test_mask[ei] & test_mask[ej])
            assert len(train_nodes) == 300 and len(test_nodes) == 100
            assert len(train) == 44850 and len(test) == 4950
            assert not np.intersect1d(np.unique(np.r_[ei[train], ej[train]]),
                                     np.unique(np.r_[ei[test], ej[test]])).size
            splits.append((repeat, fold, train, test))
    return splits


def node_model(name, X, y, splits):
    rows = [dict(model=name, repeat=repeat, fold=fold,
                 **scores(y[test], predict(X, y, train, test)))
            for repeat, fold, train, test in splits]
    print(f'Node CV: {name}', flush=True)
    return rows


def weight_model(name, X, weight_y, folds):
    rows = []
    for repeat in range(20):
        oof = np.full(len(weight_y), np.nan)
        for fold in range(5):
            test = folds[repeat] == fold
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(X[~test], weight_y[~test])
            oof[test] = model.predict(X[test])
        assert np.isfinite(oof).all()
        rows.append(dict(model=name, repeat=repeat, r2=float(r2_score(weight_y, oof)),
                         mae_log_weight=float(mean_absolute_error(weight_y, oof)),
                         spearman=float(spearmanr(weight_y, oof).statistic)))
    print(f'Weight CV: {name}', flush=True)
    return rows


def average_repeats(rows, categories, metrics):
    output = []
    for value in categories:
        field = 'control' if value in CONTROLS else 'model'
        for repeat in range(20):
            selected = [row for row in rows if row[field] == value and row['repeat'] == repeat]
            assert selected
            output.append({field: value, 'repeat': repeat,
                           **{metric: float(np.mean([row[metric] for row in selected]))
                              for metric in metrics}})
    return output


def run(data, weights, output, workers=3):
    started = time.time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    models, y, folds, strata, network, metadata = load_inputs(data)
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        edge_results = list(pool.map(lambda item: edge_model(item, y, folds), models.items()))
        edge = {name: (rows, oof) for name, rows, oof in edge_results}
        base = {(row['repeat'], row['fold']): row for row in edge['Pearson'][0]}
        all_rows = [row for name in models for row in edge[name][0]]
        add_paired(all_rows, base, ('repeat', 'fold'))
        write_csv(output / 'fold_results.csv', all_rows)

        # Evaluate original out-of-fold predictions within all displayed strata.
        stratified = []
        for name in METHODS:
            for label, mask in strata.items():
                for repeat in range(20):
                    result = scores(y[mask], edge[name][1][repeat, mask])
                    baseline = scores(y[mask], edge['Pearson'][1][repeat, mask])
                    stratified.append(dict(model=name, repeat=repeat, stratum=label,
                        n_edges=int(mask.sum()), n_positive=int(y[mask].sum()), **result,
                        pearson_ap=baseline['ap'], pearson_roc_auc=baseline['roc_auc'],
                        delta_ap_pp=100*(result['ap']-baseline['ap']),
                        delta_roc_auc_pp=100*(result['roc_auc']-baseline['roc_auc'])))
        write_csv(output / 'stratified_results.csv', stratified)

        observed = [dict(control='Observed', repeat=row['repeat'], fold=row['fold'],
                        ap=row['ap'], roc_auc=row['roc_auc'],
                        delta_ap_pp=100*row['delta_ap_vs_pearson'],
                        delta_roc_auc_pp=100*row['delta_roc_auc_vs_pearson'])
                    for row in edge['RGIC'][0]]
        randomized = list(pool.map(lambda control: control_model(control, models['RGIC']['X'],
                                   y, folds, base), CONTROLS[1:]))
        control_rows = observed + [row for result in randomized for row in result]
        controls = average_repeats(control_rows, CONTROLS,
                                   ('ap', 'roc_auc', 'delta_ap_pp', 'delta_roc_auc_pp'))
        for row in controls:
            row['feature_group'] = 'RGIC full'
        write_csv(output / 'control_repeat_results.csv', controls)

        splits = node_splits(network)
        node_results = list(pool.map(lambda name: node_model(name, models[name]['X'], y, splits), METHODS))
        node_rows = [row for result in node_results for row in result]
        node_base = {(row['repeat'], row['fold']): row for row in node_rows if row['model']=='Pearson'}
        add_paired(node_rows, node_base, ('repeat', 'fold'), pp=True)
        nodes = average_repeats(node_rows, METHODS, ('ap', 'roc_auc', 'delta_ap_pp', 'delta_roc_auc_pp'))
        write_csv(output / 'node_repeat_results.csv', nodes)

        matrix = np.load(weights, allow_pickle=False)
        assert matrix.shape == (400, 400) and np.isfinite(matrix).all()
        assert np.array_equal(matrix, matrix.T) and np.all(matrix >= 0) and np.all(np.diag(matrix)==0)
        ei, ej = np.triu_indices(400, 1)
        positive = matrix[ei, ej] > 0
        assert np.array_equal(positive, y == 1)
        weight_y = np.log(matrix[ei, ej][positive])
        weight_results = list(pool.map(lambda name: weight_model(name, models[name]['X'][positive],
                                       weight_y, folds[:, positive]), METHODS))
        weight_rows = [row for result in weight_results for row in result]
        weight_base = {(row['repeat'],): row for row in weight_rows if row['model']=='Pearson'}
        add_paired(weight_rows, weight_base, ('repeat',), metrics=('r2',))
        write_csv(output / 'weight_repeat_results.csv', weight_rows)

    summary = {'fixed_configuration': metadata['fixed_configuration'],
        'input_level': 'Cached HCP group-level features, not raw BOLD',
        'configuration_origin': metadata['configuration_origin'],
        'common_baseline': 'distance + absolute signed-mean Pearson',
        'n_edge_fits':1300, 'n_control_fits':300, 'n_node_fits':720, 'n_weight_fits':900,
        'parameter_search':False, 'elapsed_seconds':time.time()-started, 'comparisons':{}}
    for label, rows, names, metric in [('b_features', all_rows, list(GROUPS), 'roc_auc'),
            ('b_methods', all_rows, METHODS, 'roc_auc'), ('c_node', nodes, METHODS, 'roc_auc'),
            ('c_weight', weight_rows, METHODS, 'r2')]:
        values = {name: np.array([row[metric] for row in rows if row['model']==name]) for name in names}
        base_values = values['Pearson']
        statistics = {name: {'mean_percent':float(v.mean()*100),
            'sd_percent':float(v.std(ddof=1)*100), 'delta_pp_mean':float((v-base_values).mean()*100),
            'delta_pp_sd':float((v-base_values).std(ddof=1)*100)} for name,v in values.items()}
        best = max((name for name in names if name not in ('Pearson','RGIC')),
                   key=lambda name: statistics[name]['delta_pp_mean'])
        summary['comparisons'][label] = {'models':statistics, 'strongest_other':best,
            'rgic_gain_ratio':safe_gain_ratio(statistics['RGIC']['delta_pp_mean'],
                                               statistics[best]['delta_pp_mean'])}
    write_json(output/'summary.json', summary, ensure_ascii=False, allow_nan=True,
               clean=False, newline=True)
    print(f'Complete: {output} ({time.time()-started:.1f}s)', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=INPUTS/'inputs.npz')
    parser.add_argument('--weights', type=Path, default=INPUTS/'sc_weights.npy')
    parser.add_argument('--output', type=Path, default=ROOT/'output'/'main')
    parser.add_argument('--workers', type=int, choices=range(1,7), default=3)
    args = parser.parse_args()
    run(args.data, args.weights, args.output, args.workers)
