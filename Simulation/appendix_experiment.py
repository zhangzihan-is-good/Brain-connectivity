import argparse, concurrent.futures, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import core
import appendix_estimators as estimators

NON_DIRECTIONAL = ['Pearson', 'Coherence', 'GCMI', 'PhiID_UniqueTransfer']
DIRECT = ['Lag peak', 'RGIC', 'Gaussian_TE', 'VAR_Granger', 'PSI']
DELETIONS = []
PRIMARY = ['Pearson', 'Lag peak', 'RGIC']
CLASSICAL = {'Gaussian_TE': [f'Gaussian_TE history {h}' for h in range(1, 5)], 'VAR_Granger': [f'VAR_Granger order {h}' for h in range(1, 5)], 'PSI': ['PSI raw', 'PSI taper-jackknife standardized']}


def export_edges(frame, path):
    frame.to_csv(path, index=False, compression='gzip')






def external(data, cell, window, targets=False):
    tr = core.FIELDS[cell['field']]['tr']
    md = int(np.floor(2.2 / tr + 1e-10)) if window == 'physical' else 2
    x, y = data['x'], data['y']
    if targets:
        x, y = x[core.N_REFERENCE:], y[core.N_REFERENCE:]
    shape = x.shape[:2]
    out = estimators.compute_baselines_batch(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), tr, max_delay=md)
    result = {}
    for method, p in out.items():
        result[method] = {k: np.asarray(p[k]).reshape(shape) * (tr if k == 'lag' else 1) for k in ('strength', 'direction', 'lag')}
        if targets:
            result[method] = {k: np.concatenate([np.zeros((core.N_REFERENCE, shape[1])), v]) for k, v in result[method].items()}
    return result


def predictions(w, chosen, window, ext):
    tr = core.FIELDS[w['cell']['field']]['tr']
    out = {m: core.estimate(w['stats'], w['R'], m, tr, **{k: chosen.get(m, {}).get(k, 0) for k in ('lam', 'gamma')}) for m in PRIMARY}
    if window == 'physical':
        out['RGIC original'] = core.estimate(w['stats'], w['R'], 'RGIC', tr, 1, .2, seconds_parameters=False)
        out.update({m + ' matched': core.estimate(w['stats'], w['R'], m, tr, **{k: chosen['RGIC'][k] for k in ('lam', 'gamma')}) for m in DELETIONS})
        out.update(ext)
    else:
        out = {m: p for m, p in out.items() if m in DIRECT}
        out.update({m: p for m, p in ext.items() if m in DIRECT})
    return out


def edge_frame(data, cell, repeat, pred, window='physical', scenario='main', reference=True, stats_=None, R=None):
    frames = []
    nr, nt = core.N_REFERENCE, core.N_TARGET
    for method, p in pred.items():
        n = p['direction'].shape[1]
        direction = p['direction'][nr:]
        support = np.mean(np.sign(p['direction'][:nr])[None] == np.sign(direction)[:, None], axis=1) * (np.abs(direction) > 1e-12)
        present, delay = data['present'][nr:, :n], data['actual_d'][nr:, :n]
        base = dict(cell=cell['id'], field=cell['field'], family=cell['family'], relation=cell['relation'], repeat=repeat, record=np.repeat(np.arange(nt), n), edge=np.tile(np.arange(n), nt), method=method, window=window, scenario=scenario, strength=p['strength'][nr:].ravel(), lag=p['lag'][nr:].ravel(), reference_direction_support=support.ravel(), present=present.ravel(), true_delay_seconds=delay.ravel(), true_direction=np.where(present, np.sign(delay), 0).astype(int).ravel(), template_direction=np.tile(np.sign(data['template_d'][:n]).astype(int), nt), persistent_direction=cell['relation'] in ['stable', 'sign_variable'], persistent_signed=cell['relation'] == 'stable', correlation_sign=data['actual_s'][nr:, :n].astype(int).ravel(), kappa=np.tile(data['kappa'][:n], nt))
        for k in ['r0', 'peak', 'G']:
            base[k] = stats_[k][nr:, :n].ravel() if stats_ is not None else np.nan
        base['R'] = R[nr:, :n].ravel() if R is not None else np.nan
        for k in ['adjustment', 'eta', 'A', 'Q', 'signed_strength']:
            base[k] = p[k][nr:].ravel() if k in p else np.nan
        tracks = ['native', 'reference'] if reference and method not in NON_DIRECTIONAL and not method.endswith(' matched') and method != 'RGIC original' else ['native']
        frames.extend(pd.DataFrame(base | dict(track=track, direction=(direction if track == 'native' else direction * support).ravel())) for track in tracks)
    return pd.concat(frames, ignore_index=True)


def thresholds(dev):
    keys = ['field', 'family', 'window', 'method', 'track']
    return pd.DataFrame([dict(zip(keys, key)) | dict(direction_threshold=float(np.quantile(np.abs(g[g.relation.isin(['sync', 'independent'])].direction), .95))) for key, g in dev.groupby(keys, sort=False)])


def block_metrics(pred):
    keys = ['scenario', 'field', 'family', 'repeat', 'window', 'method', 'track']
    rows = []
    for key, g in pred.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        if row['method'] in NON_DIRECTIONAL:
            continue
        score = np.stack([np.maximum(g.direction.values, 0), np.maximum(-g.direction.values, 0)], axis=-1)
        truth = np.stack([g.persistent_direction.values & (g.template_direction.values > 0), g.persistent_direction.values & (g.template_direction.values < 0)], axis=-1)
        rows.append(row | dict(persistent_ap=average_precision_score(truth.ravel(), score.ravel())))
    return pd.DataFrame(rows)


def cell_metrics(pred, th):
    keys = ['field', 'family', 'window', 'method', 'track']
    pred = pred[pred.track == 'native'].merge(th, on=keys, how='left', validate='many_to_one', indicator=True)
    if (pred['_merge'] != 'both').any():
        raise ValueError('Missing development threshold for an evaluation group')
    pred = pred.drop(columns='_merge')
    keys = ['scenario', 'field', 'family', 'relation', 'cell', 'repeat', 'window', 'method', 'track']
    rows = []
    mean = lambda v: float(np.mean(v)) if len(v) else np.nan
    for key, g in pred.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        directional = row['method'] not in NON_DIRECTIONAL
        active, zero = g.present & (g.true_direction != 0), g.true_direction == 0
        detected, correct = np.abs(g.direction) > g.direction_threshold, np.sign(g.direction) == g.true_direction
        rho = []
        if row['window'] == 'physical' and row['relation'] != 'independent':
            for _, rec in g.groupby('record'):
                rec = rec[rec.present]
                if len(rec) >= 4 and np.std(rec.strength) > 1e-14:
                    rho.append(stats.spearmanr(rec.strength, rec.kappa).statistic)
        rows.append(row | dict(current_correct_detection=mean((detected & correct)[active]) if directional else np.nan, null_false_alarm=mean(detected[zero]) if directional else np.nan, strength_spearman=mean(rho), lag_mae_seconds=mean(np.abs(g.lag - g.true_delay_seconds)[active & g.lag.notna()]) if directional else np.nan, adjustment=float(g.adjustment.mean()), mean_eta=float(g.eta.mean())))
    return pd.DataFrame(rows)


def diagnostics(data, cell, repeat):
    cfg, x = core.FIELDS[cell['field']], data['x'].astype(float)
    f, psd = signal.welch(x - 100, fs=1 / cfg['tr'], nperseg=min(512, cfg['n']), axis=-1)
    lag1 = core.pearson(x[..., :-1], x[..., 1:])
    rows = [dict(cell=cell['id'], field=cell['field'], family=cell['family'], relation=cell['relation'], repeat=repeat, edge=e, tsnr=float(np.mean(x[:, e].mean(axis=-1) / x[:, e].std(axis=-1))), lag1_acf=float(lag1[:, e].mean())) for e in range(core.N_PAIRS)]
    spectra = [dict(cell=cell['id'], field=cell['field'], frequency=float(a), psd=float(b)) for a, b in zip(f, psd.mean(axis=(0, 1)))] if repeat == 0 else []
    return rows, spectra


def classical(data, cell, chosen=None):
    tr, n = core.FIELDS[cell['field']]['tr'], min(4, core.N_PAIRS)
    x, y = (data[k][:, :n].reshape(-1, data[k].shape[-1]).astype(float) for k in ('x', 'y'))
    md = int(np.floor(2.2 / tr + 1e-10))
    out = {}
    names = {name for v in CLASSICAL.values() for name in v} if chosen is None else {v['candidate'] for v in chosen.values()} | set(CLASSICAL['PSI'])
    for history in range(1, 5):
        name = f'Gaussian_TE history {history}'
        if name in names:
            out[name] = estimators.te_history(x, y, md, history)
        name = f'VAR_Granger order {history}'
        if name in names:
            f, r = estimators._batch_gc(x, y, history), estimators._batch_gc(y, x, history)
            out[name] = dict(strength=np.maximum(f, r), direction=f - r, lag=np.full(len(x), np.nan))
    out['PSI raw'], out['PSI taper-jackknife standardized'] = estimators.psi_jackknife(x, y, tr)
    return {m: {k: np.asarray(p[k]).reshape(core.N_REFERENCE + core.N_TARGET, n) * (tr if k == 'lag' else 1) for k in ('strength', 'direction', 'lag')} for m, p in out.items()}


def choose_classical(dev):
    scores = block_metrics(dev).groupby(['method', 'track']).persistent_ap.mean()
    selected = {}
    for family, names in CLASSICAL.items():
        best = max(scores.loc[(m, 'native')] for m in names)
        name = next(m for m in names if best - scores.loc[(m, 'native')] <= 1e-12)
        selected[family] = dict(candidate=name, development_persistent_ap=float(scores.loc[(name, 'native')]))
    return selected


def knn_frame(data, cell, repeat):
    tr, n = core.FIELDS[cell['field']]['tr'], min(4, core.N_PAIRS)
    md = int(np.floor(2.2 / tr + 1e-10))
    p = {k: np.zeros((core.N_REFERENCE + core.N_TARGET, n)) for k in ['strength', 'direction', 'lag']}
    for record in range(core.N_REFERENCE, core.N_REFERENCE + core.N_TARGET):
        for edge in range(n):
            v = estimators.compute_knn_te_secondary(data['x'][record, edge], data['y'][record, edge], tr, max_delay=md)['KNN_TE']
            for k in p:
                p[k][record, edge] = v[k] * (tr if k == 'lag' else 1)
    return edge_frame(data, cell, repeat, {'KNN_TE': p}, scenario='knn_subset', reference=False)




def primary_task(task):
    cell, split, repeat, chosen, classic_chosen = task
    data, frames = core.generate(cell, split, repeat), []
    lag_rows = []
    for window in ['physical', 'W2']:
        s = core.candidates(data, core.FIELDS[cell['field']]['tr'], window)
        w = dict(cell=cell, stats=s, R=core.reliability(s))
        p = predictions(w, chosen[window], window, external(data, cell, window))
        frames.append(edge_frame(data, cell, repeat, p, window, reference=window == 'physical', stats_=s, R=w['R']))
        if split == 1 and window == 'physical' and cell['relation'] == 'stable':
            tr, truth = core.FIELDS[cell['field']]['tr'], data['nominal_d'][:core.N_REFERENCE]
            for delay in [1, 2]:
                mask = np.abs(truth) == delay
                if np.any(mask):
                    lag_rows.append(dict(cell=cell['id'], field=cell['field'], family=cell['family'], relation=cell['relation'], repeat=repeat, assigned_absolute_delay_seconds=delay, reference_mean_abs_peak=float(np.abs(s['peak'][:core.N_REFERENCE])[mask].mean())))
    c = edge_frame(data, cell, repeat, classical(data, cell, classic_chosen), scenario='classical_tuning_subset')
    diag, spec = diagnostics(data, cell, repeat) if split else ([], [])
    k = knn_frame(data, cell, repeat) if split else pd.DataFrame()
    return pd.concat(frames, ignore_index=True), c, diag, spec, lag_rows, k


def sensitivity_task(task):
    cell, repeat, ratio, knee, scenario, chosen = task
    data = core.generate(cell, 1, repeat, ratio, knee)
    s = core.candidates(data, core.FIELDS[cell['field']]['tr'])
    R, tr = core.reliability(s), core.FIELDS[cell['field']]['tr']
    ext = external(data, cell, 'physical', targets=True)
    p = {'RGIC': core.estimate(s, R, 'RGIC', tr, **{k: chosen['RGIC'][k] for k in ('lam', 'gamma')}), 'Lag peak': core.estimate(s, R, 'Lag peak', tr)}
    p.update({m: ext[m] for m in ['Gaussian_TE', 'PSI']})
    return edge_frame(data, cell, repeat, p, scenario=scenario, reference=False, stats_=s, R=R)


def apply(pool, f, tasks, label):
    values = []
    for i, value in enumerate(pool.map(f, tasks), 1):
        values.append(value)
        if i % 36 == 0 or i == len(tasks):
            print(f'{label}: {i}/{len(tasks)}', flush=True)
    return values


def original_main(args):
    if min(args.workers, args.dev, args.test) < 1:
        raise ValueError('workers, dev and test must be positive')
    if args.quick:
        args.dev = args.test = 1
    out = args.output.resolve() / 'results'
    out.mkdir(parents=True, exist_ok=True)
    started, chosen, grid = time.time(), {}, []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=threadpool_limits, initargs=(1,)) as pool:
        chosen = {
            'physical': {'RGIC': {'lam': 2.0, 'gamma': 0.45}},
            'W2': {'RGIC': {'lam': 2.25, 'gamma': 0.55}},
        }
        (out / 'selection.json').write_text(json.dumps(dict(selection=chosen), indent=2)+'\n', encoding='utf-8')
        dev = apply(pool, primary_task, [(c, 0, r, chosen, None) for c in core.cells() for r in range(args.dev)], 'development outputs')
        dev_main, dev_classical = (pd.concat([v[i] for v in dev], ignore_index=True) for i in [0, 1])
        th, classic_chosen = thresholds(dev_main), choose_classical(dev_classical)
        print(json.dumps(classic_chosen), flush=True)
        th.to_csv(out / 'development_thresholds.csv', index=False)
        classic_scores = block_metrics(dev_classical)
        classic_scores.to_csv(out / 'development_classical_block_metrics.csv', index=False)
        classical_summary = classic_scores.groupby(['method','track']).persistent_ap.mean().reset_index().to_dict('records')
        (out / 'classical_selection.json').write_text(json.dumps(dict(selected=classic_chosen, candidates=classical_summary, criterion='Native persistent direction AP; independent development signals only'), indent=2)+'\n')
        block_metrics(dev_main).to_csv(out / 'development_block_metrics.csv', index=False)
        export_edges(dev_main, out / 'development_target_edges.csv.gz')
        export_edges(dev_classical, out / 'development_classical_edges.csv.gz')
        del dev, dev_main, dev_classical
        test = apply(pool, primary_task, [(c, 1, r, chosen, classic_chosen) for c in core.cells() for r in range(args.test)], 'test outputs')
        primary, classic = (pd.concat([v[i] for v in test], ignore_index=True) for i in [0, 1])
        for filename, values in [('signal_diagnostics', [r for v in test for r in v[2]]), ('spectra', [r for v in test for r in v[3]]), ('lag_diagnostics', [r for v in test for r in v[4]])]:
            pd.DataFrame(values).to_csv(out / f'{filename}.csv', index=False)
        knn = pd.concat([v[5] for v in test], ignore_index=True)
        del test
        export_edges(primary, out / 'test_target_edges.csv.gz')
        export_edges(classic, out / 'test_classical_edges.csv.gz')
        export_edges(knn, out / 'test_knn_edges.csv.gz')
        cell_metrics(primary, th).to_csv(out / 'cell_metrics.csv', index=False)
        block_metrics(primary).to_csv(out / 'block_metrics.csv', index=False)
        subset = primary[(primary.window == 'physical') & (primary.edge < 4)]
        paired = subset[(subset.track == 'native') & subset.method.isin(['Lag peak', 'RGIC', 'Gaussian_TE', 'VAR_Granger', 'PSI'])].assign(scenario='knn_subset')
        block_metrics(pd.concat([paired, knn], ignore_index=True)).to_csv(out / 'knn_block_metrics.csv', index=False)
        renamed = []
        for family, choice in classic_chosen.items():
            if family == 'PSI':
                continue
            renamed.append(classic[classic.method == choice['candidate']].assign(method={'Gaussian_TE': 'Gaussian_TE tuned history', 'VAR_Granger': 'VAR_Granger tuned order', 'PSI': 'PSI dev-selected estimator'}[family]))
        keep = subset[subset.method.isin(['RGIC', 'Gaussian_TE', 'VAR_Granger'])].assign(scenario='classical_tuning_subset')
        block_metrics(pd.concat([keep, classic[classic.method.isin(CLASSICAL['PSI'])]] + renamed, ignore_index=True)).to_csv(out / 'classical_tuning_block_metrics.csv', index=False)
        del primary, classic, subset, knn, paired, renamed, keep
        tasks = [(c, r, ratio, core.KNEE, f'noise_{ratio:g}', chosen['physical']) for c in core.cells() for r in range(min(4, args.test)) for ratio in [.25, 4]]
        tasks += [(c, r, 1, knee, f'knee_{knee:g}', chosen['physical']) for c in core.cells() for r in range(min(2, args.test)) for knee in [.02, .05]]
        sensitivity = apply(pool, sensitivity_task, tasks, 'sensitivity')
        sensitivity_edges = pd.concat(sensitivity, ignore_index=True)
        export_edges(sensitivity_edges, out / 'sensitivity_target_edges.csv.gz')
        block_metrics(sensitivity_edges).to_csv(out / 'sensitivity_block_metrics.csv', index=False)
    print(f'Completed in {time.time() - started:.1f} seconds: {out}', flush=True)



def main():
    parser = argparse.ArgumentParser(description='Recompute the two simulation appendix figures from fixed random streams.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('outputs/appendix'))
    parser.add_argument('--dev', type=int, default=core.N_DEV)
    parser.add_argument('--test', type=int, default=core.N_TEST)
    parser.add_argument('--quick', action='store_true', help='One repeat for a development smoke run; not the article protocol.')
    original_main(parser.parse_args())


if __name__ == '__main__':
    main()
