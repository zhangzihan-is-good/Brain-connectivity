"""Descriptive group-matrix and structural-support checks at fixed settings.

These are edge-level summaries, without naive edge-independence P values.
The input contains the retained W1/lambda1/gamma1 group summaries.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import _shared
from rgic_shared import fuse_cached
from shared_paths import workspace_root
from io_shared import write_csv, write_json

ROOT = Path(__file__).resolve().parent
INPUTS = workspace_root() / 'dataset' / 'reproduction_inputs' / 'HCP'
FEATURES = {
    'abs_pearson': r'$|r_0|$', 'mean_gain': '$G$', 'mean_cost': '$C$',
    'freq': '$F$', 'lag_consistency': '$L$', 'sign_consistency': '$S$',
    'reliability': '$R$', 'gain_reliability': '$GR$',
    'gain_cost_evidence': '$E$', 'raw_eta': '$A$', 'eta': r'$\eta$',
    'q_gain': r'$\eta G$', 'q_cost': r'$\eta C$', 'q_evidence': r'$\eta E$',
    'abs_ic': '$|IC|$', 'abs_gain_ic': '$[|IC|-|r_0|]_+$',
    'abs_delta_ic': '$|IC-r_0|$',
}
PLOTTED = ['abs_pearson','mean_gain','mean_cost','freq','lag_consistency',
           'sign_consistency','reliability','eta','abs_ic','abs_delta_ic']


def load_features(data):
    with np.load(data, allow_pickle=False) as z:
        f = {k: z[k].copy() for k in z.files if k != 'metadata_json'}
        metadata = json.loads(str(z['metadata_json']))
    ic = fuse_cached(f['r0'], f['eta'], f['potential'])
    f.update(ic=ic, abs_ic=np.abs(ic), abs_pearson=np.abs(f['r0']),
             abs_gain_ic=np.maximum(np.abs(ic)-np.abs(f['r0']),0),
             abs_delta_ic=np.abs(ic-f['r0']), gain_reliability=f['mean_gain']*f['reliability'])
    assert metadata['fixed_configuration']['W'] == 1
    assert metadata['fixed_configuration']['lambda'] == metadata['fixed_configuration']['gamma'] == 1
    assert all(np.isfinite(f[k]).all() for k in FEATURES)
    return f, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=INPUTS / 'appendix_inputs.npz')
    parser.add_argument('--results', type=Path, default=ROOT / 'output' / 'appendix')
    args = parser.parse_args()
    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    f, metadata = load_features(args.data)
    positive = f['sc_support'].astype(bool)
    rows = []
    for feature, label in FEATURES.items():
        x = np.asarray(f[feature],dtype=float)
        a, b = x[positive], x[~positive]
        pooled_sd = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(x)-2))
        d = float((a.mean()-b.mean())/pooled_sd) if pooled_sd > 0 else float('nan')
        rows.append(dict(feature=feature,label=label,plotted=feature in PLOTTED,
                         n_supported=len(a),n_unsupported=len(b),supported_mean=float(a.mean()),
                         unsupported_mean=float(b.mean()),supported_sd=float(a.std(ddof=1)),
                         unsupported_sd=float(b.std(ddof=1)),pooled_within_group_sd=float(pooled_sd),
                         cohen_d=d,raw_feature_roc_auc=float(roc_auc_score(positive,x))))
    write_csv(out/'feature_support_stats.csv', rows)
    delta = f['ic']-f['r0']
    matrix = dict(n_edges=len(delta),n_supported=int(positive.sum()),
                  pearson_ic_r=float(np.corrcoef(f['r0'],f['ic'])[0,1]),
                  mean_absolute_adjustment=float(np.abs(delta).mean()),
                  median_absolute_adjustment=float(np.median(np.abs(delta))),
                  fraction_unchanged=float(np.mean(delta==0)),
                  pearson_mean=float(f['r0'].mean()),ic_mean=float(f['ic'].mean()),
                  abs_pearson_mean=float(f['abs_pearson'].mean()),abs_ic_mean=float(f['abs_ic'].mean()),
                  plot_features=PLOTTED,tests='Descriptive summaries only; no edge-wise inferential P values',
                  fixed_configuration=metadata['fixed_configuration'])
    write_json(out/'matrix_relationship.json', matrix, ensure_ascii=True, allow_nan=True,
               clean=False, newline=True)
    print(json.dumps(matrix,indent=2))


if __name__=='__main__':
    main()
