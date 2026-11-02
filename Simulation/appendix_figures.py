"""Render the two supplementary simulation figures from saved experiment CSVs.

No experiment definitions or estimator implementations are changed here.
Bootstrap confidence intervals resample repeat-level means, retaining all
simulation conditions within a repeat. Source CSVs remain authoritative.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

INK = '#202020'
BLUE, GOLD, ORANGE = '#0072B2', '#E69F00', '#D55E00'
FIELDS = ['3T', '7T']
FAMILIES = ['attenuation', 'dispersion', 'control']
PROFILES = [(f, g) for f in FIELDS for g in FAMILIES]
CODES = ['3A', '3D', '3C', '7A', '7D', '7C']
REL = {'sync': 'Synchronous', 'stable': 'Stable lag',
       'lag_variable': 'Changing lag', 'sign_variable': 'Changing sign',
       'sporadic': 'Sporadic', 'independent': 'Independent'}
FAM = {'attenuation': 'Attenuation', 'dispersion': 'Dispersion',
       'control': 'No degradation'}
METHODS = ['RGIC', 'Lag peak', 'Gaussian_TE', 'VAR_Granger', 'PSI']
LABEL = {'Gaussian_TE': 'Gaussian TE', 'VAR_Granger': 'VAR Granger',
         'KNN_TE': 'KNN TE', 'Gaussian_TE tuned history': 'Gaussian TE (selected history)',
         'VAR_Granger tuned order': 'VAR Granger (selected order)',
         'PSI raw': 'PSI (raw)',
         'PSI taper-jackknife standardized': 'PSI (standardized)'}
COLORS = {'RGIC': ORANGE, 'Lag peak': BLUE, 'Gaussian_TE': '#009E73',
          'VAR_Granger': '#CC79A7', 'PSI': GOLD, 'KNN_TE': '#666666'}
MARKERS = {'RGIC': 's', 'Lag peak': '^', 'Gaussian_TE': 'v',
           'VAR_Granger': 'D', 'PSI': 'o', 'KNN_TE': 'P'}
STYLES = {'RGIC': '-', 'Lag peak': '-.', 'Gaussian_TE': ':',
          'VAR_Granger': '--', 'PSI': ':'}
BASE_KEY = {'Gaussian_TE tuned history': 'Gaussian_TE',
            'VAR_Granger tuned order': 'VAR_Granger',
            'PSI raw': 'PSI', 'PSI taper-jackknife standardized': 'PSI'}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.2,
    'axes.titlesize': 8.2, 'axes.labelsize': 7.2,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
    'axes.linewidth': .65, 'xtick.major.width': .6,
    'ytick.major.width': .6, 'xtick.major.size': 2,
    'ytick.major.size': 2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': INK,
    'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.titleweight': 'bold', 'axes.unicode_minus': True,
})

SUMMARY = []
CHART_DATA = []


def interval(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        raise ValueError('No finite values to summarize')
    rng = np.random.default_rng(20261004)
    b = v[rng.integers(0, len(v), (5000, len(v)))].mean(axis=1)
    return float(v.mean()), *np.quantile(b, [.025, .975]).tolist()


def summarize(t, metric, figure, panel, method='', field='', scenario='', grain='repeat mean'):
    if isinstance(t, pd.DataFrame):
        values = t.groupby('repeat', sort=True)[metric].mean()
    else:
        values = t
    mean, low, high = interval(values)
    SUMMARY.append(dict(figure=figure, panel=panel, metric=metric, method=method,
                        field=field, scenario=scenario, n=len(values),
                        mean=mean, ci_low=low, ci_high=high, grain=grain))
    CHART_DATA.extend(dict(figure=figure, panel=panel, metric=metric,
                           method=method, field=field, scenario=scenario,
                           repeat=int(repeat), value=float(value), grain=grain)
                      for repeat, value in values.items())
    return mean, low, high


def subset(t, methods=None, track='native', window='physical', scenario='main'):
    q = t[(t.window == window) & (t.track == track)]
    if scenario is not None and 'scenario' in q:
        q = q[q.scenario == scenario]
    return q if methods is None else q[q.method.isin(methods)]


def title(ax, letter, text, panel_x=-.17):
    ax.set_title(text, loc='left', pad=8)
    ax.text(panel_x, 1.065, letter, transform=ax.transAxes,
            fontsize=10.8, fontweight='bold', va='bottom')


def dark_errorbar(ax, mean, low, high, position, color, marker,
                  horizontal=True, size=3.2, open_marker=False):
    xy = (mean, position) if horizontal else (position, mean)
    ax.plot(*xy, marker=marker, color=color, mfc='white' if open_marker else color,
            mec=color, mew=.7, ms=size, ls='none', zorder=3)
    err = [[mean-low], [high-mean]]
    ax.errorbar(*xy, **({'xerr': err} if horizontal else {'yerr': err}),
                fmt='none', ecolor=INK, capsize=2, capthick=.65,
                elinewidth=.65, zorder=4)


def dot_axis(ax, t, methods, figure, panel, metric='persistent_ap', track='native'):
    for j, method in enumerate(methods):
        q = t[t.method == method]
        v = summarize(q, metric, figure, panel, method=method, scenario=track)
        base = BASE_KEY.get(method, method)
        dark_errorbar(ax, *v, j, COLORS[base], MARKERS[base],
                      open_marker=method in BASE_KEY)
    ax.set_yticks(range(len(methods)), [LABEL.get(m, m) for m in methods])
    ax.set_ylim(len(methods)-.45, -.55)
    ax.grid(axis='x', color='#E5E5E5', lw=.45)
    ax.set_axisbelow(True)
    if metric == 'persistent_ap':
        ax.set_xlim(0, .75)
        ax.set_xticks([0, .25, .5, .75])
        ax.set_xlabel('Persistent-direction AP')


def signal_figure(diag, spectra, lag):
    fig, axes = plt.subplots(2, 2, figsize=(190/25.4, 122/25.4))
    fig.subplots_adjust(left=.088, right=.978, bottom=.105, top=.918,
                        hspace=.55, wspace=.38)
    ax = axes[0, 0]
    for field, color, style in [('3T', BLUE, '-'), ('7T', GOLD, '--')]:
        q = spectra[spectra.field == field].groupby('frequency').psd.mean()
        q = q[q.index <= .12]
        ax.plot(q.index, q.values, color=color, ls=style, lw=1.2, label=field)
        CHART_DATA.extend(dict(figure=1, panel='a', metric='psd', field=field,
                               frequency=float(x), value=float(y),
                               grain='frequency mean across first-repeat conditions')
                          for x, y in q.items())
    ax.axvspan(.009, .08, color='#EAF1F5', zorder=-2)
    ax.set_xlim(0, .12)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, .04, .08, .12])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (%²/Hz)')
    ax.legend(frameon=False, fontsize=7, loc='upper right', handlelength=2)
    title(ax, 'a', 'Generated power spectra')

    ax = axes[0, 1]
    data = [diag[diag.field == field].tsnr.values for field in FIELDS]
    boxes = ax.boxplot(data, positions=[0, 1], widths=.4, patch_artist=True,
                       showfliers=False,
                       medianprops={'color': INK, 'linewidth': 1},
                       whiskerprops={'color': INK, 'linewidth': .7},
                       capprops={'color': INK, 'linewidth': .7})
    for box, color in zip(boxes['boxes'], [BLUE, GOLD]):
        box.set(facecolor=color, alpha=.22, edgecolor=color, linewidth=1)
    for i, value in enumerate([30.2, 28.7]):
        ax.plot(i, value, marker='D', mfc='white', mec=INK, ms=4, mew=.7, zorder=5)
    for field in FIELDS:
        a = diag[diag.field == field].tsnr
        SUMMARY.append(dict(figure=1, panel='b', metric='tsnr', field=field,
                            n=len(a), mean=a.mean(), median=a.median(),
                            q25=a.quantile(.25), q75=a.quantile(.75),
                            grain='pair-level summary over 19 records'))
    ax.set_xticks([0, 1], FIELDS)
    ax.set_ylabel('Temporal mean / temporal SD')
    ax.plot([], [], marker='D', mfc='white', mec=INK, mew=.7, ls='none', ms=4,
            label='Prescribed simulation mean')
    ax.legend(frameon=False, fontsize=6.5, loc='lower left', handletextpad=.5)
    ax.set_xlim(-.6, 1.6)
    title(ax, 'b', 'Generated temporal SNR')

    ax = axes[1, 0]
    for i, field in enumerate(FIELDS):
        q = diag[diag.field == field]
        v = summarize(q, 'lag1_acf', 1, 'c', field=field,
                      grain='repeat mean of all 18 conditions and 24 pairs')
        dark_errorbar(ax, *v, i, [BLUE, GOLD][i], ['o', 's'][i], horizontal=False,
                      size=3.2)
    ax.set_xticks([0, 1], ['3T: 0.72 s', '7T: 1.00 s'])
    ax.set_ylabel('Lag-one autocorrelation')
    ax.set_xlim(-.55, 1.55)
    ax.set_ylim(.95, .99)
    ax.set_yticks([.95, .96, .97, .98, .99])
    title(ax, 'c', 'Temporal autocorrelation')

    ax = axes[1, 1]
    for family, color, marker, style in [('attenuation', BLUE, 'o', '-'),
                                         ('dispersion', GOLD, 's', '--'),
                                         ('control', '#666666', '^', ':')]:
        t = lag[(lag.family == family) & (lag.relation == 'stable')]
        a = t.groupby('assigned_absolute_delay_seconds').reference_mean_abs_peak.mean()
        ax.plot(a.index, a.values, marker=marker, ms=3.1, ls=style,
                color=color, lw=1.15, label=FAM[family])
        CHART_DATA.extend(dict(figure=1, panel='d', metric='reference_mean_abs_peak',
                               family=family, assigned_delay_seconds=float(x),
                               value=float(y), grain='mean across stable-world repeats and fields')
                          for x, y in a.items())
    ax.set_xticks([1, 2])
    ax.set_xlim(.85, 2.15)
    ax.set_xlabel('Assigned absolute delay (s)')
    ax.set_ylabel('Reference |lag peak|')
    ax.set_ylim(0, .35)
    ax.legend(frameon=False, fontsize=6.7, loc='lower left', handlelength=1.8)
    title(ax, 'd', 'Ungated lag peaks')
    return fig


def heatmap(fig, bounds, cell, metric, relations, panel, title_text):
    ax = fig.add_axes(bounds)
    q = subset(cell, METHODS, window='W2')
    order = [(f, family, relation) for f, family in PROFILES for relation in relations]
    g = q.groupby(['field', 'family', 'relation', 'method'])[metric].mean()
    v = np.array([[g.get((*key, method), np.nan) for method in METHODS] for key in order])
    if not np.isfinite(v).all():
        raise ValueError('Incomplete heatmap: '+metric)
    # Vector cells keep PDF and SVG fully scalable.
    edge_x = np.arange(len(METHODS)+1)-.5
    edge_y = np.arange(len(order)+1)-.5
    mesh = ax.pcolormesh(edge_x, edge_y, v, shading='flat', cmap='Blues',
                         vmin=0, vmax=1, edgecolors='none', linewidth=0,
                         antialiased=False, rasterized=False)
    ax.set_ylim(len(order)-.5, -.5)
    ax.set_xlim(-.5, len(METHODS)-.5)
    code_lookup = dict(zip(PROFILES, CODES))
    ax.set_yticks(range(len(order)), [code_lookup[(f, family)]+' · '+REL[r]
                                     for f, family, r in order], fontsize=6.5)
    ax.set_xticks(range(len(METHODS)), [LABEL.get(m, m).replace('Lag peak', 'Lag\npeak').replace('Gaussian TE', 'Gaussian\nTE').replace('VAR Granger', 'VAR\nGranger')
                                        for m in METHODS], fontsize=6.2)
    ax.tick_params(length=0, pad=3)
    ax.spines[['left', 'bottom']].set_visible(False)
    for i, key in enumerate(order):
        for j, method in enumerate(METHODS):
            value = v[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=6.1,
                    color='white' if value > .56 else INK)
            CHART_DATA.append(dict(figure=2, panel=panel, metric=metric,
                                   field=key[0], family=key[1], relation=key[2],
                                   method=method, value=float(value),
                                   grain='mean of ten test repeats'))
    for j in range(len(relations), len(order), len(relations)):
        ax.axhline(j-.5, color='white', lw=.6)
    cax = fig.add_axes([bounds[0]+bounds[2]+.012, bounds[1]+.06*bounds[3],
                        .010, .88*bounds[3]])
    cb = fig.colorbar(mesh, cax=cax, ticks=[0, .25, .5, .75, 1])
    cb.solids.set_rasterized(False)
    cb.ax.tick_params(labelsize=6.3, length=1.8, pad=2)
    cb.outline.set_linewidth(.5)
    return ax


def sensitivity(ax, block, sens, prefix, levels, repeats, panel):
    methods = ['RGIC', 'Lag peak', 'Gaussian_TE', 'PSI']
    for i, method in enumerate(methods):
        points = []
        for j, level in enumerate(levels):
            if level == (1 if prefix == 'noise_' else .035):
                q = subset(block, [method])
                q = q[q.repeat < repeats]
            else:
                q = subset(sens, [method], scenario=f'{prefix}{level:g}')
            v = summarize(q, 'persistent_ap', 2, panel, method=method,
                          scenario=f'{prefix}{level:g}')
            x = j+(i-1.5)*.11
            points.append((x, v[0]))
            dark_errorbar(ax, *v, x, COLORS[method], MARKERS[method],
                          horizontal=False, size=3)
        ax.plot(*np.asarray(points).T, color=COLORS[method],
                ls=STYLES[method], lw=.6, zorder=2)
    ax.set_xticks(range(3), [f'{x:g}' for x in levels])
    ax.set_xlim(-.4, 2.4)
    ax.set_ylim(0, .62)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_ylabel('Persistent-direction AP')
    ax.set_xlabel('Residual / latent power ratio' if prefix == 'noise_' else 'Spectral knee (Hz)')
    ax.grid(axis='y', color='#E5E5E5', lw=.4)
    ax.set_axisbelow(True)


def robustness(cell, block, sens, knn, classic):
    fig = plt.figure(figsize=(200/25.4, 249/25.4))
    headings = [(.027, .979, 'a', 'W2 current-direction detection'),
                (.529, .979, 'b', 'W2 calls without assigned direction'),
                (.027, .594, 'c', 'Acquisition profiles'),
                (.366, .594, 'd', 'Delay-estimate error'),
                (.692, .594, 'e', 'Residual power'),
                (.027, .389, 'f', 'Spectral-knee sensitivity'),
                (.529, .389, 'g', 'Nonparametric-TE subset'),
                (.027, .196, 'h', 'Matched-subset native AP'),
                (.529, .196, 'i', 'Reference-supported AP')]
    for x, y, letter, text in headings:
        fig.text(x, y, letter, fontsize=10.8, fontweight='bold', va='top')
        fig.text(x+.026, y-.001, text, fontsize=8, fontweight='bold', va='top')
    heatmap(fig, [.185, .665, .245, .283], cell, 'current_correct_detection',
            ['stable', 'lag_variable', 'sign_variable', 'sporadic'], 'a', headings[0][3])
    heatmap(fig, [.681, .665, .245, .283], cell, 'null_false_alarm',
            ['sync', 'sporadic', 'independent'], 'b', headings[1][3])

    ax = fig.add_axes([.132, .446, .190, .111])
    for j, method in enumerate(METHODS):
        for shift, field, color, marker in [(-.14, '3T', BLUE, 'o'), (.14, '7T', GOLD, 's')]:
            q = subset(block, [method])
            v = summarize(q[q.field == field], 'persistent_ap', 2, 'c',
                          method=method, field=field)
            dark_errorbar(ax, *v, j+shift, color, marker, size=2.8)
    ax.set_yticks(range(5), [LABEL.get(m, m) for m in METHODS], fontsize=6.7)
    ax.set_ylim(4.5, -.5)
    ax.set_xlim(0, .75)
    ax.set_xticks([0, .25, .5, .75])
    ax.set_xlabel('Persistent-direction AP')
    ax.grid(axis='x', color='#E5E5E5', lw=.4)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=BLUE, marker='o', ls='none', ms=3, label='3T'),
                       Line2D([], [], color=GOLD, marker='s', ls='none', ms=3, label='7T')],
              frameon=False, loc='upper right', fontsize=6.1, ncol=2,
              handletextpad=.1, columnspacing=.4, borderpad=0)

    ax = fig.add_axes([.473, .446, .183, .111])
    q = subset(cell, ['RGIC', 'Lag peak', 'Gaussian_TE'])
    q = q[q.relation.isin(['stable', 'lag_variable', 'sign_variable', 'sporadic'])]
    dot_axis(ax, q, ['RGIC', 'Lag peak', 'Gaussian_TE'], 2, 'd', metric='lag_mae_seconds')
    ax.set_xlabel('Absolute delay error (s)')
    ax.xaxis.set_major_locator(MaxNLocator(3))

    ax = fig.add_axes([.786, .446, .195, .111])
    sensitivity(ax, block, sens, 'noise_', [.25, 1, 4], 4, 'e')
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ls='none',
                      ms=2.7, label=LABEL.get(m, m))
               for m in ['RGIC', 'Lag peak', 'Gaussian_TE', 'PSI']]
    ax.legend(handles=handles, fontsize=5.5, ncol=2, loc='upper right',
              frameon=False, columnspacing=.45, handletextpad=.25, borderpad=0)

    ax = fig.add_axes([.132, .256, .309, .105])
    sensitivity(ax, block, sens, 'knee_', [.02, .035, .05], 2, 'f')
    ax.legend(handles=handles, fontsize=6.3, ncol=2, loc='upper right',
              frameon=False, columnspacing=.7, handletextpad=.3, borderpad=0)
    ax = fig.add_axes([.681, .256, .294, .105])
    dot_axis(ax, subset(knn, scenario=None),
             ['RGIC', 'Lag peak', 'Gaussian_TE', 'KNN_TE', 'VAR_Granger', 'PSI'],
             2, 'g')

    methods = ['RGIC', 'Gaussian_TE', 'Gaussian_TE tuned history', 'VAR_Granger',
               'VAR_Granger tuned order', 'PSI raw',
               'PSI taper-jackknife standardized', 'KNN_TE']
    for bounds, panel, track in [([.264, .049, .196, .123], 'h', 'native'),
                                  ([.762, .049, .214, .123], 'i', 'reference')]:
        ax = fig.add_axes(bounds)
        q = subset(classic, track=track, scenario=None)
        if track == 'native':
            q = pd.concat([q, subset(knn, ['KNN_TE'], track, scenario=None)], ignore_index=True)
        dot_axis(ax, q, methods if track == 'native' else methods[:-1], 2, panel, track=track)
        ax.tick_params(axis='y', labelsize=6.4)
    return fig


def save(fig, stem, dpi):
    fig.canvas.draw()
    # Save fixed physical dimensions, so typography matches the declared scale.
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(stem.with_suffix('.'+extension), dpi=dpi,
                    metadata={'Creator': 'RGIC supplementary figure renderer'} if extension == 'pdf' else None)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    files = ['signal_diagnostics', 'spectra', 'lag_diagnostics', 'cell_metrics',
             'block_metrics', 'sensitivity_block_metrics', 'knn_block_metrics',
             'classical_tuning_block_metrics']
    frames = {name: pd.read_csv(args.input/(name+'.csv')) for name in files}
    args.output.mkdir(parents=True, exist_ok=True)
    save(signal_figure(*[frames[name] for name in files[:3]]),
         args.output/'Appendix1Figure1', args.dpi)
    save(robustness(*[frames[name] for name in files[3:]]),
         args.output/'Appendix1Figure2', args.dpi)
    pd.DataFrame(SUMMARY).to_csv(args.output/'appendix_summary.csv', index=False)
    pd.DataFrame(CHART_DATA).to_csv(args.output/'appendix_chart_data.csv', index=False)
    print(str(args.output.resolve()))



if __name__ == '__main__':
    main()
