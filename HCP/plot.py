"""Draw current HCP statistical panels from freshly computed CSV files."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from reporting_utils import safe_gain_ratio, full_axis_limits

METHODS = ['Pearson', 'RGIC', 'Lag peak', 'Coherence', 'GCMI',
           'Gaussian TE', 'VAR Granger', 'PSI', 'PhiID unique transfer']
FEATURES = ['Pearson', 'Gain', 'Cost-discounted gain', 'Reliability parts',
            'Final-IC features', 'RGIC']
CONTROLS = ['Observed', 'Random', 'Global shuffle', 'Distance-bin shuffle']
STRATA = ([f'Distance Q{i}' for i in range(1, 6)] +
          [f'Pearson Q{i}' for i in range(1, 6)] + ['Within network', 'Between networks'])
INK, BLUE, GRAY, GOLD = '#26313c', '#3b608b', '#8a96a1', '#d19436'


def label(name):
    return name.replace('PhiID', 'ΦID').replace('Final-IC', 'Final IC')


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#e8ecef', lw=.5)
    ax.set_axisbelow(True)


def paired_values(frame, metric, names, keys):
    selected = {n: frame.loc[frame.model.eq(n)].sort_values(keys) for n in names}
    reference = selected['Pearson'][keys].to_numpy()
    assert not selected['Pearson'][keys].duplicated().any()
    assert all(np.array_equal(v[keys].to_numpy(), reference) for v in selected.values())
    values = {n: v[metric].to_numpy(float) * 100 for n, v in selected.items()}
    assert all(len(v) > 1 and np.isfinite(v).all() for v in values.values())
    delta = {n: v - values['Pearson'] for n, v in values.items()}
    best = max((n for n in names if n not in ('Pearson', 'RGIC')), key=lambda n: delta[n].mean())
    return values, delta, safe_gain_ratio(delta['RGIC'].mean(), delta[best].mean())


def local(area, x, y, width=0, height=0):
    left, bottom, w, h = area
    return [left+x*w, bottom+y*h, width*w, height*h]


def caption(figure, area, x, y, text, **kwargs):
    figure.text(*local(area, x, y)[:2], text, **kwargs)


def score_and_gain(container, frame, metric, names, keys, title, area=(0, 0, 1, 1)):
    values, delta, ratio = paired_values(frame, metric, names, keys)
    ax = container.add_axes(local(area, .335, .12, .337, .73))
    gain = container.add_axes(local(area, .742, .12, .233, .73))
    caption(container, area, .335, .96, title, ha='left', fontsize=16, weight='bold')
    ratio_text = 'NA' if ratio is None else f'{ratio:.2f}×'
    caption(container, area, .335, .905, f'RGIC gain / best comparator: {ratio_text}',
            color=BLUE, fontsize=12, weight='bold', ha='left')
    limits = full_axis_limits(np.concatenate([
        np.r_[v, v.mean()-v.std(ddof=1), v.mean()+v.std(ddof=1)] for v in values.values()]))
    dlimits = full_axis_limits(np.concatenate([
        np.r_[v, v.mean()-v.std(ddof=1), v.mean()+v.std(ddof=1)] for v in delta.values()]))
    n, span = len(names), limits[1] - limits[0]
    for row, name in enumerate(names):
        v, dv, y = values[name], delta[name], n - 1 - row
        color = BLUE if name == 'RGIC' else INK if name == 'Pearson' else GRAY
        if metric != 'r2' and np.ptp(v) > 0 and v.std() > 0:
            x = np.linspace(max(limits[0],v.mean()-4*v.std()),
                            min(limits[1],v.mean()+4*v.std()),450)
            density = gaussian_kde(v)(x)
            if density.max() > 0:
                density *= .16 / density.max()
                ax.fill_between(x, y+.25, y+.25+density, color=color, alpha=.22,
                                lw=.5, edgecolor=color)
        jitter = np.random.default_rng(551+row).uniform(-.025, .025, len(v))
        ax.scatter(v, y+.08+jitter, s=5 if len(v) == 100 else 8,
                   color=color, alpha=.52, lw=0, zorder=3)
        ax.errorbar(v.mean(), y-.10, xerr=v.std(ddof=1), fmt='o', ms=1.7,
                    color=color, ecolor=INK, elinewidth=.85, capsize=2.1, zorder=4)
        tx = np.clip(v.mean(), limits[0]+.28*span, limits[1]-.28*span)
        ax.text(tx, y-.37, f'{v.mean():.2f} ± {v.std(ddof=1):.2f}', ha='center',
                va='center', fontsize=13.4, bbox=dict(facecolor='white', edgecolor='none', pad=.07), zorder=5)
        gain.plot([0, dv.mean()], [y-.10, y-.10], color=color, alpha=.3, lw=3)
        gain.errorbar(dv.mean(), y-.10, xerr=dv.std(ddof=1), fmt='o', ms=2.1,
                      color=color, elinewidth=1.1, capsize=2.4, zorder=3)
        dx = np.clip(dv.mean(), dlimits[0]+.38*(dlimits[1]-dlimits[0]),
                     dlimits[1]-.38*(dlimits[1]-dlimits[0]))
        gain.text(dx, y-.37, f'{dv.mean():+.3f}', ha='center', va='center', fontsize=14.5,
                  bbox=dict(facecolor='white', edgecolor='none', pad=.07), zorder=5)
    ax.axvline(values['Pearson'].mean(), color='#70808c', lw=.75, ls=(0, (2, 3)))
    gain.axvline(0, color=INK, lw=.8)
    ax.set_xlim(*limits)
    gain.set_xlim(*dlimits)
    for item in (ax, gain):
        item.set_ylim(-.65, n-.43)
        style(item)
        item.tick_params(axis='x', labelsize=11.8, pad=4)
        item.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_yticks(np.arange(n)[::-1], [label(x) for x in names])
    ax.tick_params(axis='y', length=0, pad=7, labelsize=15.2)
    gain.set_yticks([])
    ax.set_xlabel('R² (%) ↑' if metric == 'r2' else 'ROC-AUC (%) ↑', fontsize=13.2)
    gain.set_xlabel('Δ vs baseline (pp)', fontsize=12.1)


def controls_plot(container, frame, area=(0, 0, 1, 1)):
    ax = container.add_axes(local(area, .46, .16, .51, .70))
    caption(container, area, .02, .96, 'd  Edge correspondence', ha='left', fontsize=16, weight='bold')
    plot_values = []
    for i, name in enumerate(CONTROLS):
        v = frame.loc[frame.control.eq(name)].sort_values('repeat').delta_ap_pp.to_numpy(float)
        assert len(v) == 20
        assert np.isfinite(v).all()
        plot_values.extend(np.r_[v,v.mean()-v.std(ddof=1),v.mean()+v.std(ddof=1)])
        y = 3-i
        if np.ptp(v) > 0 and v.std() > 0:
            x = np.linspace(v.min()-v.std(),v.max()+v.std(),360)
            den = gaussian_kde(v)(x)
            if den.max() > 0:
                den *= .16 / den.max()
                ax.fill_between(x, y+.22, y+.22+den, color=BLUE, alpha=.23, lw=.5, edgecolor=BLUE)
        ax.scatter(v, y+.07+np.random.default_rng(220+i).uniform(-.025, .025, len(v)),
                   s=5, color=BLUE, alpha=.55, lw=0)
        ax.errorbar(v.mean(), y-.14, xerr=v.std(ddof=1), fmt='o', ms=1.6,
                    color=BLUE, ecolor=INK, elinewidth=.8, capsize=2)
    ax.axvline(0, color=INK, lw=.9)
    ax.set(xlim=full_axis_limits(plot_values), ylim=(-.55, 3.6), yticks=[3, 2, 1, 0], yticklabels=CONTROLS)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis='y', length=0, labelsize=12)
    ax.set_xlabel('ΔPR-AUC (pp)', fontsize=12)
    style(ax)


def strata_plot(container, frame, area=(0, 0, 1, 1)):
    ax = container.add_axes(local(area, .18, .17, .81, .65))
    caption(container, area, .02, .96, 'e  Stratified structural prediction',
            ha='left', fontsize=16, weight='bold')
    names = [n for n in METHODS if n != 'Pearson']
    xpos = np.r_[np.arange(5), np.arange(5)+5.5, [11, 12]]
    for i, name in enumerate(names):
        for j, group in enumerate(STRATA):
            v = frame.loc[frame.model.eq(name) & frame.stratum.eq(group), 'delta_ap_pp'].to_numpy(float)
            assert len(v) == 20
            mean, sd = v.mean(), v.std(ddof=1)
            color = BLUE if mean >= 0 else GOLD
            if abs(mean) < .025:
                ax.scatter(xpos[j], i-.16, s=6, marker='+', color=GRAY, lw=.7)
            else:
                ax.scatter(xpos[j], i-.16, s=14*abs(mean), marker='D',
                           facecolor=color if name == 'RGIC' else 'white', edgecolor=color, lw=.85)
            text = '0.0' if abs(mean) < .05 else f'{mean:+.1f}'
            ax.text(xpos[j], i+.28, f'{text}±{sd:.2f}'.replace('±0.', '±.'),
                    ha='center', va='center', fontsize=9.2)
    ax.set(xlim=(-.5, 12.5), ylim=(7.68, -.66), yticks=range(8), yticklabels=[label(n) for n in names])
    ax.set_xticks(xpos, [str(i) for i in range(1, 6)]*2+['Within', 'Between'])
    ax.tick_params(axis='both', length=0, pad=4, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for pos in [4.75, 10.25]:
        ax.axvline(pos, color='#BFCBD5', lw=.85)
    ax.axhline(.58, color=INK, lw=.8)
    for x, text in [(2, 'Distance: short to long'), (7.5, 'Pearson: low to high'), (11.5, 'Network')]:
        ax.text(x, -.92, text, ha='center', fontsize=11)
    handles = [Line2D([], [], ls='', marker='D', color=c, ms=4, label=t)
               for c, t in [(BLUE, 'Positive'), (GOLD, 'Negative')]]
    container.legend(handles=handles, loc='lower left', bbox_to_anchor=local(area, .18, .01)[:2],
                     bbox_transform=container.transFigure, ncol=2, frameon=False, fontsize=10)
    caption(container, area, .53, .035, 'Area: |ΔPR-AUC| (pp)', fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent/'output'/'main')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--panels', action='store_true', help='Also export separate statistical panels.')
    args = parser.parse_args()
    out = args.output or args.results/'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.edgecolor': '#56616a',
                         'axes.linewidth': .7, 'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK, 'svg.fonttype': 'none'})
    fold, node, weight, control, strata = [pd.read_csv(args.results/name) for name in (
        'fold_results.csv', 'node_repeat_results.csv', 'weight_repeat_results.csv',
        'control_repeat_results.csv', 'stratified_results.csv')]
    if 'feature_group' in control:
        control = control[control.feature_group.eq('RGIC full')]
    jobs = [(fold, 'roc_auc', FEATURES, ['repeat', 'fold'], 'b  RGIC feature models'),
            (fold, 'roc_auc', METHODS, ['repeat', 'fold'], 'b  Method scores'),
            (node, 'roc_auc', METHODS, ['repeat'], 'c  Held-out brain regions'),
            (weight, 'r2', METHODS, ['repeat'], 'c  Anatomical connection weight')]
    fig = plt.figure(figsize=(20, 18.5))
    areas = [(0, .66, .5, .34), (.5, .66, .5, .34),
             (0, .32, .5, .34), (.5, .32, .5, .34)]
    for job, area in zip(jobs, areas):
        score_and_gain(fig, *job, area=area)
    controls_plot(fig, control, area=(0, 0, .24, .32))
    strata_plot(fig, strata, area=(.24, 0, .76, .32))
    for extension in ('png', 'svg'):
        fig.savefig(out/f'statistical_results.{extension}', dpi=180, facecolor='white')
    plt.close(fig)
    if args.panels:
        stems = ['b_features', 'b_methods', 'c_node', 'c_weight']
        for stem, job in zip(stems, jobs):
            figure = plt.figure(figsize=(8, 6.8))
            score_and_gain(figure, *job)
            figure.savefig(out/f'{stem}.png', dpi=300, facecolor='white')
            plt.close(figure)
        for stem, draw, frame, size in [('d_controls', controls_plot, control, (4.5, 3.5)),
                                        ('e_strata', strata_plot, strata, (15, 5))]:
            figure = plt.figure(figsize=size)
            draw(figure, frame)
            figure.savefig(out/f'{stem}.png', dpi=300, facecolor='white')
            plt.close(figure)


if __name__ == '__main__':
    main()
