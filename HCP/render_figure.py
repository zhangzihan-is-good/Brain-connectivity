#!/usr/bin/env python3
"""Draw the HCP appendix from newly computed diagnostic tables.

Distance remains a common fitted covariate; its coefficient is not displayed.
This renderer does not fit models or select parameters.
"""
from __future__ import annotations
from io import BytesIO
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,LogNorm,TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import _shared
from rgic_shared import fuse_cached
from shared_paths import workspace_root

ROOT=Path(__file__).resolve().parent
INPUTS=workspace_root()/'dataset'/'reproduction_inputs'/'HCP'
DATA_FILE=INPUTS/'appendix_inputs.npz'
RESULTS=ROOT/'output'/'appendix';OUT=RESULTS/'figures'
BLUE='#3B608B';ORANGE='#D19436';INK='#26313C';NEUTRAL='#73808C'
COEFFICIENTS=['abs_pearson','mean_gain','mean_cost','freq','lag_consistency','sign_consistency',
              'reliability','gain_reliability','gain_cost_evidence','raw_eta','eta',
              'q_gain','q_cost','q_evidence','abs_ic','abs_gain_ic']
LABELS={'abs_pearson':r'$|r_0|$','mean_gain':r'$G$','mean_cost':r'$C$',
        'freq':r'$F$','lag_consistency':r'$L$','sign_consistency':r'$S$',
        'reliability':r'$R$','gain_reliability':r'$GR$','gain_cost_evidence':r'$E$',
        'raw_eta':r'$A$','eta':r'$\eta$','q_gain':r'$\eta G$','q_cost':r'$\eta C$',
        'q_evidence':r'$\eta E$','abs_ic':r'$|IC|$','abs_delta_ic':r'$|IC-r_0|$',
        'abs_gain_ic':r'$(|IC|-|r_0|)_+$'}
NETWORKS=[f'N{i}' for i in range(1,8)]


def load_data():
    with np.load(DATA_FILE,allow_pickle=False) as z:
        r0=z['r0'].astype(float);eta=z['eta'].astype(float);potential=z['potential'].astype(float)
        support=z['sc_support'].astype(bool)
    ic=fuse_cached(r0,eta,potential)
    if r0.shape!=(79800,) or not np.isfinite(np.r_[r0,ic]).all():raise ValueError('Invalid signed edge readouts.')
    r=float(np.corrcoef(r0,ic)[0,1])
    description=json.loads((RESULTS/'matrix_relationship.json').read_text())
    if not np.isclose(r,description['pearson_ic_r'],rtol=0,atol=1e-12):raise ValueError('Stored and fresh matrix correlations disagree.')
    stats=pd.read_csv(RESULTS/'feature_support_stats.csv')
    order=description['plot_features']
    effects=stats.set_index('feature').loc[order].reset_index()
    if len(effects)!=10 or not effects.plotted.eq(True).all():
        raise ValueError('Expected ten descriptive support contrasts.')
    coef=pd.read_csv(RESULTS/'coefficient_summary.csv')
    full=coef[coef['group'].eq('RGIC full')]
    if len(full)!=17 or 'distance' not in full.feature.tolist():raise ValueError('The fitted full model must retain 17 predictors including distance.')
    coef=full.set_index('feature').loc[COEFFICIENTS].reset_index()
    if len(coef)!=16 or not np.isfinite(coef[['coefficient_mean','coefficient_sd']].to_numpy(float)).all():
        raise ValueError('Expected sixteen visible conditional coefficients.')
    pairs=pd.read_csv(RESULTS/'network_pair_summary.csv')
    if len(pairs)!=56 or set(pairs['group'])!={'Reliability parts','RGIC full'}:
        raise ValueError('Expected both sets of all28 network pairs.')
    for group,rows in pairs.groupby('group'):
        if len(rows)!=28 or rows[['network_index_i','network_index_j']].drop_duplicates().shape[0]!=28:
            raise ValueError('Missing or duplicated network pairs.')
        if not (rows.network_index_i<=rows.network_index_j).all():raise ValueError('Expected unique unordered upper-triangle pairs.')
        if set(rows.network_i).union(rows.network_j)!=set(NETWORKS):raise ValueError('Network IDs must be N1-N7.')
    valid=pairs.valid.eq(True)&pairs.n_positive.gt(0)
    if not np.isfinite(pairs.loc[valid,['delta_ap_pp_mean','delta_ap_pp_sd']].to_numpy(float)).all():
        raise ValueError('Nonfinite valid network effects.')
    limit=max(1,float(np.ceil(np.max(np.abs(pairs.loc[valid,'delta_ap_pp_mean'])))))
    return r0,ic,r,support,effects,coef,pairs,description,limit


def atomic_write(path,payload):
    temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('wb') as handle:
        handle.write(payload);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def export(fig,path):
    b=BytesIO();fig.savefig(b,format=path.suffix.lstrip('.'),dpi=400,facecolor='white')
    payload=b.getvalue()
    atomic_write(path,payload)
    return path


def style_axis(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x',length=3,width=.6)
    ax.tick_params(axis='y',length=0,pad=4)


def network_matrix(ax,frame,cmap,norm):
    values=np.ma.masked_all((7,7),dtype=float)
    for _,row in frame.iterrows():
        i,j=int(row.network_index_i),int(row.network_index_j)
        if row['valid'] and row.n_positive>0:values[i,j]=float(row.delta_ap_pp_mean)
    image=ax.imshow(values,cmap=cmap,norm=norm,aspect='equal',interpolation='nearest')
    for _,row in frame.iterrows():
        i,j=int(row.network_index_i),int(row.network_index_j)
        valid=bool(row['valid']) and row.n_positive>0
        value=float(row.delta_ap_pp_mean) if valid else np.nan
        label='NA' if not valid else ('0.0' if abs(value)<.05 else f'{value:+.1f}')
        if row.n_positive<20:label+='*'
        color='white' if valid and abs(value)>.72*norm.vmax else INK
        ax.text(j,i,label,ha='center',va='center',fontsize=6.6,color=color)
        ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='none',edgecolor='white',lw=.7))
    ax.set_xticks(range(7),NETWORKS);ax.set_yticks(range(7),NETWORKS)
    ax.tick_params(length=0,pad=3,labelsize=6.8)
    ax.set_xlim(-.5,6.5);ax.set_ylim(6.5,-.5)
    for spine in ax.spines.values():spine.set_visible(False)
    return image


def draw_effect_panel(ax,effects):
    """Keep every effect row visible, including zero, tiny and NA effects."""
    values=effects.cohen_d.to_numpy(float)
    finite=values[np.isfinite(values)]
    low=min(0.,float(finite.min())) if finite.size else 0.
    high=max(0.,float(finite.max())) if finite.size else 0.
    span=max(high-low,1.)
    offset=.035*span
    for i,value in enumerate(values):
        if not np.isfinite(value):
            ax.text(offset,i,'NA',ha='left',va='center',fontsize=7,color=NEUTRAL)
            continue
        ax.barh(i,value,height=.60,color=BLUE if value>=0 else ORANGE,zorder=2)
        ax.text(value+(offset if value>=0 else -offset),i,f'{value:+.2f}',
                ha='left' if value>=0 else 'right',va='center',fontsize=7,color=INK)
    ax.set_yticks(range(len(values)),[LABELS[f] for f in effects.feature])
    ax.set_ylim(len(values)-.5,-.5)
    ax.set_xlim(low-.20*span,high+.20*span)
    ax.xaxis.set_major_locator(MaxNLocator(5));ax.axvline(0,color=INK,lw=.7)
    ax.set_xlabel("Cohen's $d$",labelpad=3);style_axis(ax)
    return dict(limits=ax.get_xlim(),offset=offset,rows=len(values),n_NA=int((~np.isfinite(values)).sum()))


def main():
    global DATA_FILE, RESULTS, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA_FILE)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    DATA_FILE=args.data;RESULTS=args.results;OUT=args.output or RESULTS/'figures'
    r0,ic,r,support,effects,coef,pairs,description,limit=load_data()
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'text.color':INK,
                         'axes.edgecolor':INK,'axes.labelcolor':INK,'axes.labelsize':8,
                         'axes.linewidth':.65,'xtick.color':INK,'ytick.color':INK,
                         'xtick.labelsize':7,'ytick.labelsize':7.5,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(190/25.4,200/25.4),facecolor='white')
    axa=fig.add_axes([.14,.665,.275,.26125])
    acbar=fig.add_axes([.435,.665,.012,.26125])
    axb=fig.add_axes([.66,.665,.30,.26125])
    axc=fig.add_axes([.22,.08,.32,.485])
    axd=fig.add_axes([.66,.335,.26,.247])
    axe=fig.add_axes([.66,.030,.26,.247])
    ncbar=fig.add_axes([.945,.030,.012,.552])
    fig.text(.025,.986,'Supplementary HCP diagnostics',fontsize=11,weight='bold',va='top')
    for letter,x,y in [('a',.025,.946),('b',.555,.946),('c',.025,.601),('d',.575,.603),('e',.575,.293)]:
        fig.text(x,y,letter,fontsize=12,weight='bold')
    fig.text(.14,.948,'Matrix relationship',fontsize=9,weight='bold')
    fig.text(.66,.948,'Structural support',fontsize=9,weight='bold')
    fig.text(.22,.603,'Conditional model weights',fontsize=8.8,weight='bold')
    fig.text(.79,.605,'Reliability parts',fontsize=8.8,weight='bold',ha='center')
    fig.text(.79,.295,'RGIC',fontsize=8.8,weight='bold',ha='center')
    # a: signed group-level matrices; density is descriptive, not classifier AP.
    lo=min(float(r0.min()),float(ic.min()));hi=max(float(r0.max()),float(ic.max()))
    margin=.035*(hi-lo);limits=(lo-margin,hi+margin)
    sequential=LinearSegmentedColormap.from_list('white_blue',['#F0F4F8',BLUE],N=257)
    density=axa.hexbin(r0,ic,gridsize=43,mincnt=1,extent=[*limits,*limits],
                       cmap=sequential,norm=LogNorm(),linewidths=0,rasterized=True)
    axa.plot(limits,limits,color=INK,lw=.8,ls=(0,(3,3)),zorder=3)
    axa.set_xlim(*limits);axa.set_ylim(*limits);axa.set_aspect('equal',adjustable='box')
    axa.set_xlabel(r'Pearson FC ($r_0$)',labelpad=3);axa.set_ylabel(r'RGIC ($IC$)',labelpad=3)
    axa.xaxis.set_major_locator(MaxNLocator(nbins=4));axa.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axa.text(.035,.965,fr'Pearson $r={r:.3f}$',transform=axa.transAxes,fontsize=7.6,
             va='top',ha='left',bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':1})
    axa.tick_params(length=3,width=.6)
    for spine in axa.spines.values():spine.set_linewidth(.65)
    counts=fig.colorbar(density,cax=acbar)
    count_values=np.asarray(density.get_array())
    count_ticks=[10**k for k in range(int(np.ceil(np.log10(count_values.min()))),int(np.floor(np.log10(count_values.max())))+1)]
    counts.set_ticks(count_ticks);counts.set_ticklabels([str(int(t)) for t in count_ticks])
    counts.outline.set_visible(False);counts.ax.tick_params(length=2,width=.5,labelsize=6.2,pad=2)
    counts.ax.minorticks_off();fig.text(.441,.940,'Edges',fontsize=6.8,ha='center')
    # b: unadjusted class contrasts only; no edge-level inferential error bars.
    draw_effect_panel(axb,effects)
    # c: weights from standardized predictors in the unchanged full classifier.
    for i,(_,row) in enumerate(coef.iterrows()):
        value=float(row.coefficient_mean);color=BLUE if value>=0 else ORANGE
        axc.errorbar(value,i,xerr=row.coefficient_sd,fmt='D',color=color,mfc=color,mec=color,
                     ms=3.5,mew=.7,elinewidth=.8,capsize=1.6,capthick=.65,zorder=3)
    axc.set_yticks(range(16),[LABELS[f] for f in coef.feature]);axc.set_ylim(15.5,-.5)
    low=float((coef.coefficient_mean-coef.coefficient_sd).min());high=float((coef.coefficient_mean+coef.coefficient_sd).max());span=high-low
    axc.set_xlim(low-.10*span,high+.10*span);axc.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
    axc.axvline(0,color=INK,lw=.7);axc.grid(axis='x',color='#E6E9EC',lw=.5,zorder=0)
    axc.set_xlabel('Standardized coefficient',labelpad=4);style_axis(axc)
    # d/e: one upper-triangle cell per unordered network pair; no pair refitting.
    diverging=LinearSegmentedColormap.from_list('gold_white_blue',[ORANGE,'#FFFFFF',BLUE],N=257)
    diverging.set_bad('#FFFFFF',alpha=1)
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    rparts=pairs[pairs['group'].eq('Reliability parts')];fullpairs=pairs[pairs['group'].eq('RGIC full')]
    netimage=network_matrix(axd,rparts,diverging,norm);network_matrix(axe,fullpairs,diverging,norm)
    shared=fig.colorbar(netimage,cax=ncbar,ticks=[-limit,0,limit])
    shared.outline.set_visible(False);shared.ax.tick_params(length=2,width=.5,labelsize=6.6,pad=2)
    fig.text(.951,.605,'ΔPR-AUC\n(pp)',fontsize=6.6,ha='center',va='bottom',linespacing=1)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    clipped=[]
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:
                clipped.append({'text':text.get_text(),'bbox':list(box.bounds)})
    if clipped:raise ValueError(f'Clipped text: {clipped}')
    centres=density.get_offsets();numbers=np.asarray(density.get_array())
    hexdata=pd.DataFrame({'pearson_bin_centre':centres[:,0],'ic_bin_centre':centres[:,1],'n_edges':numbers})
    atomic_write(OUT/'panel_a_hexbin_source.csv',hexdata.to_csv(index=False).encode('utf-8'))
    rows=[dict(panel='a',item='Signed matrix Pearson correlation',mean=r,sd=np.nan,
               unit='Pearson r',n=int(len(r0)),statistical_unit='79800 edge descriptions')]
    for _,v in effects.iterrows():
        rows.append(dict(panel='b',item=v.feature,mean=v.cohen_d,sd=np.nan,unit='Cohen d',
                         n=int(v.n_supported+v.n_unsupported),statistical_unit='Descriptive SC support contrast'))
    for _,v in coef.iterrows():
        rows.append(dict(panel='c',item=v.feature,mean=v.coefficient_mean,sd=v.coefficient_sd,
                         unit='Standardized logistic coefficient',n=100,statistical_unit='100 test-fold model fits'))
    for _,v in pairs.iterrows():
        rows.append(dict(panel='d' if v['group']=='Reliability parts' else 'e',item=f'{v.network_i}-{v.network_j}',
                         mean=v.delta_ap_pp_mean if v['valid'] else np.nan,sd=v.delta_ap_pp_sd if v['valid'] else np.nan,
                         unit='AP percentage points',n=20,statistical_unit='20 paired repeat effects',
                         n_edges=int(v.n_edges),n_positive=int(v.n_positive),rare_support=bool(v.n_positive<20)))
    sourcedata=pd.DataFrame(rows);assert len(sourcedata)==83
    atomic_write(OUT/'figure_source_data.csv',sourcedata.to_csv(index=False).encode('utf-8'))
    effects.to_csv(OUT/'panel_b_effect_sizes.csv',index=False)
    coef.to_csv(OUT/'panel_c_coefficients.csv',index=False)
    pairs.to_csv(OUT/'panels_de_network_pairs.csv',index=False)
    for ext in ['png','pdf','svg']:export(fig,OUT/f'RGIC_HCP_supplementary.{ext}')
    caption=('Supplementary HCP diagnostics for the fixed configuration. '
             'a, Signed group Pearson and RGIC connections; hexagons show edge counts on a logarithmic scale, '
             'the dashed line indicates equality, and r is their descriptive Pearson correlation. '
             'b, Descriptive SC-supported versus unsupported contrasts, expressed as Cohen d using pooled within-class SD; positive values indicate larger supported-edge values. '
             'c, Conditional standardized logistic weights in the full model, mean±SD across 100 fold fits. Distance remains a model covariate but is omitted from display; collinear weights are not independent component effects. '
             'd,e, Network-pair AP changes for reliability parts and RGIC relative to the same-pair Pearson baseline, using global held-out predictions. '
             'Upper triangles contain all 28 unique pairs; numbers are means across 20 paired repeats and SDs are supplied in source data. '
             'N1-N7 are network codes, not assigned anatomical names. Asterisks indicate fewer than 20 SC-positive edges. '
             'All analyses retain W1, lambda=1 and gamma=1; this is not a new independent cohort validation.')
    (OUT/'figure_caption_en.txt').write_text(caption+'\n',encoding='utf-8')
    plt.close(fig)
    print('Updated supplementary diagnostics exported as PNG/PDF/SVG; 83 summary rows and complete hexbin source.',flush=True)

if __name__=='__main__':main()
