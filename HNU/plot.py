"""Static source-backed five-task paired-paradigm figures from final CSV only.

No estimator, archived experiment, array, model fitting or inferential routine
is imported. All ranks and intervals are read unchanged from the CSV export.
The renderer is deliberately separate from every frozen scientific source map.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
KINDS=['local','global','pairwise','consensus','highgain']
MODES=['retest','temporal']
SCOPES=['A_to_B','B_to_A','equal_direction_mean','A_selected_all29','B_selected_all29']
STRENGTH=['RGIC','Lag_peak','Gaussian_TE','VAR_Granger','PSI','PhiID_unique_transfer','Pearson','PureLag_including_zero','Coherence','GCMI']
DIRECTION=STRENGTH[:6]
TITLES={'local':'Local partner rank','global':'Global strength rank','pairwise':'Pairwise unit arrows',
        'consensus':'Arrow consensus','highgain':'High lag-gain rank'}
SCOPE_LABELS={'A_to_B':'A14 to B15','B_to_A':'B15 to A14','equal_direction_mean':'Two-transfer mean',
              'A_selected_all29':'A-selected all29 descriptive','B_selected_all29':'B-selected all29 descriptive'}
SHORT={'RGIC':'RGIC','Lag_peak':'Lag peak','Gaussian_TE':'Gaussian TE','VAR_Granger':'VAR Granger',
       'PSI':'PSI','PhiID_unique_transfer':'PhiID UT','Pearson':'Pearson',
       'PureLag_including_zero':'PureLag incl 0','Coherence':'Coherence','GCMI':'GCMI'}
COLORS=dict(zip(STRENGTH,['#D26B30','#225D85','#407D9E','#668FA9','#4F7F86','#365B75',
                          '#7999AC','#557889','#315C66','#779A94']))
FONT=18.0
WIDTH=16.5
HEIGHT=31.0


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    p=Path(path).resolve()
    if p.suffix.lower()!='.csv':raise ValueError('Only frozen CSV is a plotting input')
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def val(row,key):
    try:return float(row.get(key,''))
    except (TypeError,ValueError):return float('nan')


def truth(value):return str(value).lower() in ('true','1')


def rank_label(row):
    a,b=val(row,'rank_best'),val(row,'rank_worst')
    if not math.isfinite(a) or not math.isfinite(b):return 'NA'
    return str(int(a)) if a==b else str(int(a))+'–'+str(int(b))


def context(row):return tuple(row.get(k,'') for k in ('strategy','scope','mode','kind'))


def methods(kind):return DIRECTION if kind in ('pairwise','consensus') else STRENGTH


def sort_rows(rows,key):
    # Point ordering follows each context's actual estimate; RGIC has no fixed
    # privileged position. Stable lexical ties do not alter reported ranks.
    return sorted(rows,key=lambda r:(not math.isfinite(val(r,key)), -val(r,key) if math.isfinite(val(r,key)) else 0,
                                      r.get('method',r.get('comparator',''))))


def validate(rows,paired=False):
    required={'strategy','scope','mode','kind','representation','endpoint','main','estimate'}
    required.add('comparator' if paired else 'method')
    if not rows or not required.issubset(rows[0]):raise ValueError('Final export schema is missing fields: '+str(required-set(rows[0] if rows else [])))
    grouped=defaultdict(list)
    for r in rows:
        if r.get('strategy')!='shared':
            raise ValueError('Only the fixed shared strategy is accepted')
        if truth(r['main']):grouped[context(r)].append(r)
    for key,group in grouped.items():
        expected=set(methods(key[3]))-({'RGIC'} if paired else set())
        actual=[r['comparator' if paired else 'method'] for r in group]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:
            raise ValueError('Complete method universe required, including NA: '+str(key))
    return grouped


def axes_text_inside(fig):
    from matplotlib.text import Text
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    failures=[];minfont=math.inf
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():continue
        minfont=min(minfont,text.get_fontsize())
        box=text.get_window_extent(renderer)
        # Ignore dormant offset labels with a zero-area box.
        if box.width<=0 or box.height<=0:continue
        if box.x0 < -1 or box.y0 < -1 or box.x1>width+1 or box.y1>height+1:
            failures.append(dict(text=text.get_text(),bbox=list(map(float,box.extents))))
    return dict(min_source_font_pt=minfont,min_font_at_18_3cm_width_pt=minfont*((18.3/2.54)/WIDTH),
                all_text_inside_canvas=not failures,outside_text=failures)


def plot_panel(ax,rows,kind,mode,paired=False,wide=False):
    rows=sort_rows(rows,'estimate')
    labels=[];y=list(range(len(rows)))
    for index,row in enumerate(rows):
        method=row['comparator' if paired else 'method'];estimate=val(row,'estimate')
        label=SHORT[method]+('' if paired else '  ['+rank_label(row)+']')
        labels.append(label)
        if paired:
            lo=val(row,'both_all29_main148_low' if wide else 'main74_low')
            hi=val(row,'both_all29_main148_high' if wide else 'main74_high')
        else:lo,hi=val(row,'ci_low'),val(row,'ci_high')
        if not math.isfinite(estimate):
            ax.text(.98,index,'NA',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=FONT)
            continue
        if math.isfinite(lo) and math.isfinite(hi):
            ax.plot([lo,hi],[index,index],color=COLORS[method],lw=1.8,zorder=1)
            ax.plot([lo,lo],[index-.08,index+.08],color=COLORS[method],lw=1.6)
            ax.plot([hi,hi],[index-.08,index+.08],color=COLORS[method],lw=1.6)
        ax.scatter([estimate],[index],s=62 if method=='RGIC' else 44,color=COLORS[method],zorder=3)
    ax.set_yticks(y);ax.set_yticklabels(labels,fontsize=FONT)
    for tick,row in zip(ax.get_yticklabels(),rows):tick.set_color(COLORS[row['comparator' if paired else 'method']])
    ax.set_ylim(len(rows)-.5,-.6)
    ax.tick_params(axis='x',labelsize=FONT)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(axis='x',color='#D9E0E4',linewidth=.7,zorder=0)
    ax.set_axisbelow(True);ax.margins(x=.1)
    if paired:ax.axvline(0,color='#333333',linewidth=1.0,zorder=0)
    ax.set_xlabel('RGIC minus comparator' if paired else 'Own-person similarity S',fontsize=FONT)
    # Matplotlib creates dormant tick Text objects beyond the current limits;
    # retain only ticks that are actually inside the axis for geometry QA.
    xmin,xmax=ax.get_xlim()
    ax.set_xticks([x for x in ax.get_xticks() if xmin <= x <= xmax])
    ax.set_xlim(xmin,xmax)
    return rows


def make_one(grouped,strategy,scope,out,paired=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':FONT,'axes.labelsize':FONT,
                         'xtick.labelsize':FONT,'ytick.labelsize':FONT,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(5,2,figsize=(WIDTH,HEIGHT))
    fig.subplots_adjust(left=.185,right=.975,top=.955,bottom=.085,wspace=.86,hspace=.52)
    figure_rows=[]
    for ri,kind in enumerate(KINDS):
        for ci,mode in enumerate(MODES):
            key=(strategy,scope,mode,kind)
            if key not in grouped:raise ValueError('Missing main context '+str(key))
            rows=plot_panel(axes[ri,ci],grouped[key],kind,mode,paired,scope in ('A_selected_all29','B_selected_all29'))
            axes[ri,ci].set_title(chr(97+ri*2+ci)+'  '+TITLES[kind]+'\n'+('Within late5' if mode=='retest' else 'Early5 to late5'),
                                 fontsize=FONT+.5,pad=13,loc='left')
            figure_rows.extend(rows)
    fig.suptitle(('Paired comparisons' if paired else 'Five paired retest questions')+' | '+SCOPE_LABELS[scope]+' | '+strategy,
                 fontsize=FONT+2,y=.985)
    if paired:
        note='Intervals: original '+('148 contrasts across two all29 selectors.' if scope in ('A_selected_all29','B_selected_all29') else '74 contrasts across five tasks and two modes.')
    else:note='Method intervals: marginal conditional 95% whole-person CI. Brackets: supplied point rank, including ties.'
    extra='All29 is descriptive; selectors are separate.' if scope in ('A_selected_all29','B_selected_all29') else 'Two-transfer mean averages endpoints and aligned draws; it is not pooled29.'
    fig.text(.02,.044,note,fontsize=FONT-0.5,ha='left')
    fig.text(.02,.030,extra,fontsize=FONT-0.5,ha='left')
    fig.text(.02,.016,'PhiID UT = unique transfer. NA methods remain visible.',fontsize=FONT-0.5,ha='left')
    qa=axes_text_inside(fig)
    stem=('paired_' if paired else 'main_')+strategy+'_'+scope
    paths=[]
    for suffix in ('.png','.pdf'):
        p=out/(stem+suffix);fig.savefig(p,dpi=200,facecolor='white');paths.append(p)
    plt.close(fig)
    return dict(name=stem,strategy=strategy,scope=scope,panel_grid=[5,2],source_rows=len(figure_rows),
                interval='both_all29_main148' if paired and scope in ('A_selected_all29','B_selected_all29') else ('main74' if paired else 'marginal95'),
                figure_inches=[WIDTH,HEIGHT],qa_geometry=qa,
                outputs={p.as_posix():sha(p) for p in paths})


def render(summary,paired,output):
    source=Path(summary).resolve();pairsource=Path(paired).resolve() if paired else None
    rows=read(source);grouped=validate(rows);pairgrouped=validate(read(pairsource),True) if pairsource else None
    out=Path(output).resolve()
    if out==ROOT/'code' or ROOT/'code' in out.parents:raise ValueError('Never write plots into code')
    out.mkdir(parents=True,exist_ok=True)
    primary=[r for r in rows if truth(r['main'])]
    fields=list(rows[0])
    with (out/'mainpoints.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(primary)
    artifacts=[]
    for strategy in sorted({k[0] for k in grouped}):
        for scope in SCOPES:
            if not any(k[:2]==(strategy,scope) for k in grouped):continue
            artifacts.append(make_one(grouped,strategy,scope,out,False))
            if pairgrouped:artifacts.append(make_one(pairgrouped,strategy,scope,out,True))
    manifest=dict(status='rendered',plot_source_sha256=sha(__file__),
        input_sha256_map={source.as_posix():sha(source),**({pairsource.as_posix():sha(pairsource)} if pairsource else {})},
        data_contract=dict(filter='main=True only; meanS supplied by exporter',methods='All10 strength and all6 native directional methods, including NA',
                           order='Descending actual context point; no forced RGIC-first ordering',ranks='Copied rank_best/worst; not recomputed',
                           inference='CI fields are copied; no bootstrap or scoring performed'),
        mainpoints_sha256=sha(out/'mainpoints.csv'),artifacts=artifacts)
    (out/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(not a['qa_geometry']['all_text_inside_canvas'] for a in artifacts):
        raise RuntimeError('Outside-canvas text detected; inspect figure_manifest before delivery')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True,type=Path)
    parser.add_argument('--paired',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=render(args.summary,args.paired,args.output)
    print(json.dumps(dict(status=result['status'],figure_sets=len(result['artifacts']),output=args.output.as_posix()),ensure_ascii=False))


if __name__=='__main__':main()
