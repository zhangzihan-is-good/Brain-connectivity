"""Two-page HNU supplementary figures, using newly generated CSV values only."""
import sys
sys.dont_write_bytecode=True
import argparse
import csv
import math
from pathlib import Path

STRENGTH=['RGIC','Pearson','Lag_peak','Coherence','GCMI','Gaussian_TE','VAR_Granger','PSI','PhiID_unique_transfer']
DIRECTION=['RGIC','Lag_peak','Gaussian_TE','VAR_Granger','PSI','PhiID_unique_transfer']
MODES=['retest','temporal']
SHORT={'RGIC':'RGIC','Pearson':'Pearson','Lag_peak':'Lag peak','Coherence':'Coherence','GCMI':'GCMI',
       'Gaussian_TE':'Gaussian TE','VAR_Granger':'VAR Granger','PSI':'PSI','PhiID_unique_transfer':'PhiID UT'}
MODE_COLOR={'retest':'#225D85','temporal':'#D26B30'}
METHOD_COLOR=dict(zip(DIRECTION,['#D26B30','#225D85','#407D9E','#668FA9','#4F7F86','#365B75']))

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def number(value):
    try:return float(value)
    except (TypeError,ValueError):return float('nan')

def value(rows,**keys):
    found=[row for row in rows if all(row.get(key)==val for key,val in keys.items())]
    if len(found)!=1:raise ValueError('Exactly one source row required: '+str(keys))
    return found[0]

def point_panel(ax,summary,kind,representation,methods,title):
    rows=[];extent=[]
    for method in methods:
        for mode in MODES:
            row=value(summary,strategy='shared',scope='equal_direction_mean',kind=kind,representation=representation,
                      endpoint='meanS',method=method,mode=mode)
            point,lo,hi=[100*number(row.get(key)) for key in ['estimate','ci_low','ci_high']]
            rows.append((method,mode,point,lo,hi))
            extent.extend(x for x in [point,lo,hi] if math.isfinite(x))
    low=min(extent+[0.]);high=max(extent+[0.]);span=max(high-low,1.)
    ax.set_xlim(low-.06*span,high+.36*span)
    for method,mode,point,lo,hi in rows:
        y=methods.index(method)+(-.16 if mode=='retest' else .16)
        color=MODE_COLOR[mode]
        if math.isfinite(lo) and math.isfinite(hi):ax.plot([lo,hi],[y,y],color=color,lw=1.5)
        if math.isfinite(point):ax.scatter([point],[y],marker='o' if mode=='retest' else 'D',s=27,color=color,zorder=3)
        ax.text(.99,y,f'{point:.2f}' if math.isfinite(point) else 'NA',transform=ax.get_yaxis_transform(),
                va='center',ha='right',color=color,fontsize=9)
    ax.set_yticks(range(len(methods)));ax.set_yticklabels([SHORT[m] for m in methods]);ax.set_ylim(len(methods)-.5,-.6)
    ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Score × 100')
    ax.grid(axis='x',color='#D9E0E4',lw=.7);ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)

def density_panel(ax,density,kind,representation,title):
    levels=sorted({number(row['density']) for row in density if row['kind']==kind and row['representation']==representation})
    if not levels or not all(math.isfinite(x) for x in levels):raise ValueError('Four finite density levels required')
    for method in DIRECTION:
        for mode in MODES:
            ys=[]
            for level in levels:
                parts=[]
                for scope in ['A_to_B','B_to_A']:
                    found=[r for r in density if r['strategy']=='shared' and r['scope']==scope and r['kind']==kind
                           and r['representation']==representation and r['method']==method and r['mode']==mode
                           and number(r['density'])==level]
                    if len(found)!=1:raise ValueError('Missing/duplicate density source row')
                    parts.append(number(found[0]['meanS']))
                # Equal transfer means; ordinary addition propagates missing values.
                ys.append(100*(parts[0]+parts[1])/2)
            ax.plot([100*x for x in levels],ys,color=METHOD_COLOR[method],ls='-' if mode=='retest' else '--',
                    marker='o' if mode=='retest' else 'D',ms=4,label=SHORT[method] if mode=='retest' else None)
    ax.set_xticks([100*x for x in levels]);ax.set_xlabel('Connection density (%)');ax.set_ylabel('Score × 100')
    ax.set_title(title,loc='left',fontweight='bold');ax.grid(color='#D9E0E4',lw=.7)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)

def global_table(ax,summary):
    ax.axis('off');body=[]
    for method in STRENGTH:
        cells=[SHORT[method]]
        for mode in MODES:
            for endpoint in ['meanS','betweenD','gap','ID']:
                row=value(summary,strategy='shared',scope='equal_direction_mean',kind='global',representation='global',
                          method=method,mode=mode,endpoint=endpoint)
                val=100*number(row['estimate']);cells.append(f'{val:.2f}' if math.isfinite(val) else 'NA')
        body.append(cells)
    table=ax.table(cellText=body,colLabels=['Method','S','D','S−D','ID','S','D','S−D','ID'],cellLoc='center',
                   colWidths=[.25]+[.09375]*8,loc='center',bbox=[0,.02,1,.84])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#D9E0E4');cell.set_linewidth(.5)
        if r==0:cell.set_facecolor('#EEF3F5');cell.set_text_props(fontweight='bold')
    ax.set_title('e  Global strength: reproducibility and participant specificity',loc='left',fontweight='bold',pad=13)
    ax.text(.4375,.91,'Within scans 6–10',transform=ax.transAxes,ha='center',color=MODE_COLOR['retest'])
    ax.text(.8125,.91,'Scans 1–5 to 6–10',transform=ax.transAxes,ha='center',color=MODE_COLOR['temporal'])

def render(summary,density,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    s=read(summary);d=read(density)
    if any(r.get('strategy')!='shared' for r in s+d):raise ValueError('Fixed shared results required')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    targets=[out/'HNU_supplement.pdf',out/'HNU_supplement_page1.png',out/'HNU_supplement_page2.png']
    if any(p.exists() for p in targets):raise ValueError('Supplement figure files already exist; choose another output')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    figures=[]
    f,a=plt.subplots(3,2,figsize=(16.5,18.5));f.subplots_adjust(left=.13,right=.97,top=.93,bottom=.11,wspace=.53,hspace=.43)
    point_panel(a[0,0],s,'pairwise','coverage',DIRECTION,'a  Single-scan support overlap')
    point_panel(a[0,1],s,'consensus','coverage',DIRECTION,'b  Consensus support overlap')
    point_panel(a[1,0],s,'pairwise','conditional',DIRECTION,'c  Single-scan conditional orientation')
    point_panel(a[1,1],s,'consensus','conditional',DIRECTION,'d  Consensus conditional orientation')
    density_panel(a[2,0],d,'pairwise','signed','e  Single-scan directional recurrence')
    density_panel(a[2,1],d,'consensus','signed','f  Consensus directional recurrence')
    f.suptitle('HNU supplementary analyses: support, orientation and density',y=.975,fontsize=17)
    figures.append(f)
    f=plt.figure(figsize=(16.5,18.5));g=f.add_gridspec(3,2,height_ratios=[1,1,1.3],left=.13,right=.97,top=.93,bottom=.11,wspace=.53,hspace=.43)
    point_panel(f.add_subplot(g[0,0]),s,'polarity','polarity',STRENGTH,'a  Polarity-aware strength')
    point_panel(f.add_subplot(g[0,1]),s,'roles','net_cosine',DIRECTION,'b  Net directional role')
    point_panel(f.add_subplot(g[1,0]),s,'roles','degree_cosine',DIRECTION,'c  Unsigned degree')
    density_panel(f.add_subplot(g[1,1]),d,'roles','net_cosine','d  Net-role density sensitivity')
    global_table(f.add_subplot(g[2,:]),s)
    f.suptitle('HNU supplementary analyses: regional roles and participant specificity',y=.975,fontsize=17)
    figures.append(f)
    mode_handles=[Line2D([0],[0],marker='o',color=MODE_COLOR['retest'],label='Within scans 6–10'),
                  Line2D([0],[0],marker='D',color=MODE_COLOR['temporal'],label='Scans 1–5 to 6–10')]
    method_handles=[Line2D([0],[0],color=METHOD_COLOR[m],label=SHORT[m]) for m in DIRECTION]
    with PdfPages(targets[0]) as pdf:
        for i,f in enumerate(figures):
            f.legend(handles=mode_handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
            f.legend(handles=method_handles,loc='lower center',bbox_to_anchor=(.5,.043),ncol=6,frameon=False)
            f.text(.13,.019,'Equal transfer means. Point panels: marginal conditional 95% person-bootstrap intervals. Density curves: points only.',fontsize=9)
            pdf.savefig(f,facecolor='white');f.savefig(targets[i+1],dpi=180,facecolor='white');plt.close(f)
    print('Saved',targets[0])

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary',required=True,type=Path)
    ap.add_argument('--density',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();render(args.summary,args.density,args.output)
