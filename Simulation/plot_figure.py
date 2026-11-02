#!/usr/bin/env python3
"""Draw the complete main simulation figure from freshly computed result tables."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

INK='#22272B'; BLUE='#0072B2'; GOLD='#D55E00'
FIELDS=['3T','7T']; FAMILIES=['attenuation','dispersion','control']
RELATIONS=['sync','stable','lag_variable','sign_variable','sporadic','independent']
PROFILES=[(f,g) for f in FIELDS for g in FAMILIES]; CODES=['3A','3D','3C','7A','7D','7C']
REL=dict(zip(RELATIONS,['Synchronous','Stable lag','Changing lag','Changing sign','Sporadic','Independent']))
DIRECT=['Lag peak','RGIC','Gaussian_TE','VAR_Granger','PSI']
STRENGTH=['RGIC','Pearson','Lag peak','Gaussian_TE','VAR_Granger','PSI','Coherence','GCMI','PhiID_UniqueTransfer']
LABEL={'Gaussian_TE':'Gaussian TE','VAR_Granger':'VAR Granger','PhiID_UniqueTransfer':'ΦID unique transfer'}
COLORS=dict(zip(STRENGTH,[GOLD,'#666666',BLUE,'#009E73','#CC79A7','#E69F00','#56B4E9','#597194','#8E705D']))
STYLES=dict(zip(DIRECT,[(BLUE,'^','-.'),(GOLD,'s','-'),('#009E73','v',':'),('#CC79A7','D','--'),('#E69F00','o',':')]))
COMPONENTS=['Reliability','Gain','Cost','Compensation']
COMP_COLORS=['#0072B2','#009E73','#CC79A7','#E69F00']; COMP_MARKERS=['o','s','^','D']
MUTED_AP=LinearSegmentedColormap.from_list('muted_ap',['#F0EDE9','#D7DEE0','#ABC2C5','#7D9FA6'])
MUTED_MAE=LinearSegmentedColormap.from_list('muted_mae',['#90B5BC','#C7D6D8','#E7E8E2','#EFE2CC'])
GRID=plt.get_cmap('viridis')
# Normalized figure-coordinate regions; all images include axes labels, never headings.
PANELS={'a':[0,.790,1,.210], 'b':[0,.525,.605,.260], 'c':[0,0,.605,.520],
        'd':[.607,.458,.393,.327], 'e':[.607,.290,.393,.165], 'f':[.607,0,.393,.287]}
HEADINGS={'a':'Direction detection','b':'Persistent-direction discrimination','c':'Native output MAE',
          'd':'Component ablations','e':'Parameter sensitivity','f':'Transfer to test signals'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.8,'axes.labelsize':6.5,
        'xtick.labelsize':5.7,'ytick.labelsize':5.8,'axes.linewidth':.65,
        'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2,'ytick.major.size':2,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':INK,
        'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
        'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})


def export_figure(fig, path, **kwargs):
    fig.savefig(Path(path), **kwargs)


def interval(x,draws=5000,seed=20261004):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if len(x)==0: raise ValueError('No finite repeat averages')
    means=x[np.random.default_rng(seed).integers(0,len(x),(draws,len(x)))].mean(1)
    return float(x.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))

def limits(values,zero=False,lower=None,upper=None,pad=.10,minspan=.01):
    v=np.asarray(values,float).ravel();v=v[np.isfinite(v)];lo,hi=v.min(),v.max()
    if zero:lo,hi=min(0,lo),max(0,hi)
    span=max(hi-lo,minspan);lo-=span*pad;hi+=span*pad
    if lower is not None:lo=max(lo,lower)
    if upper is not None:hi=min(hi,upper)
    return float(lo),float(hi)

def native(t):
    q=t.copy()
    for col,value in [('window','physical'),('track','native'),('scenario','main')]:
        if col in q:q=q[q[col]==value]
    return q

def cloud(ax,values,y,color,summary=None,draws=5000,pointsize=3.8,seed=20261004,meansize=2.7):
    """Half-density, interquartile bar, raw points and repeat-bootstrap mean interval."""
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    if len(v)>1 and np.ptp(v)>1e-10:
        x=np.linspace(v.min(),v.max(),160);d=gaussian_kde(v)(x);d=d/d.max()*.29
        ax.fill_between(x,y-.12-d,y-.12,color=color,alpha=.32,edgecolor=color,lw=.4)
    q=np.quantile(v,[.25,.5,.75]);ax.plot(q[[0,2]],[y,y],color=color,lw=2.2,solid_capstyle='round')
    ax.plot(q[1],y,'|',color='white',ms=3.2,mew=.8)
    j=np.random.default_rng(20261005).uniform(.08,.26,len(v))
    ax.scatter(v,y+j,s=pointsize,color=color,alpha=.45,linewidths=0,zorder=3)
    if summary is not None:
        if len(summary)==3 and isinstance(summary,tuple):m,l,h=summary
        else:m,l,h=interval(summary,draws,seed)
        mean_style={'markeredgewidth':.25,'barsabove':True} if meansize<2 else {}
        ax.errorbar(m,y-.07,xerr=[[m-l],[h-m]],fmt='D',color=INK,ecolor=INK,
                    ms=meansize,lw=.95,capsize=1.8,capthick=.8,zorder=7,**mean_style)

def put_title(fig,titles,text,x,y,size=6.8,weight='normal',ha='left',va='top',panel=None):
    a=fig.text(x,y,text,fontsize=size,fontweight=weight,ha=ha,va=va)
    titles.append((a,dict(text=text,x=x,y=y,fontsize=size,fontweight=weight,ha=ha,va=va,panel=panel)))
    return a

def table(ax,values,cmap,vmin,vmax,fontsize=5.0):
    nrows,ncols=values.shape
    im=ax.pcolormesh(np.arange(ncols+1)-.5,np.arange(nrows+1)-.5,values.to_numpy(),cmap=cmap,vmin=vmin,vmax=vmax,shading='flat',edgecolors='none',antialiased=False,rasterized=False)
    ax.set_xlim(-.5,ncols-.5);ax.set_ylim(nrows-.5,-.5)
    for i,j in np.ndindex(values.shape):
        ax.text(j,i,f'{values.iloc[i,j]:.3f}',ha='center',va='center',fontsize=fontsize,color=INK)
    ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
    return im

def trace_rows(rows,panel,metric,frame,valuecol):
    for _,r in frame.iterrows():
        item={k:r[k] for k in ['field','family','relation','repeat','method','configuration'] if k in r.index}
        rows.append(dict(panel=panel,metric=metric,value=float(r[valuecol]),**item))

def draw(data:Path,out:Path,dpi=600):
    style();out.mkdir(parents=True,exist_ok=True)
    cell=native(pd.read_csv(data/'cell_metrics.csv'));block=native(pd.read_csv(data/'block_metrics.csv'))
    grid=pd.read_csv(data/'development_grid.csv');world=pd.read_csv(data/'native_world_MAE.csv')
    # Store one long row per world/method for complete traceability.
    if 'method' in world and 'mae' in world:world=world.pivot(index=['field','family','relation','repeat'],columns='method',values='mae').reset_index()
    if len(world)!=300 or world[STRENGTH].isna().any().any():raise ValueError('Expected 300 complete nine-method strength worlds')
    titles=[];rows=[];fig=plt.figure(figsize=(7.15,8.65))
    for k,(x,y,w,h) in PANELS.items():
        put_title(fig,titles,k,x+.008,y+h-.003,11,'bold',panel=k)
        put_title(fig,titles,HEADINGS[k],x+.038,y+h-.005,7.8,'bold',panel=k)
    # a: condition-wise direction detection, with the original five directional methods.
    for rect,metric,rels,yl in [([.070,.898,.916,.053],'current_correct_detection',RELATIONS[1:5],'Correct detection'),
            ([.070,.815,.916,.053],'null_false_alarm',['sync','sporadic','independent'],'False alarm')]:
        ax=fig.add_axes(rect);t=cell[cell.method.isin(DIRECT)]
        vals=t.groupby(['field','family','relation','method'])[metric].mean();allv=[]
        for ri,r in enumerate(rels):
            off=ri*7
            if ri:ax.axvline(off-.85,color='#DBDFE1',lw=.55)
            name=REL[r]+((' (present)' if metric=='current_correct_detection' else ' (absent)') if r=='sporadic' else '')
            xx=rect[0]+rect[2]*(off+3)/((len(rels)-1)*7+6)
            put_title(fig,titles,name,xx,rect[1]+rect[3]+.008,6.6,ha='center',panel='a')
            for method in DIRECT:
                c,m,ls=STYLES[method];v=[vals.loc[(f,g,r,method)] for f,g in PROFILES];allv.extend(v)
                ax.plot(np.arange(6)+off,v,color=c,marker=m,ls=ls,lw=1.05 if method=='RGIC' else .8,ms=2.35)
        ax.set_xticks([i*7+j for i in range(len(rels)) for j in range(6)],CODES*len(rels),fontsize=5.45)
        ax.set_xlim(-.5,(len(rels)-1)*7+5.5);ax.set_ylim(limits(allv,zero=True,lower=0,minspan=.05))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3));ax.set_ylabel(yl,fontsize=6.25)
        ax.grid(axis='y',color='#E7EAEC',lw=.4);ax.set_axisbelow(True)
        trace_rows(rows,'a',metric,t[t.relation.isin(rels)].dropna(subset=[metric]),metric)
    handles=[Line2D([0],[0],color=STYLES[m][0],marker=STYLES[m][1],ls=STYLES[m][2],lw=.9,ms=3,label=LABEL.get(m,m)) for m in DIRECT]
    fig.legend(handles=handles,ncol=5,loc='upper right',bbox_to_anchor=(.994,.995),frameon=False,fontsize=6.0,columnspacing=.8,handlelength=1.8,handletextpad=.4,borderpad=0)
    # b: the original 60 native AP blocks and a muted condition-mean table.
    t=block[block.method.isin(DIRECT)];ax=fig.add_axes([.118,.664,.430,.080])
    for i,m in enumerate(DIRECT):
        q=t[t.method==m];cloud(ax,q.persistent_ap,i,COLORS[m],q.groupby('repeat').persistent_ap.mean(),meansize=1.2)
    ax.set_yticks(range(5),[LABEL.get(m,m) for m in DIRECT],fontsize=6);ax.set_ylim(4.48,-.5)
    ax.set_xlim(limits(t.persistent_ap,zero=True,lower=0,upper=1,minspan=.1));ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.set_xlabel('AP',labelpad=2)
    put_title(fig,titles,'Overall AP',.333,.757,6.8,ha='center',panel='b')
    v=t.groupby(['method','field','family']).persistent_ap.mean().unstack(['field','family']).reindex(index=DIRECT,columns=pd.MultiIndex.from_tuples(PROFILES))
    ax=fig.add_axes([.118,.555,.430,.068]);im=table(ax,v,MUTED_AP,.10,.40,5.25)
    ax.set_xticks(range(6),CODES);ax.set_yticks(range(5),[LABEL.get(m,m) for m in DIRECT],fontsize=6)
    cb=fig.colorbar(im,cax=fig.add_axes([.564,.555,.009,.068]),ticks=[.1,.25,.4]);cb.ax.tick_params(labelsize=5.4,pad=1)
    put_title(fig,titles,'AP',.568,.638,6,ha='center',panel='b');trace_rows(rows,'b','persistent_ap',t,'persistent_ap')
    # c: 300-world rainclouds, with intervals over ten repeat means, then all 30 condition means.
    ax=fig.add_axes([.158,.378,.390,.113]);c_cloud_ax=ax
    for i,m in enumerate(STRENGTH):cloud(ax,world[m],i,COLORS[m],world.groupby('repeat')[m].mean(),draws=10000,pointsize=2.6,seed=20261022,meansize=1.2)
    ax.set_yticks(range(9),[LABEL.get(m,m) for m in STRENGTH],fontsize=5.9);ax.set_ylim(8.48,-.5)
    ax.set_xlim(limits(world[STRENGTH],zero=True,lower=0,minspan=.1));ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.set_xlabel('MAE (κ) ↓',labelpad=1.5,loc='right')
    order=pd.MultiIndex.from_tuples([(f,g,r) for f,g in PROFILES for r in RELATIONS[:-1]],names=['field','family','relation'])
    cond=world.groupby(['field','family','relation'])[STRENGTH].mean().reindex(order)
    ax=fig.add_axes([.146,.064,.391,.270]);c_table_ax=ax;im=table(ax,cond,MUTED_MAE,0,.70,4.95)
    short=['Sync','Stable lag','Changing lag','Changing sign','Sporadic']
    ax.set_yticks(range(30),[f'{code} · {r}' for code in CODES for r in short],fontsize=5.45)
    ax.set_xticks(range(9),[LABEL.get(m,m) if m!='PhiID_UniqueTransfer' else 'ΦID UT' for m in STRENGTH],rotation=50,ha='right',fontsize=5.7)
    for j in range(5,30,5):ax.axhline(j-.5,color='white',lw=.6)
    cb=fig.colorbar(im,cax=fig.add_axes([.550,.064,.009,.270]),ticks=[0,.35,.7]);cb.ax.tick_params(labelsize=5.4,pad=1);cb.ax.set_ylabel('')
    c_table_heading=put_title(fig,titles,'Condition mean MAE',.146,.347,6.3,panel='c')
    trace_rows(rows,'c','mae',world.melt(id_vars=[x for x in ['cell','field','family','relation','repeat'] if x in world],value_vars=STRENGTH,var_name='method',value_name='mae'),'mae')
    # d: six true ABC deletion endpoints; positive means performance worsens on removal.
    d=pd.read_csv(data/'ablation_repeat_differences.csv')
    if 'component' not in d:d['component']=d.method.str.replace('No ','',regex=False).str.replace(' matched','',regex=False).str.capitalize()
    endpoints=['a_detection','b_native_ap','mae'];yl=[.650,.562,.474];dh=.055;dx=[.715,.872];dw=.113
    for ci,setting in enumerate(['Original','Shared interference']):
        put_title(fig,titles,setting,dx[ci]+dw/2,.738,6.0,ha='center',panel='d')
    put_title(fig,titles,'Removed',.629,.738,6.0,panel='d')
    for ri,(ep,name) in enumerate(zip(endpoints,['a · Δ detection (pp)','b · Δ native AP','c · ΔMAE'])):
        td=d[d.endpoint==ep];scale=100 if ep=='a_detection' else 1;lim=limits(td.difference*scale,zero=True,pad=.15)
        put_title(fig,titles,name,.715,yl[ri]+dh+.016,6.0,'bold',panel='d')
        for ci,setting in enumerate(['Original','Shared interference']):
            ax=fig.add_axes([dx[ci],yl[ri],dw,dh]);ax.axvline(0,color=INK,lw=.7)
            for j,(component,color,marker) in enumerate(zip(COMPONENTS,COMP_COLORS,COMP_MARKERS)):
                q=td[(td.setting==setting)&(td.component==component)].sort_values('repeat');v=q.difference.to_numpy()*scale
                if len(v)!=10:raise ValueError(f'Expected ten d repeats: {setting}/{ep}/{component}, got {len(v)}')
                jitter=np.linspace(-.16,.16,len(v));ax.scatter(v,j+jitter,color=color,marker=marker,s=4.5,alpha=.65,linewidths=0)
                m,l,h=interval(v,10000 if setting!='Original' or ep=='mae' else 5000,20261022 if ep=='mae' else 20261004)
                ax.errorbar(m,j,xerr=[[m-l],[h-m]],fmt=marker,color=color,ecolor=INK,ms=3.5,lw=.8,capsize=1.7,zorder=5)
            ax.set_xlim(lim);ax.set_ylim(3.5,-.5);ax.set_yticks(range(4),COMPONENTS if ci==0 else ['']*4,fontsize=5.6)
            ax.tick_params(axis='y',length=0,pad=1.5);ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.tick_params(axis='x',labelsize=5.3,pad=1.5)
            if ep=='a_detection':ax.set_xticks([0,5,10])
    for _,r in d.iterrows():rows.append(dict(panel='d',metric=r.endpoint,method=r.get('method'),repeat=int(r['repeat']),setting=r.setting,value=float(r.difference)))
    # e: expanded development grids, with the same frozen operating points.
    fmin=np.floor(grid.development_ap.min()*20)/20;fmax=np.ceil(grid.development_ap.max()*20)/20
    for rect,w,title,lam,gamma in [([.670,.348,.125,.062],'physical','Physical window',2,.45),([.850,.348,.125,.062],'W2','W2',2.25,.55)]:
        ax=fig.add_axes(rect);q=grid[(grid.window==w)&(grid.method=='RGIC')].pivot(index='gamma',columns='lam',values='development_ap')
        im=ax.pcolormesh(np.arange(q.shape[1]+1)-.5,np.arange(q.shape[0]+1)-.5,q.to_numpy(),cmap=GRID,vmin=fmin,vmax=fmax,shading='flat',edgecolors='none',antialiased=False,rasterized=False)
        ax.set_xlim(-.5,q.shape[1]-.5);ax.set_ylim(-.5,q.shape[0]-.5)
        xt=[x for x in [0,1,lam,4,12] if x in q.columns];yt=[x for x in [0,.2,gamma,.8,2] if x in q.index]
        ax.set_xticks([list(q.columns).index(x) for x in xt],[f'{x:g}' for x in xt],fontsize=5.5);ax.set_yticks([list(q.index).index(x) for x in yt],[f'{x:g}' for x in yt],fontsize=5.5)
        ax.add_patch(Rectangle((list(q.columns).index(lam)-.5,list(q.index).index(gamma)-.5),1,1,fill=False,edgecolor=GOLD,lw=1.4))
        ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False);ax.set_ylabel('γ (s⁻¹)',fontsize=6.1,labelpad=1)
        put_title(fig,titles,title,rect[0]+rect[2]/2,.423,6.5,ha='center',panel='e')
    put_title(fig,titles,'□ Fixed',.986,.448,5.8,ha='right',panel='e')
    put_title(fig,titles,'λ (s⁻¹)',.833,.328,6.1,ha='center',panel='e')
    cb=fig.colorbar(im,cax=fig.add_axes([.736,.302,.220,.008]),orientation='horizontal',ticks=[fmin,(fmin+fmax)/2,fmax]);cb.ax.tick_params(labelsize=5.4,length=1.3,pad=1)
    put_title(fig,titles,'AP',.711,.306,5.8,ha='right',va='center',panel='e')
    for _,r in grid.iterrows():rows.append(dict(panel='e',metric='development_ap',window=r.window,lam=float(r.lam),gamma=float(r.gamma),value=float(r.development_ap)))
    # f: three matched task comparisons. Preserve the original scatter + paired raincloud style.
    det=pd.read_csv(data/'parameter_detection_world_metrics.csv')
    det.configuration=det.configuration.replace({'Default':'Sample-unit defaults','Physical-time':'Fixed physical-time'})
    det=det[det.relation.isin(RELATIONS[1:5])]
    dv=det.groupby(['field','family','repeat','configuration']).current_correct_detection.mean().unstack('configuration')
    bv=block[block.method.isin(['RGIC original','RGIC'])].pivot(index=['field','family','repeat'],columns='method',values='persistent_ap').rename(columns={'RGIC original':'Sample-unit defaults','RGIC':'Fixed physical-time'})
    sm=pd.read_csv(data/'parameter_strength_world_MAE.csv')
    sv=sm.groupby(['field','family','repeat','configuration']).mae.mean().unstack('configuration')
    fmetrics=[('Direction detection',dv,100,'Δ detection (pp)',False),('Direction AP',bv,1,'ΔAP',False),('Strength MAE',sv,1,'ΔMAE',True)]
    fx=[.646,.773,.900];fw=.089
    for fi,(name,v,scale,delta_label,lowbetter) in enumerate(fmetrics):
        x=fx[fi];v=v[['Sample-unit defaults','Fixed physical-time']].copy()*scale
        if v.isna().any().any():raise ValueError('Incomplete f metric '+name)
        ax=fig.add_axes([x,.148,fw,.100])
        lo,hi=limits(v.to_numpy(),lower=0,pad=.10,minspan=.02 if scale==1 else 1)
        ax.plot([lo,hi],[lo,hi],color=INK,ls=':',lw=.7)
        for field,color,marker in [('3T',BLUE,'o'),('7T',GOLD,'s')]:
            q=v.xs(field);ax.scatter(q['Sample-unit defaults'],q['Fixed physical-time'],s=6.5 if len(q)<50 else 3.8,color=color,marker=marker,alpha=.65,linewidths=.15,edgecolors='white')
        ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal',adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2));ax.yaxis.set_major_locator(MaxNLocator(nbins=2));ax.tick_params(labelsize=5.0,pad=1)
        ax.set_xlabel('Default',fontsize=5.8,labelpad=1);ax.set_ylabel('Fixed',fontsize=5.8,labelpad=1)
        put_title(fig,titles,'Detection (%)' if name=='Direction detection' else name,x+fw/2,.257,5.9,ha='center',panel='f')
        a=fig.add_axes([x,.048,fw,.063]);dvalues=[]
        for j,(field,color) in enumerate([('3T',BLUE),('7T',GOLD)]):
            q=v.xs(field);delta=q['Fixed physical-time']-q['Sample-unit defaults'];dvalues.extend(delta)
            cloud(a,delta,j,color,delta.groupby('repeat').mean(),draws=10000 if lowbetter else 5000,pointsize=3.0,seed=20261022 if lowbetter else 20261004)
        a.set_xlim(limits(dvalues,zero=True,pad=.12));a.set_ylim(1.45,-.5);a.axvline(0,color=INK,ls=':',lw=.65)
        a.set_yticks([0,1],['3T','7T'],fontsize=5.5);a.xaxis.set_major_locator(MaxNLocator(nbins=2));a.tick_params(labelsize=5.0,pad=1)
        a.spines['left'].set_visible(False);a.set_xlabel(delta_label,fontsize=5.7,labelpad=1)
        for key,r in v.reset_index().iterrows():
            rows.append(dict(panel='f',metric=name,field=r['field'],family=r['family'],repeat=int(r['repeat']),
                             relation=r.get('relation',None),default=float(r['Sample-unit defaults']),fixed=float(r['Fixed physical-time']),difference=float(r['Fixed physical-time']-r['Sample-unit defaults'])))
    fig.legend(handles=[Line2D([0],[0],color=BLUE,marker='o',ls='none',ms=3,label='3T'),Line2D([0],[0],color=GOLD,marker='s',ls='none',ms=3,label='7T')],loc='upper right',bbox_to_anchor=(.994,.288),ncol=2,frameon=False,fontsize=5.8,handletextpad=.2,columnspacing=.5,borderpad=0)
    for ax in fig.axes:
        for collection in ax.collections:
            collection.set_rasterized(False)
    for extension in ('png', 'pdf', 'svg'):
        export_figure(fig, out / f'Figure1_updated.{extension}', dpi=dpi)
    pd.DataFrame(rows).to_csv(out / 'plotted_chart_data.csv', index=False)
    plt.close(fig)
    return out / 'Figure1_updated.png'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dpi',type=int,default=600);a=p.parse_args();print(draw(a.results,a.output,a.dpi))
