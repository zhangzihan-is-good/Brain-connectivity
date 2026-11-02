"""Only the 35 AUROC panels selected by the user. No training or bootstrap.

Supply a newly generated selected-run summary with --input. Rendering also
requires --output; the code package contains no existing results or figures.
"""
from __future__ import annotations
import argparse,json,hashlib,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
METHODS=('RGIC','Pearson','GCMI','Lag_peak','Gaussian_TE','VAR_Granger','PSI','PhiID_unique_transfer','Coherence')
MLABEL=dict(zip(METHODS,('RGIC','Pearson','GCMI','Lag peak','Gaussian TE','VAR Granger','PSI','PhiID transfer','Coherence')))
CLASSES=('fcn','logistic');CLABEL={'fcn':'FCN','logistic':'逻辑回归 LR'}
COLOR={'fixed':'#bd7137','rescreen':'#286581'}
BASE=[('signed_joint','RGIC','完整 IC+D'),('IC_only','RGIC','仅 IC 通道'),('D_only','RGIC','仅 D 通道'),('abs_fixed','RGIC','D取绝对值 / 固定选列'),('abs_rescreen','RGIC','D取绝对值 / 重新筛选'),('signed_joint','Pearson','Pearson'),('signed_joint','GCMI','GCMI'),('signed_joint','PSI','PSI')]
VARIANTS=[('no_cost','移除代价 C'),('no_reliability','移除可靠性 R'),('no_compensation','移除幅度补偿 M'),('no_cost__no_reliability','移除 C+R'),('no_cost__no_compensation','移除 C+M'),('no_reliability__no_compensation','移除 R+M'),('no_cost__no_reliability__no_compensation','移除 C+R+M')]
NOTE='全标签F预筛（训练+验证+测试）；AAL116 / W1；AUROC为原十折等权均值。\n区间为已拟合OOF的条件配对95%区间，不含重新选择/训练不确定性，未作多重比较校正。'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def K(f,c,cl,m='RGIC'):return(f,c,cl,m)
class Data:
 def __init__(self,path):
  j=json.loads(path.read_text(encoding='utf-8-sig'));assert j['status']=='complete_selected_AUROC_only'
  self.rows={tuple(r[x] for x in ('family','condition','classifier','method')):r for r in j['groups']}
  assert len(self.rows)==572 and len(j['groups'])==572 and j['selected_fold_records']==5720
  self.pairs={(tuple(r['reference']),tuple(r['target'])):r for r in j['comparisons']}
  assert len(self.pairs)==len(j['comparisons']) and len(self.pairs) in (460,578)
  expected={K('direction',c,cl,m) for c,m,_ in BASE for cl in CLASSES}
  expected|={K('components',c,cl) for cl in CLASSES for c in ['full']+[v+'_'+mode for v,_ in VARIANTS for mode in ('fixed','rescreen')]}
  expected|={K('population',pop+'_'+condition,cl,m) for pop in ('all','lowFD') for cl in CLASSES for condition in ('brain_only','brain_plus_demo') for m in METHODS}
  expected|={K('population',pop+'_demo_only',cl,'demographics') for pop in ('all','lowFD') for cl in CLASSES}
  raw_conditions=['prefix_100']+[f'prefix_{q:03d}__{mode}' for q in (25,50,75) for mode in ('fixed','rescreen')]+[f'noise_{q:03d}_seed{seed}__{mode}' for q in (10,20,30) for seed in (101,202,303) for mode in ('fixed','rescreen')]
  expected|={K('raw_robustness',c,cl,m) for c in raw_conditions for cl in CLASSES for m in METHODS}
  assert set(self.rows)==expected
  required={(K('components','full',cl),K('components',v+'_'+mode,cl)) for cl in CLASSES for v,_ in VARIANTS for mode in ('fixed','rescreen')}
  required|={(K('raw_robustness','prefix_100',cl,m),K('raw_robustness',c,cl,m)) for cl in CLASSES for m in METHODS for c in raw_conditions if c!='prefix_100'}
  assert len(required)==460 and required<=set(self.pairs)
  for (r,t),p in self.pairs.items():
   assert r in self.rows and t in self.rows and p['orientation']=='target_minus_reference'
   assert abs(p['delta_AUROC']-(self.value(t)-self.value(r)))<1e-10
   b=p['conditional_CI95_AUROC'];assert len(b)==2 and all(math.isfinite(x) for x in b) and b[0]<=b[1]
  for key,r in self.rows.items():
   assert r['complete_folds']==10 and math.isfinite(r['mean_AUROC']) and 0<=r['mean_AUROC']<=1
   assert r['atlas']=='aal' and r['window']==1 and r['seed']==42
   expected_n=884 if key[0]=='population' and key[1].startswith('lowFD_') else 1035
   assert r['n_test']==expected_n
   assert r.get('status','complete')=='complete' and r.get('n_defined_folds_AUROC',10)==10
  for key in self.rows:
   f,c,cl,m=key
   if f=='direction':assert (c,m) in {(x,y) for x,y,_ in BASE}
   elif f=='components':assert c=='full' or c in {v+'_'+mode for v,_ in VARIANTS for mode in ('fixed','rescreen')}
   else:assert f in ('population','raw_robustness')
 def value(self,k):return self.rows[k]['mean_AUROC']
 def delta(self,t,r):
  if t==r:return 0.,(0.,0.)
  if (r,t) in self.pairs:p=self.pairs[r,t];v=p['delta_AUROC'];b=tuple(p['conditional_CI95_AUROC'])
  else:p=self.pairs[t,r];v=-p['delta_AUROC'];lo,hi=p['conditional_CI95_AUROC'];b=(-hi,-lo)
  assert abs(v-(self.value(t)-self.value(r)))<1e-10
  return v,b
def init_plot():
 global plt,PdfPages
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 from matplotlib import font_manager
 font=Path('C:/Windows/Fonts/msyh.ttc')
 if font.exists():font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
 else:family=['Noto Sans CJK SC','DejaVu Sans']
 plt.rcParams.update({'font.family':family,'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'axes.linewidth':.7,'grid.color':'#dce1e5','grid.linewidth':.55,'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
def page(nr,nc,title,note,size):
 f,a=plt.subplots(nr,nc,figsize=size,squeeze=False)
 f.subplots_adjust(left=.11,right=.97,top=.81,bottom=.22,wspace=.38,hspace=.63)
 f.suptitle(title,x=.05,y=.975,ha='left',va='top',fontsize=18,fontweight='bold')
 f.text(.05,.12,NOTE,fontsize=9,va='top',linespacing=1.5,color='#45515a')
 f.text(.05,.055,note,fontsize=9,va='top',linespacing=1.4,color='#45515a')
 return f,a
def direction(d,marks):
 f,a=page(1,2,'补充实验1 · 方向表示与AUROC','仅保留指定第一页的两张AUROC图。虚线为完整RGIC；本页是点估计，不额外生成绝对成绩置信区间。',(17,6.7))
 for j,cl in enumerate(CLASSES):
  ax=a[0,j];ax.axvline(d.value(K('direction','signed_joint',cl)),color='#68747c',ls='--',lw=1)
  for y,(c,m,label) in enumerate(BASE):
   k=K('direction',c,cl,m);v=d.value(k);color='#286581' if y==0 else ('#bd7137' if y<5 else '#59666e')
   ax.plot(v,y,'o',color=color,ms=7 if y==0 else 6);ax.text(v+.014,y,f'{v:.4f}',va='center',color=color)
   marks.append(dict(kind='value',key=k,value=v,figure='S1',panel=cl))
  ax.set_yticks(range(8),[x[2] for x in BASE]);ax.set_ylim(7.6,-.6);ax.set_xlim(.45,1.02);ax.grid(axis='x');ax.set_xlabel('AUROC (0-1)');ax.set_title(CLABEL[cl],loc='left',fontweight='bold')
 return [f]
def components(d,marks):
 f,a=page(1,2,'补充实验3 · 三组件全组合消融与AUROC','橙圆=固定选列；蓝方=重新筛选。差值=消融−完整RGIC；横线为已有条件95%区间，0表示完整参考。',(18,7))
 endpoints=[0.]
 for cl in CLASSES:
  for v,_ in VARIANTS:
   for mode in ('fixed','rescreen'):x,b=d.delta(K('components',v+'_'+mode,cl),K('components','full',cl));endpoints.extend([x,*b])
 lo,hi=min(endpoints),max(endpoints);pad=(hi-lo)*.1
 for j,cl in enumerate(CLASSES):
  ax=a[0,j];ref=K('components','full',cl)
  for y,(v,label) in enumerate(VARIANTS):
   for mode,dy,mk in [('fixed',-.13,'o'),('rescreen',.13,'s')]:
    k=K('components',v+'_'+mode,cl);x,b=d.delta(k,ref);ax.plot(b,[y+dy,y+dy],color=COLOR[mode],lw=1.6);ax.plot(x,y+dy,mk,color=COLOR[mode],ms=5.5)
    marks.append(dict(kind='delta',key=k,reference=ref,value=x,bounds=b,figure='S3',panel=cl))
  ax.axvline(0,color='#4f5b62',lw=1);ax.set_yticks(range(7),[x[1] for x in VARIANTS]);ax.set_ylim(6.5,-.6);ax.set_xlim(lo-pad,hi+pad);ax.grid(axis='x');ax.set_xlabel('AUROC差（消融−完整）');ax.set_title(CLABEL[cl]+f' · 完整参考 {d.value(ref):.4f}',loc='left',fontweight='bold')
 return[f]
def population(d,marks):
 f,a=page(2,2,'补充实验4 · 人口学追加前后的AUROC','蓝圆=仅连接；橙方=连接＋性别/年龄/利手/IQ；灰虚线=仅人口学。人口学拼接到分类器，不进入RGIC公式，不等于去混杂。\n全样本和低头动是不同样本构成；横轴局部放大。',(16,12))
 for i,(pop,label,n) in enumerate([('all','全样本',1035),('lowFD','低头动 FD≤0.2mm',884)]):
  for j,cl in enumerate(CLASSES):
   ax=a[i,j]
   for y,m in enumerate(METHODS):
    kb=K('population',pop+'_brain_only',cl,m);kp=K('population',pop+'_brain_plus_demo',cl,m);vb,vp=d.value(kb),d.value(kp)
    ax.plot([vb,vp],[y-.06,y+.06],color='#b8c0c6',lw=1.3);ax.plot(vb,y-.06,'o',color='#286581',ms=6);ax.plot(vp,y+.06,'s',color='#bd7137',ms=6)
    high=max(vb,vp)>.9;x=min(vb,vp)-.025 if high else max(vb,vp)+.025;align='right' if high else 'left'
    ax.text(x,y-.22,f'{vb:.4f}',va='center',ha=align,color='#286581',fontsize=9.5);ax.text(x,y+.22,f'{vp:.4f}',va='center',ha=align,color='#bd7137',fontsize=9.5)
    marks.extend([dict(kind='value',key=kb,value=vb,figure='S4',panel=f'{pop}/{cl}'),dict(kind='value',key=kp,value=vp,figure='S4',panel=f'{pop}/{cl}')])
   kd=K('population',pop+'_demo_only',cl,'demographics');vd=d.value(kd);ax.axvline(vd,color='#69747c',ls='--',lw=1);ax.text(vd,-.65,f'仅人口学 {vd:.4f}',ha='center',fontsize=9,color='#69747c');marks.append(dict(kind='value',key=kd,value=vd,figure='S4',panel=f'{pop}/{cl}'))
   ax.set_yticks(range(9),[MLABEL[m] for m in METHODS]);ax.set_ylim(8.5,-1.05);ax.set_xlim(.5,1.01);ax.grid(axis='x');ax.set_xlabel('AUROC（0-1，局部放大）');ax.set_title(f'{label} N={n} · {CLABEL[cl]}',loc='left',fontweight='bold')
 return[f]
def raw_point(d,c,cl,m,marks,panel):
 k=K('raw_robustness',c,cl,m);ref=K('raw_robustness','prefix_100',cl,m);v,b=d.delta(k,ref);marks.append(dict(kind='delta',key=k,reference=ref,value=v,bounds=b,figure='S5',panel=panel));return v,b
def curve(ax,xs,points,color,marker,ls,alpha=1,lw=1.8):
 for x,(v,b) in zip(xs,points):ax.plot([x,x],b,color=color,alpha=.35,lw=.9)
 ax.plot(xs,[x[0] for x in points],color=color,marker=marker,linestyle=ls,alpha=alpha,lw=lw,ms=4)
def raw(d,marks):
 figs=[];modes=('fixed','rescreen');seeds=(101,202,303)
 # One common AUROC-difference domain per page, including every saved endpoint.
 def limits(p):
  vals=[z for points in p.values() for value,b in points for z in (value,*b)]+[0.];lo,hi=min(vals),max(vals);pad=max((hi-lo)*.075,.008);return(lo-pad,hi+pad)
 f,a=page(3,3,'补充实验5 · 时序长度与AUROC：相对自身100%输入','橙色=固定选列；蓝色=重新筛选。实线圆点=FCN；虚线方点=LR。每点独立条件95%区间。\nN=1035；截取已全长预处理的ROI前缀，各条件重新拟合分类器；不代表独立短扫描预处理。',(18,15))
 prepared={}
 for m in METHODS:
  for cl in CLASSES:
   for mode in modes:prepared[m,cl,mode]=[raw_point(d,'prefix_100' if q==100 else f'prefix_{q:03d}__{mode}',cl,m,marks,'length') for q in (25,50,75,100)]
 lim=limits(prepared)
 for ax,m in zip(a.flat,METHODS):
  for cl in CLASSES:
   for mode in modes:curve(ax,(25,50,75,100),prepared[m,cl,mode],COLOR[mode],'o' if cl=='fcn' else 's','-' if cl=='fcn' else '--')
  ax.axhline(0,color='#69747c',lw=.8);ax.set_ylim(*lim);ax.set_xticks([25,50,75,100],['25%','50%','75%','100%']);ax.set_xlabel('保留的原时序长度');ax.set_ylabel('AUROC差');ax.grid(axis='y');ax.set_title(MLABEL[m]+f'\n100%: FCN {d.value(K("raw_robustness","prefix_100","fcn",m)):.4f} / LR {d.value(K("raw_robustness","prefix_100","logistic",m)):.4f}',loc='left',fontsize=11)
 figs.append(f)
 for cl in CLASSES:
  f,a=page(3,3,'补充实验5 · 噪声与AUROC：'+CLABEL[cl]+'，相对自身原输入','橙色=固定选列；蓝色=重新筛选；101圆点/点线，202三角/虚线，303方点/点划线。粗线只显示三点均值，不平均区间。\nN=1035，同人三次扰动不是三倍样本或预测融合；σ按原ROI标准差计，扰动后重新拟合分类器。',(18,15))
  prepared={}
  for m in METHODS:
   for mode in modes:
    for seed in seeds:prepared[m,mode,seed]=[raw_point(d,'prefix_100' if q==0 else f'noise_{q:03d}_seed{seed}__{mode}',cl,m,marks,'noise/'+cl) for q in (0,10,20,30)]
  lim=limits(prepared)
  for ax,m in zip(a.flat,METHODS):
   for mode in modes:
    for seed,mk,ls in zip(seeds,('o','^','s'),(':','--','-.')):curve(ax,(0,.1,.2,.3),prepared[m,mode,seed],COLOR[mode],mk,ls,alpha=.6,lw=.9)
    means=[sum(prepared[m,mode,seed][i][0] for seed in seeds)/3 for i in range(4)]
    ax.plot((0,.1,.2,.3),means,color=COLOR[mode],lw=2.2)
   ax.axhline(0,color='#69747c',lw=.8);ax.set_ylim(*lim);ax.set_xticks([0,.1,.2,.3]);ax.set_xlabel('噪声强度 σ');ax.set_ylabel('AUROC差');ax.grid(axis='y');ax.set_title(MLABEL[m]+f'\n原输入 AUROC {d.value(K("raw_robustness","prefix_100",cl,m)):.4f}',loc='left',fontsize=11)
  figs.append(f)
 return figs
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-data',action='store_true');parser.add_argument('--input',type=Path,required=True,help='Newly generated selected-run summarize JSON');parser.add_argument('--output',type=Path);args=parser.parse_args()
 datafile=args.input.resolve()
 d=Data(datafile)
 if args.verify_data:print(json.dumps({'status':'PASS','groups':len(d.rows),'pairs':len(d.pairs),'fold_records':5720,'fits':0}));return
 if args.output is None:parser.error('Rendering requires --output')
 out=args.output.resolve()
 if out==ROOT or ROOT in out.parents or out in ROOT.parents:parser.error('Render output must be outside the code package')
 if out==datafile or out in datafile.parents:parser.error('Output cannot be the data file or its containing directory/ancestor')
 workspace=ROOT.parents[1]
 for name in ('第四部分','第三部分最终版','dataset'):
  protected=(workspace/name).resolve()
  if out==protected or protected in out.parents:parser.error('Output cannot write into original scientific inputs/results')
 init_plot();out.mkdir(parents=True,exist_ok=True);marks=[];files=[]
 def output_name(q):
  try:return str(q.relative_to(ROOT))
  except ValueError:return str(q)
 for name,func in [('S1_direction_AUROC',direction),('S3_components_AUROC',components),('S4_population_AUROC',population),('S5_length_noise_AUROC',raw)]:
  figs=func(d,marks);pdf=out/(name+'.pdf')
  with PdfPages(pdf,metadata={'Title':name,'Subject':'User-selected AUROC panels only; completed full-label F experiment'}) as book:
   for i,fig in enumerate(figs,1):
    book.savefig(fig,bbox_inches='tight',pad_inches=.16)
    for suffix in ('png','svg'):
     q=out/(name+f'_p{i:02d}.'+suffix);fig.savefig(q,dpi=600,bbox_inches='tight',pad_inches=.16);files.append({'path':output_name(q),'sha256':sha(q)})
    plt.close(fig)
  files.append({'path':output_name(pdf),'sha256':sha(pdf),'pages':len(figs)})
 auditdir=out
 (auditdir/'selected_figure_marks.json').write_text(json.dumps(marks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (auditdir/'selected_figure_receipt.json').write_text(json.dumps({'status':'rendered','data_sha256':sha(datafile),'files':files,'panels':35,'PDF_pages':6,'fits':0,'new_bootstrap':0},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('Selected35 AUROC panels exported in4 PDFs /6 pages')
if __name__=='__main__':main()
