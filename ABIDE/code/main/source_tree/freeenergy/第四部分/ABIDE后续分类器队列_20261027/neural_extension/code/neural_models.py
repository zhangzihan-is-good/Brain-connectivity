"""Validation-only neural fitting and separately authorized checkpoint inference.

Inputs are already train-transformed and beta-weighted by the caller. This
module never fits a scaler and never applies beta a second time. fit accepts no
test labels. Only GCN accepts unlabeled held-out features for its fixed graph;
fit never extracts held-out probabilities, decisions, or metrics.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
_dll_handles = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for _path in (r"D:\miniconda\Library\bin", r"D:\miniconda\DLLs"):
        if os.path.isdir(_path):
            _dll_handles.append(os.add_dll_directory(_path))
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from vendor_report_networks import FCN, LatentFeatureTransformer
from vendor_site_gcn import SiteGCN
from vendor_roi_profile import ROIProfileNetwork
from vendor_original_made import OriginalAE, OriginalMLP, train_original_ae

FORMAT = "validation-only-neural-checkpoint-v1"
KINDS = ("gcn", "fcn", "lstm", "transformer", "made")
DEFAULT_CONFIGS = json.loads((CODE / "neural_default_configs.json").read_text(encoding="utf-8"))
AXES = {
    "gcn": dict(hidden_dim=[32,64,128,256],dropout=[0.,.2,.3,.5],lr=[.0001,.0003,.001,.003],weight_decay=[0.,.00001,.0001,.0005,.001]),
    "fcn": dict(hidden1=[128,256,500,768],hidden2=[16,30,64],dropout=[.2,.5,.7],lr=[.0001,.0003,.0008,.002],weight_decay=[0.,.00001,.0001,.001]),
    "transformer": dict(d_model=[32,64,128],num_layers=[1,2],dropout=[0.,.2,.4],lr=[.0001,.0003,.001],weight_decay=[0.,.00001,.0001,.001]),
    "lstm": dict(hidden=[32,64],dropout=[0.,.1,.3,.5],lr=[.0001,.0003,.001,.003],weight_decay=[0.,.00001,.0001,.001]),
    "made": dict(lr=[.0001,.0005,.001],ae_lr=[.00003,.0001,.0003],hidden_size=[50,100,150]),
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_kind(kind):
    kind = str(kind).lower()
    return {"latentfeaturetransformer":"transformer", "latent_feature_transformer":"transformer",
            "sitegcn":"gcn", "roi_lstm":"lstm", "made_atlas":"made", "ssdae_mlp":"made"}.get(kind,kind)


def default_config(kind):
    kind = canonical_kind(kind)
    _need(kind in KINDS, "Unknown neural classifier")
    result = copy.deepcopy(DEFAULT_CONFIGS[kind])
    result.update(device="auto", threads=1)
    return result


def canonical_config(kind, config=None):
    kind = canonical_kind(kind)
    result = default_config(kind)
    given = dict(config or {})
    _need(not (set(given)-set(result)), "Unknown configuration keys: " + str(sorted(set(given)-set(result))))
    result.update(copy.deepcopy(given))
    int_keys = ("epochs","threads","batch_size","hidden_dim","cheb_k","hidden1","hidden2","d_model","nhead",
                "num_layers","dim_feedforward","num_latents","mlp_hidden","hidden","layers","heads","patience",
                "ae1_size","ae2_size","hidden_size","ae1_epochs","ae2_epochs","ae1_batch_size","ae2_batch_size",
                "saturate_momentum")
    for key in int_keys:
        if key in result:
            _need(not isinstance(result[key],bool) and int(result[key])==result[key] and result[key]>0,
                  "Positive integer required for " + key)
            result[key] = int(result[key])
    for key in ("lr","ae_lr"):
        if key in result:
            result[key] = float(result[key])
            _need(math.isfinite(result[key]) and result[key]>0, "Positive finite " + key + " required")
    for key in ("weight_decay","dropout","feature_dropout","clip_grad_norm","grad_clip","min_delta",
                "direction_weight","ae1_corruption","ae2_corruption","ae1_sparse_p","ae1_sparse_coefficient",
                "initial_momentum","final_momentum"):
        if key in result:
            result[key] = float(result[key])
            _need(math.isfinite(result[key]) and result[key]>=0, "Invalid " + key)
    for key in ("dropout","feature_dropout","ae1_corruption","ae2_corruption"):
        if key in result:
            _need(result[key]<1, key+" must be less than 1")
    _need(result["threads"]==1, "Only one numerical thread is authorized")
    _need(str(result["device"]) in ("auto","cpu","cuda","cuda:0"), "Unsupported device")
    if kind=="gcn":
        _need(result["cheb_k"]==2 and result["normalization"]=="symmetric_laplacian_lambda_max_2", "Original GCN requires K2 symmetric Laplacian")
    if kind=="transformer":
        _need(result["d_model"]%result["nhead"]==0, "d_model must be divisible by nhead")
    if kind=="lstm":
        _need(result["layers"]==1 and result["direction_weight"]==1., "LSTM has one layer and internal D gate1; beta is already applied upstream")
    if kind=="made":
        _need(result["hidden_size"]<=result["ae2_size"], "Original MLP hidden_size must not exceed AE2 width")
        _need(result["weight_decay"]==0, "Archive MADE uses plain SGD without weight decay")
        keep = [float(v) for v in result["dropout_keep_probabilities"]]
        _need(len(keep)==3 and all(0<v<=1 for v in keep), "Three original keep probabilities required")
        result["dropout_keep_probabilities"] = keep
        result["start_saving_after_zero_based_epoch"] = int(result["start_saving_after_zero_based_epoch"])
        _need(result["checkpoint_restore_policy"]=="correctly_restore_trained_encoders", "Trained encoder restoration is required")
    return result


def search_axes(kind):
    return copy.deepcopy(AXES[canonical_kind(kind)])


model_axes = search_axes


def model_presets(kind):
    """Eight fixed deterministic probes, not a full Cartesian product."""
    kind = canonical_kind(kind)
    base = default_config(kind)
    patches = {
        "gcn":[{},dict(lr=.0003),dict(lr=.003),dict(hidden_dim=64),dict(hidden_dim=256),dict(dropout=0.),dict(dropout=.5),dict(weight_decay=.0001)],
        "fcn":[{},dict(lr=.0003),dict(lr=.002),dict(hidden1=256),dict(hidden2=64),dict(dropout=.2),dict(dropout=.7),dict(weight_decay=.0001)],
        "transformer":[{},dict(lr=.0001),dict(lr=.001),dict(d_model=32,num_layers=1),dict(num_layers=1),dict(dropout=0.),dict(dropout=.4),dict(weight_decay=.001)],
        "lstm":[{},dict(lr=.0003),dict(lr=.003),dict(hidden=64),dict(dropout=0.),dict(dropout=.3),dict(weight_decay=0.),dict(weight_decay=.0001)],
        "made":[{},dict(lr=.0001),dict(lr=.001),dict(ae_lr=.00003),dict(ae_lr=.0003),dict(hidden_size=50),dict(hidden_size=150),dict(ae_lr=.0003,lr=.001)],
    }[kind]
    return [canonical_config(kind,dict(base,**patch)) for patch in patches]


def array_sha(values):
    values = np.ascontiguousarray(values)
    h=hashlib.sha256(str((values.shape,values.dtype.str)).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def _matrix(values, name, width=None):
    values = np.ascontiguousarray(values,dtype=np.float32)
    _need(values.ndim==2 and len(values)>0 and values.shape[1]>0 and np.isfinite(values).all(), name+" must be a finite nonempty 2D matrix")
    if width is not None:
        _need(values.shape[1]==width, name+" has a different width")
    return values


def _labels(values, n, name):
    raw=np.asarray(values)
    _need(raw.ndim==1 and len(raw)==n and np.isin(raw,[0,1]).all(), name+" must contain native ASD0/TC1 labels")
    return raw.astype(np.int64)


def _spec(spec,width,kind):
    result=copy.deepcopy(dict(spec or {}))
    nd=int(result.get("n_demographics",0))
    nc=int(result.get("n_connection_features",width-nd))
    _need(nd in (0,2,4) and nc>0 and nc+nd==width, "Expected K connection columns followed by 0/2/4 demo columns")
    result.update(n_demographics=nd,n_connection_features=nc)
    if kind=="lstm":
        r=int(result["n_rois"])
        edges=np.asarray(result["selected_edge_ids"],dtype=np.int64)
        channels=np.asarray(result["selected_channel_ids"],dtype=np.int64)
        _need(r>1 and edges.shape==channels.shape==(nc,), "LSTM needs one selected edge/channel per connection column")
        _need(np.all((edges>=0)&(edges<r*(r-1)//2)) and np.isin(channels,[0,1]).all(), "Invalid selected edge/channel")
        _need(len(set(zip(edges.tolist(),channels.tolist())))==nc, "Duplicate selected channel/edge columns")
        result.update(n_rois=r,selected_edge_ids=edges.tolist(),selected_channel_ids=channels.tolist())
    return result


def reconstruct_profiles(values,input_spec):
    """Only selected columns. D[row=source,col=target], positive low->high."""
    values=_matrix(values,"profile inputs")
    spec=_spec(input_spec,values.shape[1],"lstm")
    n,r=len(values),spec["n_rois"]
    edges=np.asarray(spec["selected_edge_ids"])
    channels=np.asarray(spec["selected_channel_ids"])
    hi,lo=np.tril_indices(r,-1)
    ic=np.zeros((n,r,r),dtype=np.float32)
    d=np.zeros_like(ic)
    scalar=channels==0
    directed=channels==1
    if scalar.any():
        ic[:,hi[edges[scalar]],lo[edges[scalar]]]=values[:,:spec["n_connection_features"]][:,scalar]
        ic[:,lo[edges[scalar]],hi[edges[scalar]]]=values[:,:spec["n_connection_features"]][:,scalar]
    if directed.any():
        v=values[:,:spec["n_connection_features"]][:,directed]
        d[:,lo[edges[directed]],hi[edges[directed]]]=v
        d[:,hi[edges[directed]],lo[edges[directed]]]=-v
    return dict(ic=ic,d=d,demographics=values[:,spec["n_connection_features"]:])


class ROIWithDemographics(nn.Module):
    def __init__(self,rois,n_demo,cfg,effective_direction_weight):
        super().__init__()
        self.network=ROIProfileNetwork("lstm",rois,dict(cfg,direction_weight=effective_direction_weight))
        self.demo_head=nn.Linear(n_demo,1,bias=False) if n_demo else None
    def forward(self,ic,d,demo):
        result=self.network(ic,d)
        return result if self.demo_head is None else result+self.demo_head(demo).squeeze(-1)


def _seed(seed,device):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if hasattr(torch.backends,"cuda"):
        torch.backends.cuda.matmul.allow_tf32=False
        for name,value in (("enable_flash_sdp",False),("enable_mem_efficient_sdp",False),("enable_math_sdp",True)):
            if hasattr(torch.backends.cuda,name): getattr(torch.backends.cuda,name)(value)
    torch.backends.cudnn.allow_tf32=False
    if device.type=="cpu": torch.backends.mkldnn.enabled=False


def _device(name):
    if name=="auto": name="cuda" if torch.cuda.is_available() else "cpu"
    result=torch.device(name)
    _need(result.type!="cuda" or torch.cuda.is_available(), "Requested CUDA runtime unavailable")
    return result


def _site_codes(roles,arrays,device):
    sites=[]
    for role,values in zip(roles,arrays):
        _need(role is not None and len(role)==len(values), "GCN needs role-aligned site IDs")
        sites.extend(str(value) for value in role)
    names,codes=np.unique(sites,return_inverse=True)
    tensor=torch.tensor(codes,dtype=torch.long,device=device)
    counts=np.bincount(codes)
    membership=torch.zeros((len(names),len(codes)),dtype=torch.float32,device=device)
    membership[torch.tensor(codes,device=device),torch.arange(len(codes),device=device)]=1.
    tensor._report_site_membership=membership
    tensor._report_site_counts=torch.tensor(counts,dtype=torch.float32,device=device)
    tensor._report_n_groups=len(names)
    return tensor,dict(names=names.tolist(),counts=counts.tolist(),codes_sha256=array_sha(codes),
                       directed_offdiagonal_edges=int(np.sum(counts*(counts-1))))


def _build_model(kind,width,spec,cfg,architecture,device):
    if kind=="fcn": model=FCN(width,cfg["hidden1"],cfg["hidden2"],cfg["dropout"])
    elif kind=="gcn": model=SiteGCN(width,cfg["hidden_dim"],cfg["dropout"])
    elif kind=="transformer":
        names=("d_model","nhead","num_layers","dim_feedforward","dropout","num_latents","mlp_hidden","feature_dropout")
        model=LatentFeatureTransformer(width,**{name:cfg[name] for name in names})
    elif kind=="lstm": model=ROIWithDemographics(spec["n_rois"],spec["n_demographics"],cfg,architecture["effective_direction_weight"])
    else:
        ae1=OriginalAE(spec["n_connection_features"],cfg["ae1_size"])
        ae2=OriginalAE(cfg["ae1_size"],cfg["ae2_size"])
        model=OriginalMLP(ae1,ae2,spec["n_demographics"],cfg["hidden_size"],cfg["dropout_keep_probabilities"])
    return model.to(device)


def _outputs(logits,kind):
    if kind=="lstm":
        tc=logits.reshape(-1)
        p_tc=torch.sigmoid(tc)
        probabilities=torch.stack((1.-p_tc,p_tc),dim=1)
        prediction=(tc>0).long()
        score=-tc
    else:
        probabilities=logits.exp() if kind=="gcn" else torch.softmax(logits,dim=1)
        prediction=logits.argmax(1)
        score=logits[:,0]-logits[:,1]
    return dict(score=score.detach().cpu().numpy().astype(np.float64),
                prediction=prediction.detach().cpu().numpy().astype(np.int64),
                probabilities=probabilities.detach().cpu().numpy().astype(np.float64))


def _tensor_roles(values,kind,spec,device):
    if kind=="lstm":
        profiles=reconstruct_profiles(values,spec)
        return tuple(torch.tensor(profiles[name],device=device) for name in ("ic","d","demographics"))
    return (torch.tensor(values,device=device),)


def _logits(model,parts,kind,batch_size):
    model.eval()
    values=[]
    size=len(parts[0]) if kind in ("fcn","made") else batch_size
    with torch.no_grad():
        for start in range(0,len(parts[0]),size): values.append(model(*(p[start:start+size] for p in parts)))
    return torch.cat(values)


def _loss(logits,labels,kind,class_weights=None):
    if kind=="gcn": return F.nll_loss(logits,labels)
    if kind=="lstm":
        weights=class_weights[labels] if class_weights is not None else None
        return F.binary_cross_entropy_with_logits(logits,labels.float(),weight=weights)
    return F.cross_entropy(logits,labels)


def _cpu_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def state_sha(state):
    h=hashlib.sha256()
    for name in sorted(state):
        value=state[name].detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def _source_hashes():
    names=("neural_models.py","vendor_report_networks.py","vendor_site_gcn.py","vendor_roi_profile.py",
           "vendor_original_made.py","neural_default_configs.json","neural_vendor_provenance.json")
    return {name:hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in names}


def save_model(bundle,path):
    path=_checkpoint_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp.%d"%os.getpid())
    try:
        torch.save(bundle,str(temporary))
        os.replace(str(temporary),str(path))
    finally:
        if temporary.exists(): temporary.unlink()
    return str(path)


def _checkpoint_path(path):
    path=Path(path).resolve()
    try: path.relative_to(ROOT.resolve())
    except ValueError: raise ValueError("Checkpoint writes must stay inside the new neural_extension root")
    return path


def load_model(path):
    try: bundle=torch.load(str(path),map_location="cpu",weights_only=False)
    except TypeError: bundle=torch.load(str(path),map_location="cpu")
    _need(bundle.get("format")==FORMAT,"Unsupported neural checkpoint")
    _need(bundle["source_hashes"]==_source_hashes(),"Checkpoint network source changed")
    _need(state_sha(bundle["state_dict"])==bundle["state_sha256"],"Checkpoint state hash mismatch")
    return bundle


def _train_made(a,y,z,yv,spec,cfg,device):
    nc=spec["n_connection_features"]
    tr=torch.tensor(a[:,:nc],device=device); va=torch.tensor(z[:,:nc],device=device)
    ae1,h1,e1=train_original_ae(tr,va,code_size=cfg["ae1_size"],epochs=cfg["ae1_epochs"],
        batch_size=cfg["ae1_batch_size"],lr=cfg["ae_lr"],corruption=cfg["ae1_corruption"],
        sparse_p=cfg["ae1_sparse_p"],sparse_coefficient=cfg["ae1_sparse_coefficient"])
    with torch.no_grad(): et,ev=ae1.encode(tr),ae1.encode(va)
    ae2,h2,e2=train_original_ae(et,ev,code_size=cfg["ae2_size"],epochs=cfg["ae2_epochs"],
        batch_size=cfg["ae2_batch_size"],lr=cfg["ae_lr"],corruption=cfg["ae2_corruption"])
    model=OriginalMLP(ae1,ae2,spec["n_demographics"],cfg["hidden_size"],cfg["dropout_keep_probabilities"]).to(device)
    del ae1,ae2,et,ev,tr,va
    tr=torch.tensor(a,device=device); va=torch.tensor(z,device=device)
    ty=torch.tensor(y,device=device); vy=torch.tensor(yv,device=device)
    optimizer=torch.optim.SGD(model.parameters(),lr=cfg["lr"],momentum=cfg["initial_momentum"])
    bs=cfg["batch_size"]; batches=len(a)//bs
    _need(batches>0,"Original MADE requires at least one full minibatch")
    best_accuracy,best_epoch,best_state=0.,-1,None
    history=[]
    for epoch_zero in range(cfg["epochs"]):
        alpha=min(max(epoch_zero/cfg["saturate_momentum"],0.),1.)
        momentum=cfg["initial_momentum"]*(1.-alpha)+cfg["final_momentum"]*alpha
        optimizer.param_groups[0]["momentum"]=momentum
        sums=[torch.zeros((),dtype=torch.float64,device=device) for _ in range(3)]
        for start in range(0,batches*bs,bs):
            model.train(); optimizer.zero_grad()
            logits=model(tr[start:start+bs]); loss=F.cross_entropy(logits,ty[start:start+bs])
            _need(bool(torch.isfinite(loss)),"Nonfinite MADE training loss")
            loss.backward(); optimizer.step(); sums[0]+=loss.detach()
            model.eval()
            with torch.no_grad():
                logits=model(va)
                sums[1]+=F.cross_entropy(logits,vy)
                sums[2]+=(logits.argmax(1)==vy).float().mean()
        train_loss,valid_loss,accuracy=[float(value.item())/batches for value in sums]
        _need(all(math.isfinite(value) for value in (train_loss,valid_loss,accuracy)),"Nonfinite original MADE validation statistic")
        improved=accuracy>best_accuracy and epoch_zero>cfg["start_saving_after_zero_based_epoch"]
        if improved: best_accuracy,best_epoch,best_state=accuracy,epoch_zero+1,_cpu_state(model)
        history.append(dict(epoch=epoch_zero+1,train_loss=train_loss,valid_loss=valid_loss,valid_accuracy=accuracy,
                            momentum=momentum,selected=bool(improved)))
    _need(best_state is not None,"No original MADE validation checkpoint after warmup")
    model.load_state_dict(best_state)
    bill=dict(supervised_epochs=len(history),supervised_optimizer_steps=batches*len(history),
              ae1_epochs=len(h1),ae2_epochs=len(h2),ae1_selected_epoch=e1,ae2_selected_epoch=e2,
              ae1_optimizer_steps=(len(a)//cfg["ae1_batch_size"])*len(h1),
              ae2_optimizer_steps=(len(a)//cfg["ae2_batch_size"])*len(h2),
              dropped_supervised_remainder=len(a)%bs,
              dropped_ae1_remainder=len(a)%cfg["ae1_batch_size"],dropped_ae2_remainder=len(a)%cfg["ae2_batch_size"],
              checkpoint_statistic="mean_within_epoch_valid_accuracy_strict_improvement_after_original_warmup",
              selected_checkpoint_statistic=best_accuracy,pretraining_history=dict(ae1=h1,ae2=h2))
    return model,history,best_epoch,bill


def fit(kind,Xtr,ytr,Xva,yva,*,checkpoint_path,config=None,seed=42,transductive_X=None,
        sites_train=None,sites_valid=None,sites_test=None,input_spec=None):
    """Train/validate only. Unlabeled test features allowed only for GCN graph."""
    started=time.perf_counter(); kind=canonical_kind(kind); cfg=canonical_config(kind,config)
    checkpoint_path=_checkpoint_path(checkpoint_path)
    a=_matrix(Xtr,"training"); z=_matrix(Xva,"validation",a.shape[1])
    y=_labels(ytr,len(a),"training labels"); yv=_labels(yva,len(z),"validation labels")
    _need(len(np.unique(y))==2,"Training requires both native classes")
    _need(kind!="fcn" or len(a)>=2,"FCN batch normalization requires two training examples")
    spec=_spec(input_spec,a.shape[1],kind)
    device=_device(cfg["device"]); _seed(seed,device)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    hashes=dict(train=array_sha(a),valid=array_sha(z),train_labels=array_sha(y),valid_labels=array_sha(yv))
    architecture={}; graph_meta=None; held=None
    if kind=="gcn":
        _need(transductive_X is not None,"GCN requires fixed unlabeled test-node features")
        held=_matrix(transductive_X,"unlabeled GCN nodes",a.shape[1])
        hashes["transductive_features"]=array_sha(held)
        codes,graph_meta=_site_codes((sites_train,sites_valid,sites_test),(a,z,held),device)
        graph_x=torch.tensor(np.vstack((a,z,held)),device=device)
    else:
        _need(transductive_X is None,"Only GCN may accept held-out features during fitting")
    if kind=="lstm":
        profile=reconstruct_profiles(a,spec)
        architecture["effective_direction_weight"]=1.
        architecture["training_direction_present"]=bool(np.any(profile["d"]!=0))
        del profile
    history=[]; bill={}; optimizer=None
    with threadpool_limits(limits=1):
        if kind=="made":
            model,history,best_epoch,bill=_train_made(a,y,z,yv,spec,cfg,device)
        else:
            model=_build_model(kind,a.shape[1],spec,cfg,architecture,device)
            tr_parts=_tensor_roles(a,kind,spec,device); va_parts=_tensor_roles(z,kind,spec,device)
            ty=torch.tensor(y,device=device); vy=torch.tensor(yv,device=device)
            optimizer_cls=torch.optim.AdamW if kind in ("lstm","transformer") else torch.optim.Adam
            optimizer=optimizer_cls(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
            weights=torch.tensor(len(y)/(2.*np.bincount(y,minlength=2)),dtype=torch.float32,device=device) if kind=="lstm" else None
            generator=torch.Generator(device="cpu"); generator.manual_seed(int(seed))
            best_accuracy,best_loss,best_epoch,best_state,stale=-1.,float("inf"),-1,None,0
            steps=0
            for epoch in range(1,cfg["epochs"]+1):
                model.train(); total=0.
                if kind in ("gcn","fcn"):
                    optimizer.zero_grad()
                    logits=model(graph_x,codes)[:len(a)] if kind=="gcn" else model(*tr_parts)
                    loss=_loss(logits,ty,kind)
                    _need(bool(torch.isfinite(loss)),"Nonfinite training loss")
                    loss.backward(); optimizer.step(); steps+=1; total=float(loss.item())*len(a)
                else:
                    order=torch.randperm(len(a),generator=generator)
                    for start in range(0,len(a),cfg["batch_size"]):
                        ix=order[start:start+cfg["batch_size"]].to(device)
                        optimizer.zero_grad(); logits=model(*(p[ix] for p in tr_parts))
                        loss=_loss(logits,ty[ix],kind,weights)
                        _need(bool(torch.isfinite(loss)),"Nonfinite training loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"] if kind=="lstm" else cfg["clip_grad_norm"])
                        optimizer.step(); steps+=1; total+=float(loss.item())*len(ix)
                model.eval()
                with torch.no_grad():
                    validation_logits=(model(graph_x,codes)[len(a):len(a)+len(z)] if kind=="gcn" else
                                       _logits(model,va_parts,kind,cfg.get("batch_size",len(z))))
                    accuracy=float(np.mean(_outputs(validation_logits,kind)["prediction"]==yv))
                    val_loss=float(_loss(validation_logits,vy,kind).item())
                _need(math.isfinite(val_loss),"Nonfinite validation loss")
                improved=accuracy>best_accuracy or (accuracy==best_accuracy and val_loss<best_loss)
                if improved:
                    best_accuracy,best_loss,best_epoch,best_state,stale=accuracy,val_loss,epoch,_cpu_state(model),0
                else: stale+=1
                history.append(dict(epoch=epoch,train_loss=total/len(a),valid_accuracy=accuracy,valid_loss=val_loss,selected=bool(improved)))
                if kind=="lstm" and stale>=cfg["patience"]: break
            _need(best_state is not None,"No finite validation checkpoint")
            model.load_state_dict(best_state)
            bill=dict(supervised_epochs=len(history),supervised_optimizer_steps=steps,
                      checkpoint_statistic="maximum_epoch_end_valid_native_accuracy_then_minimum_unweighted_validation_loss",
                      selected_checkpoint_statistic=best_accuracy)
        model.eval()
        with torch.no_grad():
            validation_logits=(model(graph_x,codes)[len(a):len(a)+len(z)] if kind=="gcn" else
                               _logits(model,_tensor_roles(z,kind,spec,device),kind,cfg.get("batch_size",len(z))))
            validation=_outputs(validation_logits,kind)
            actual_valid_loss=float(_loss(validation_logits,torch.tensor(yv,device=device),kind).item())
        state=_cpu_state(model)
        if device.type=="cuda": torch.cuda.synchronize(device)
        bill.update(elapsed_seconds=time.perf_counter()-started,parameter_count=sum(p.numel() for p in model.parameters()),
                    numerical_threads=1,device=str(device),selected_epoch=best_epoch,
                    peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0)
        metadata=dict(kind=kind,config=cfg,seed=int(seed),n_train=len(a),n_valid=len(z),input_width=a.shape[1],
                      input_spec=spec,input_sha256=hashes,architecture=architecture,best_epoch=best_epoch,
                      trained_epochs=len(history),validation_accuracy=float(np.mean(validation["prediction"]==yv)),
                      validation_loss=actual_valid_loss,checkpoint_rule=bill["checkpoint_statistic"],
                      label_semantics=dict(ASD=0,TC=1),score_semantics="ASD0-oriented logit difference / log odds",
                      preprocessing="Caller already fitted train-only scaler/demo fill and applied beta once; adapter applies none",
                      actual_demo_columns=spec["n_demographics"],
                      actual_connection_columns=spec["n_connection_features"],
                      effective_concat_demographics=spec["n_demographics"]>0,
                      legacy_concat_pheno_flag_ignored=cfg.get("concat_pheno") if kind=="gcn" else None,
                      beta_applied_by_adapter=False,test_labels_accepted=False,test_score_readout_during_fit=False,
                      heldout_features_seen_during_fit=(kind=="gcn"),graph=graph_meta,
                      actual_execution="eager_validation_only",requested_execution=cfg.get("execution"),
                      torch_version=str(torch.__version__),numpy_version=np.__version__,
                      LSTM_demo_extension="additive bias-free demographic TC1 logit head" if kind=="lstm" and spec["n_demographics"] else None,
                      LSTM_direction_convention="D[low,high]=native_positive_low_to_high; D[high,low]=-value; only selected columns; diag/unselected0" if kind=="lstm" else None,
                      LSTM_legacy_min_delta_ignored=cfg.get("min_delta") if kind=="lstm" else None,
                      candidate_resume_policy="reuse completed atomic checkpoint; interrupted fits restart deterministically from same seed")
        bundle=dict(format=FORMAT,kind=kind,config=cfg,input_spec=spec,architecture=architecture,
                    input_width=a.shape[1],state_dict=state,state_sha256=state_sha(state),
                    source_hashes=_source_hashes(),metadata=metadata,training_bill=bill)
        saved=save_model(bundle,checkpoint_path)
    del model,optimizer
    if device.type=="cuda": torch.cuda.empty_cache()
    return dict(model=bundle,checkpoint_path=saved,validation_score_ASD=validation["score"],
                validation_prediction_TC1=validation["prediction"],validation_probabilities=validation["probabilities"],
                score=validation["score"],prediction=validation["prediction"],metadata=metadata,history=history,training_bill=bill)


def predict(kind,checkpoint_path,Xtr,Xva,Xte,*,sites_train=None,sites_valid=None,sites_test=None,
            config=None,input_spec=None,device=None):
    """After the caller's seal: infer saved selected weights, without any labels."""
    kind=canonical_kind(kind); bundle=load_model(checkpoint_path)
    _need(bundle["kind"]==kind,"Requested classifier differs from checkpoint")
    cfg=bundle["config"]
    if config is not None:
        supplied=canonical_config(kind,config)
        _need({k:v for k,v in supplied.items() if k!="device"}=={k:v for k,v in cfg.items() if k!="device"},"Prediction configuration differs from saved model")
    spec=bundle["input_spec"]
    if input_spec is not None: _need(_spec(input_spec,bundle["input_width"],kind)==spec,"Prediction input specification differs")
    a=_matrix(Xtr,"training graph context",bundle["input_width"])
    z=_matrix(Xva,"validation",bundle["input_width"])
    x=_matrix(Xte,"held-out",bundle["input_width"])
    _need(array_sha(a)==bundle["metadata"]["input_sha256"]["train"] and
          array_sha(z)==bundle["metadata"]["input_sha256"]["valid"],"Saved train/validation context changed")
    actual_device=_device(device or (config or {}).get("device",cfg["device"]))
    _seed(bundle["metadata"]["seed"],actual_device)
    model=_build_model(kind,bundle["input_width"],spec,cfg,bundle["architecture"],actual_device)
    model.load_state_dict(bundle["state_dict"]); model.eval()
    with threadpool_limits(limits=1),torch.no_grad():
        if kind=="gcn":
            _need(array_sha(x)==bundle["metadata"]["input_sha256"]["transductive_features"],"GCN test-node features differ from its trained transductive graph")
            codes,graph_meta=_site_codes((sites_train,sites_valid,sites_test),(a,z,x),actual_device)
            _need(graph_meta==bundle["metadata"]["graph"],"GCN graph site identities changed")
            logits=model(torch.tensor(np.vstack((a,z,x)),device=actual_device),codes)
            valid=_outputs(logits[len(a):len(a)+len(z)],kind)
            test=_outputs(logits[len(a)+len(z):],kind)
        else:
            valid=_outputs(_logits(model,_tensor_roles(z,kind,spec,actual_device),kind,cfg.get("batch_size",len(z))),kind)
            test=_outputs(_logits(model,_tensor_roles(x,kind,spec,actual_device),kind,cfg.get("batch_size",len(x))),kind)
    del model
    if actual_device.type=="cuda": torch.cuda.empty_cache()
    return dict(validation_score_ASD=valid["score"],validation_prediction_TC1=valid["prediction"],
                validation_probabilities=valid["probabilities"],test_score_ASD=test["score"],
                test_prediction_TC1=test["prediction"],test_probabilities=test["probabilities"],metadata=bundle["metadata"],
                checkpoint_path=str(Path(checkpoint_path).resolve()))


predict_checkpoint = predict
