
import copy
import math
import pickle
import sys
import time
import traceback
import os

import hist
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hist import Hist
from jetnet.evaluation import (fpd, fpnd, get_fpd_kpd_jet_features, kpd, w1efp,
                               w1m, w1p)
from scipy.stats import wasserstein_distance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   StandardScaler)
from torch.nn import functional as FF
from torch.nn.utils import spectral_norm, weight_norm
from torch.nn.utils.rnn import pad_sequence
# from comet_ml import Experiment
from torch.nn.utils.weight_norm import WeightNorm
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from tqdm import tqdm

import utils.losses
import wandb
from fit import *

import pytorch_lightning as pl
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter
from utils.helpers import (MultiheadL2Attention, WeightNormalizedLinear, mass,
                     plotting_point_cloud, get_hists, plotting_thesis)
import pandas as pd


def create_mask(n, size=30):
    # Ensure n is a 1D tensor
    n = n.flatten()

    # Create a range tensor [0, 1, 2, ..., size-1]
    range_tensor = torch.arange(size).unsqueeze(0)

    # Compare range_tensor with n to create the mask
    mask = range_tensor >= n.unsqueeze(1)

    return mask

def load_model(ckpt,model):
    flow_state_dict = {k.replace('flow.', ''): v for k, v in ckpt["state_dict"].items() if 'flow' in k}
    if "adversarial" not in ckpt["hyper_parameters"].keys():
        ckpt["hyper_parameters"]["adversarial"]=False

    config=ckpt["hyper_parameters"]
    if "ckpt_flow" in config.keys():
        config["ckpt_flow"]="/".join(config["ckpt_flow"].split("/")[:-1])+"/jetnet30/"+config["ckpt_flow"].split("/")[-1]
    flow=model(**ckpt["hyper_parameters"])

    flow.flow.load_state_dict(flow_state_dict)
    if config["context_features"]>0 and config["pf"]:
        context_state_dict = {k.replace('shape.', ''): v for k, v in ckpt["state_dict"].items() if 'shape' in k}
        flow.shape.load_state_dict(context_state_dict)
        if config["adversarial"]:
            context_state_dict = {k.replace('dis_net.', ''): v for k, v in ckpt["state_dict"].items() if 'dis_net' in k}
            flow.dis_net.load_state_dict(context_state_dict)
    elif config["context_features"]>0 and config["model_name"].find("tnf")>-1:
        context_state_dict = {k.replace('gen_net.', ''): v for k, v in ckpt["state_dict"].items() if 'gen_net' in k}
        flow.gen_net.load_state_dict(context_state_dict)
    return flow

def format_mean_sd(mean, sd):
    """round mean and standard deviation to most significant digit of sd and apply latex formatting"""
    decimals = -int(np.floor(np.log10(sd)))
    decimals -= int((sd * 10 ** decimals) >= 9.5)

    if decimals < 0:
        ten_to = 10 ** (-decimals)
        if mean > ten_to:
            mean = ten_to * (mean // ten_to)
        else:
            mean_ten_to = 10 ** np.floor(np.log10(mean))
            mean = mean_ten_to * (mean // mean_ten_to)
        sd = ten_to * (sd // ten_to)
        decimals = 0

    if mean >= 1e3 and sd >= 1e3:
        mean = np.round(mean * 1e-3)
        sd = np.round(sd * 1e-3)
        return f"${mean:.{decimals}f}$k $\\pm {sd:.{decimals}f}$k"
    else:
        return f"${mean:.{decimals}f} \\pm {sd:.{decimals}f}$"
def print_table(results):
    results.to_csv("results.csv")
    def weighted_mean(x):
        x,w=np.array(x[0]),np.array(x[1])
        weights=1/w**2
        return np.sum(x*weights)/np.sum(weights)
    def weighted_std(x):
        w=np.array(x[1])**(-2)
        sigma=np.sqrt(1/np.sum(w))
        return sigma
    df=results
    df["pmm"]=df["w1m"].apply(lambda x:x).apply(weighted_std)
    df["w1m"]=df["w1m"].apply(lambda x:x).apply(weighted_mean)
    df["pmp"]=df["w1p"].apply(lambda x:x).apply(weighted_std)
    df["w1p"]=df["w1p"].apply(lambda x:x).apply(weighted_mean)
    df["pme"]=df["w1efp"].apply(lambda x:x).apply(weighted_std)
    df["w1efp"]=df["w1efp"].apply(lambda x:x).apply(weighted_mean)

    df["fpd_std"]=df["fpd"].apply(lambda x:x[1]).apply(np.mean)
    df["fpd"]=df["fpd"].apply(lambda x:x[0]).apply(np.mean)
    df["kpd_std"]=df["kpd"].apply(lambda x:x[1]).apply(np.mean)
    df["kpd"]=df["kpd"].apply(lambda x:x[0]).apply(np.mean)

    replace_dict={"MPGAN":"MPGAN","t_cpflow":"PF","t_ipflow":"IPF","t_apflow":"APF","t_nflow":"NF","t_ccnflow":"NF(cc)","t_cnflow":"NF(c)","t_tnflow":"TNF","IN":"IN"}
    df.loc[:,"model"]=df["model"].apply(lambda x:replace_dict[x])
    df=df.set_index("model",drop=True)
    df.loc[:,"w1m"]*=1000
    df.loc[:,"w1p"]*=1000
    df["w1p"]=df["w1p"]
    df.loc[:,"w1efp"]*=100000
    df["w1efp"]=df["w1efp"]
    df.loc[:,"pmm"]*=1000
    df.loc[:,"pmp"]*=1000
    df.loc[:,"pme"]*=100000
    df.loc[:,"fpd"]*=10000
    df.loc[:,"fpd_std"]*=10000
    df.loc[:,"kpd"]*=10000

    df.loc[:,"kpd_std"]*=10000
    df.loc[:,"w1m"]=df.apply(lambda x:format_mean_sd(float(x["w1m"]),float(x["pmm"])),axis=1)
    df.loc[:,"w1p"]=df.apply(lambda x:format_mean_sd(float(x["w1p"]),float(x["pmp"])),axis=1)
    df.loc[:,"w1efp"]=df.apply(lambda x:format_mean_sd(float(x["w1efp"]),float(x["pme"])),axis=1)
    df.loc[:,"kpd"]=df.apply(lambda x:format_mean_sd(float(x["kpd"]),float(x["kpd_std"])),axis=1)
    df.loc[:,"fpd"]=df.apply(lambda x:format_mean_sd(float(x["fpd"]),float(x["fpd_std"])),axis=1)
    df.loc[:,"time"]*=1e6
    df.loc[:,"time"]=np.round(df["time"],decimals=1)

    order=["PF","IPF","APF","NF","NF(c)","NF(cc)","TNF","MPGAN","IN"]
    df=df.loc[order,:]
    print(df)
    # def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print (count_parameters(model.gen_net))
    tex=""
    for p in ["t"]:
        temp=df
        for col in df.columns:
            if col not in ["w1m","w1p","w1efp","fpd","kpd","time","parameters"]:
                continue
            temp_index=temp[col].astype(str).str.replace("$","").str.replace("k","").str.split("\\").str[0].astype(float)
            mins=temp_index==temp_index.drop("IN").min()
            if col=="parameters":
                mins=temp_index==temp_index.drop("IN").max()
            temp.loc[mins,col]="$\mathbf{"+temp.loc[mins,col].astype(str).str.replace("$","")+"}$"
        temp=temp.reset_index()[["model","w1m","w1p","w1efp","kpd","fpd","time","parameters"]]
        temp.columns=["model","$W_1^M (\times 10^{3})$","$W_1^P (\times 10^{3})$","$W_1^{EFP}(\times 10^{5})$","$\texttt{KPD}(\times 10^{4})$","$\texttt{FPD}$","Time $\mu s$", "# Parameters"]
        text=temp.to_latex(index=False,escape=False)
        parton="Gluon" if p=="g" else "Light Quark" if p=="q" else "Top Quark"
        tex+="\multirow{9}{*}{"+parton+"} & "+text.split("FPD")[1].split("\\bottomrule")[0].replace("\\\\","\\\\&").replace("\\midrule","").replace("  ","")[:-2]+"\cline{1-8}"
        tex+="\n"
    print(tex)

def plot(name,fake,true):
    true=true[:,:,:3]
    m_f,m_t=mass(fake),mass(true)
    mins=torch.quantile(torch.cat((fake.reshape(-1,3),true.reshape(-1,3))),0.001,dim=0)
    maxs=torch.quantile(torch.cat((fake.reshape(-1,3),true.reshape(-1,3))),0.999,dim=0)
    fake=torch.clamp(fake,min=mins,max=maxs)
    true=torch.clamp(true,min=mins,max=maxs)
    mins=torch.cat((mins,torch.quantile(torch.cat((m_f,m_t)),0.001,dim=0).unsqueeze(0)))
    maxs=torch.cat((maxs,torch.quantile(torch.cat((m_f,m_t)),0.999,dim=0).unsqueeze(0)))*1.01
    m_f=torch.clamp(m_f,min=torch.quantile(torch.cat((m_f,m_t)),0.001,dim=0),max=torch.quantile(torch.cat((m_f,m_t)),0.999,dim=0))
    m_t=torch.clamp(m_t,min=torch.quantile(torch.cat((m_f,m_t)),0.001,dim=0),max=torch.quantile(torch.cat((m_f,m_t)),0.999,dim=0))
    hists=get_hists([30,30,30,30],mins,maxs,calo=model.name=="calo")
    masks=torch.cat(model.masks)

    for var in range(3):
        hists["hists_real"][var].fill(true.reshape(-1,3)[(true.reshape(-1,3)!=0).all(1)][:,var].cpu().numpy())
        hists["hists_fake"][var].fill(fake.reshape(-1,3)[(fake.reshape(-1,3)!=0).all(1)][:,var].cpu().numpy())
    hists["hists_real"][3].fill(m_t.cpu().numpy())
    hists["hists_fake"][3].fill(m_f.cpu().numpy())

    plot=plotting_thesis()
    plot.plot_ratio(hists["hists_real"],hists["hists_fake"],weighted=False,leg=2,model_name=name)
    plot.plot_corr(true,fake,name,leg=-1)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def jetnet_eval(model_dict,data_module,time_dict,debug=False):

    results=pd.DataFrame()
    true=data_module.unscaled_real_test

    mpgan=torch.tensor(np.load("/gpfs/dust/maxwell/user/kaechben/thesis/jetnet30/MPGAN/t.npy")[:50000]).cuda().float()
    #epic= np.load("/home/kaechben/EPiC-GAN/{}_epic.npy".format("t"))[..., [1, 2, 0]]
    for name,temp in zip(model_dict.keys(),model_dict.values()):
        if debug:
            results_temp={"name": ["t"], "model": [name], "w1m": [0], "w1p": [0],  "w1efp": [0], "kpd": [0], "fpd": [0],"time":[0],"parameters":[0]}
        else:
            plot(name,temp,true)
            w1m_ = w1m(temp[:, :, :3], true[:, :, :3], num_eval_samples=len(true),num_batches=16)
            kpd_real = get_fpd_kpd_jet_features(true, efp_jobs=20)
            kpd_fake = get_fpd_kpd_jet_features(temp[: len(true),:,:3], efp_jobs=20)
            kpd_ = kpd(kpd_real, kpd_fake)
            fpd_ = fpd(kpd_real, kpd_fake, min_samples=10000, max_samples=len(true))
            i = 0
            w_dist_list = []
            w1efp_ =w1efp(temp[: len(true)], true[:, :, :3], num_eval_samples=len(true), efp_jobs=20)
            w1p_ = w1p(temp[: len(true),:,:3], true[:, :, :3], num_eval_samples=len(true))
            results_temp = {"name": ["t"], "model": [name], "w1m": [w1m_], "w1p": [w1p_],  "w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_],"time":[time_dict[name]],"parameters":[model.num_param]}#"w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_]
        results=pd.concat((results,pd.DataFrame(results_temp).set_index("name",drop=True)),axis=0)
    for name,temp in zip(["MPGAN","IN"],[mpgan,true]):
        if debug:
            results_temp={"name": ["t"], "model": [name], "w1m": [0], "w1p": [0],  "w1efp": [0], "kpd": [0], "fpd": [0],"time":[0],"parameters":[0]}
        else:
            w1m_ = w1m(temp[:, :, :3], true[:, :, :3], num_eval_samples=len(true),num_batches=16)
            w1efp_ =w1efp(temp[: len(true)], true[:, :, :3], num_eval_samples=len(true), efp_jobs=20)
            w1p_ = w1p(temp[: len(true),:,:3], true[:, :, :3], num_eval_samples=len(true))
            kpd_real = get_fpd_kpd_jet_features(true, efp_jobs=20)
            kpd_fake = get_fpd_kpd_jet_features(temp[: len(true),:,:3], efp_jobs=20)
            kpd_ = kpd(kpd_real, kpd_fake)
            fpd_ = fpd(kpd_real, kpd_fake, min_samples=10000, max_samples=len(true))
            results_temp = {"name": ["t"], "model": [name], "w1m": [w1m_], "w1p": [w1p_],  "w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_],"time":[35.7*1e-6 if name=="MPGAN" else -1],"parameters":[361123+355617]}#"w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_]
        results=pd.concat((results,pd.DataFrame(results_temp).set_index("name",drop=True)),axis=0)
    return results



from utils.dataloader_jetnet import PointCloudDataloader

def create_mask(n, size=30):
    # Ensure n is a 1D tensor
    n = n.flatten()

    # Create a range tensor [0, 1, 2, ..., size-1]
    range_tensor = torch.arange(size).unsqueeze(0)

    # Compare range_tensor with n to create the mask
    mask = range_tensor >= n.unsqueeze(1)

    return mask
def fit_kde(n,m,n_max=30):

    from sklearn.neighbors import KernelDensity
    kde=KernelDensity(bandwidth="scott").fit(n.unsqueeze(1).numpy())
    nhat=kde.sample(100000)
    nhat=nhat[nhat<n_max+1]
    mass_distributions = {int(i):m[n==i] for i in n.unique()}
    mass_kdes = {int(i):KernelDensity(bandwidth=1e-3).fit(mass_distributions[i].unsqueeze(1).numpy()) for i in mass_distributions.keys()}


    return kde,mass_kdes
def sample_kde(n,n_kde,m_kde=False):
    #fit kde
    nhat=n_kde.sample(n)
    np.random.shuffle(nhat)
    if m_kde:
        nhat_hist,bins=np.histogram(nhat+0.01,bins=np.arange(0,31),density=False)
        nhat_hist,bins=torch.tensor(nhat_hist),torch.tensor(bins)
        nhat_hist,bins=nhat_hist[nhat_hist>0],bins[1:][nhat_hist>0]

        n_dict={int(i):j for i,j in zip(bins,nhat_hist)}
        ms=[torch.from_numpy(m_kde[int(i)].sample(n_dict[int(i)])) for i in bins]
        mhat=torch.cat(ms).numpy()
        np.random.shuffle(mhat)

        return torch.from_numpy(nhat),torch.from_numpy(mhat)
    else:
        return nhat
torch.set_float32_matmul_precision('medium' )



model_dict={}
time_dict={}
name="jet"
ckpt_dir="./ckpts/"
ckpt_files = [f.split(".ckpt")[0] for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
for model_name in ckpt_files:
    if model_name=="t_flow" or model_name.find("calo")>-1:
        continue
    print(model_name)
    state_dict=torch.load(ckpt_dir+model_name+".ckpt")
    config=state_dict["hyper_parameters"]
    config["model_name"]=model_name
    if model_name.find("_ipf")>-1 or model_name.find("_cpf")>-1 or model_name.find("_apf")>-1:
        config["pf"]=True if not model_name.find("ipf")>-1 else False
        from fit.fit_pnf import PNF as model
    elif model_name.find("_nf")>-1 or model_name.find("_ccnf")>-1 or model_name.find("_cnf")>-1:
        from fit.fit_nf import NF as model
        config["pf"]=False
    elif model_name.find("_tnf")>-1:
        from fit.fit_tnf import TNF as model
        config["pf"]=False
    # elif model_name.find("mdma_fm")>-1:
    #     from fit.fit_jet_fm import FM as model
    #     config["num_part_eval"]=30
    # elif model_name.find("mdma_gan")>-1:
    #     from fit.fit import MDMA as model
    #     config["num_part_eval"]=30
    else:
        continue

    model=load_model(state_dict,model)
    config["batch_size"]=10000
    data_module = PointCloudDataloader(**config)
    data_module.setup("fit")
    model.bins=[100,100,100]
    model.n_dim = 3
    model.scaler=data_module.scaler
    model.min_pt=data_module.min_pt
    model.max_pt=data_module.max_pt
    mins=torch.ones(config["n_dim"]).unsqueeze(0)
    maxs=torch.ones(config["n_dim"]).unsqueeze(0)
    n=[]
    for i in data_module.train_dataloader():
        mins=torch.min(torch.cat((mins,i[0][~i[1]].min(0,keepdim=True)[0]),dim=0),dim=0)[0].unsqueeze(0)
        maxs=torch.max(torch.cat((maxs,i[0][~i[1]].max(0,keepdim=True)[0]),dim=0),dim=0)[0].unsqueeze(0)
        n.append((~i[1]).sum(1))
    model.maxs=maxs.cuda()
    model.mins=mins.cuda()
    model.avg_n=torch.cat(n,dim=0).float().cuda().mean()
    model.swa=False
    model.scaler=model.scaler.to("cuda")
    model.scaler.std=model.scaler.std.cuda()
    model.scaled_mins=torch.tensor(data_module.mins).cuda()
    model.scaled_maxs=torch.tensor(data_module.maxs).cuda()
    data=torch.cat((data_module.train_dataloader().dataset,data_module.test_dataloader().dataset),dim=0)
    m=mass(data_module.scaler.inverse_transform(data[:,:,:3].cuda())).cpu()
    n=(~data[:,:,-1].bool()).float().sum(1).cpu()
    n_kde,m_kde=fit_kde(n,m)
    n,m=sample_kde(len(data),n_kde,m_kde)
    # Trainer setup and model validation
    trainer = pl.Trainer(devices=1, accelerator="gpu")
    model.eval_metrics=False
    model.batch=[]
    model.masks=[]
    model.fake=[]
    model.conds=[]
    model=model.cuda()
    model.times_=[]
    with torch.no_grad():
        for batch in data_module.test_dataloader():

            mask=batch[1].cuda().bool()
            batch=batch[0].cuda()
            m=mass(model.scaler.inverse_transform(batch)).unsqueeze(1)
            if model.hparams.context_features>0:
                start=time.time()
                n,m=sample_kde(len(batch)*10,n_kde,m_kde)
                m=m[:len(batch)]
                n=n[:len(batch)]
                mask=create_mask(n).cuda()
            else:
                start=time.time()
            fake,_=model.sampleandscale(batch.cuda(),mask=mask, cond=m[:len(batch)].cuda().float() if model.hparams.context_features>0 else None,scale=True)
            model.times_.append(time.time()-start)
            batch=model.scaler.inverse_transform(batch)
            model.batch.append(batch.cpu())
            model.fake.append(fake.cpu())
            model.masks.append(mask.cpu())
            model.num_param=count_trainable_params(model)
            if model.hparams.context_features>0:
                model.conds.append(m.unsqueeze(1).cpu())

    # concatenate all batches
    fake = torch.cat(model.fake)
    true = torch.cat(model.batch)
    sorted_indices = torch.argsort(fake[:,:,2], dim=1, descending=True)
    fake = torch.gather(fake, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, fake.shape[2]))
    model_dict[model_name]=torch.cat(model.fake)

    if "num_part_eval" in config:
                sorted_indices = torch.argsort(fake[:, :, 2], dim=1,descending=True)

# Use gathered indices to sort the entire tensor
                N, M, P = fake.shape
                N_range = torch.arange(N).view(N, 1).expand(N, M)
                fake = fake[N_range, sorted_indices]
                fake=fake[:,:config["num_part_eval"]]
                real=real[:,:config["num_part_eval"]]
                if len(masks.shape)==2:
                    masks=masks.unsqueeze(-1)
                masks=masks[N_range, sorted_indices]
                masks=masks[:,:config["num_part_eval"]]
    time_dict[model_name]=np.mean(np.array(model.times_))/len(model_dict[model_name])

results=jetnet_eval(model_dict,data_module,time_dict,debug=False)
print_table(results)