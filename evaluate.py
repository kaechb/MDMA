
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

    def weighted_mean(x):
        x,w=np.array(x[0]),np.array(x[1])
        weights=1/w**2
        return np.sum(x*weights)/np.sum(weights)
    def weighted_std(x):
        w=np.array(x[1])
        sigma=np.sqrt(1/sum(w))
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

    cols=["name","w1m","w1efp","w1m","pmm","pme","pmp","cov","mmd","fpd","kpd","time"]
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
    df.loc[:,"pmf"]*=10000
    df.loc[:,"kpd"]*=10000

    df.loc[:,"pmk"]*=10000
    df.loc[:,"w1m"]=df.apply(lambda x:format_mean_sd(float(x["w1m"]),float(x["pmm"])),axis=1)
    df.loc[:,"w1p"]=df.apply(lambda x:format_mean_sd(float(x["w1p"]),float(x["pmp"])),axis=1)
    df.loc[:,"w1efp"]=df.apply(lambda x:format_mean_sd(float(x["w1efp"]),float(x["pme"])),axis=1)
    df.loc[:,"kpd"]=df.apply(lambda x:format_mean_sd(float(x["kpd"]),float(x["pmk"])),axis=1)
    df.loc[:,"fpd"]=df.apply(lambda x:format_mean_sd(float(x["fpd"]),float(x["pmf"])),axis=1)
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
            if col not in ["w1m","w1p","w1efp","fpd","kpd","time"]:
                continue
            temp_index=temp[col].astype(str).str.replace("$","").str.replace("k","").str.split("\\").str[0].astype(float)
            mins=temp_index==temp_index.drop("IN").min()
            temp.loc[mins,col]="$\mathbf{"+temp.loc[mins,col].astype(str).str.replace("$","")+"}$"
        temp=temp.reset_index()[["model","w1m","w1p","w1efp","kpd","fpd","time"]]
        temp.columns=["model","$W_1^M (\times 10^{3})$","$W_1^P (\times 10^{3})$","$W_1^{EFP}(\times 10^{5})$","$KPD(\times 10^{4})$","$FPD$","Time $\mu s$"]
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
def jetnet_eval(model_dict,data_module,time_dict):

    results=pd.DataFrame()
    true=data_module.unscaled_real_test

    mpgan=torch.tensor(np.load("/gpfs/dust/maxwell/user/kaechben/thesis/jetnet30/MPGAN/t.npy")[:50000]).cuda().float()
    #epic= np.load("/home/kaechben/EPiC-GAN/{}_epic.npy".format("t"))[..., [1, 2, 0]]
    for name,temp in zip(model_dict.keys(),model_dict.values()):
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
        results_temp = {"name": ["t"], "model": [name], "w1m": [w1m_], "w1p": [w1p_],  "w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_],"time":[time_dict[name]]}#"w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_]
        results=pd.concat((results,pd.DataFrame(results_temp).set_index("name",drop=True)),axis=0)
    for name,temp in zip(["MPGAN","IN"],[mpgan,true]):
        w1m_ = w1m(temp[:, :, :3], true[:, :, :3], num_eval_samples=len(true),num_batches=16)
        w1efp_ =w1efp(temp[: len(true)], true[:, :, :3], num_eval_samples=len(true), efp_jobs=20)
        w1p_ = w1p(temp[: len(true),:,:3], true[:, :, :3], num_eval_samples=len(true))
        kpd_real = get_fpd_kpd_jet_features(true, efp_jobs=20)
        kpd_fake = get_fpd_kpd_jet_features(temp[: len(true),:,:3], efp_jobs=20)
        kpd_ = kpd(kpd_real, kpd_fake)
        fpd_ = fpd(kpd_real, kpd_fake, min_samples=10000, max_samples=len(true))
        results_temp = {"name": ["t"], "model": [name], "w1m": [w1m_], "w1p": [w1p_],  "w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_],"time":[35.7 if name=="MPGAN" else -1]}#"w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_]
        results=pd.concat((results,pd.DataFrame(results_temp).set_index("name",drop=True)),axis=0)
    return results



from utils.dataloader_jetnet import PointCloudDataloader


torch.set_float32_matmul_precision('medium' )



model_dict={}
time_dict={}
name="jet"
ckpt_dir="/gpfs/dust/maxwell/user/kaechben/thesis/jetnet30/"
ckpt_files = [f.split(".ckpt")[0] for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
for model_name in ckpt_files:
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
    else:
        continue

    model=load_model(state_dict,model)
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
    # model.on_validation_epoch_start=on_validation_epoch_start
    trainer = pl.Trainer(
            devices=1,
            accelerator="gpu",
        )
    model.w1m_best=0.00001
    trainer.validate(model, dataloaders=data_module.test_dataloader())
    model_dict[model_name]=torch.cat(model.fake)
    time_dict[model_name]=np.mean(model.times)/len(model_dict[model_name])

results=jetnet_eval(model_dict,data_module,time_dict)
print_table(results)