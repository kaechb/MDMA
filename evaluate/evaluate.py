
import ast
import copy
import math
import os
import pickle
import sys
import time
import traceback

import hist
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
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
from torch.nn import Parameter
from torch.nn import functional as FF
from torch.nn.utils import spectral_norm, weight_norm
from torch.nn.utils.rnn import pad_sequence
# from comet_ml import Experiment
from torch.nn.utils.weight_norm import WeightNorm
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from tqdm import tqdm

import wandb
from fit import *
from utils import losses
from utils.helpers import (MultiheadL2Attention, WeightNormalizedLinear,
                           create_mask, fit_kde, get_hists, mass,
                           plotting_point_cloud, plotting_thesis, sample_kde)

notebook_path = os.getcwd()
import time

# Construct the path to the directory containing 'fit'

os.chdir(notebook_path)
import sys

# Add this parent directory to the system path
sys.path.insert(0, notebook_path)


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
def convert_to_numpy_arrays(s):
    s = s.replace('array(', '[').replace(')', ']').replace("(","[")    # Safely evaluate the string to a tuple
    evaluated = ast.literal_eval(s)

    # Convert each element of the tuple (which should be a list) to a NumPy array
    return tuple(np.array(element) for element in evaluated)

# Apply the function to each element of the series

def print_table(results):
    def weighted_mean(x):
        x,w=np.array(x[0]),np.array(x[1])
        weights=1/w**2
        return np.sum(x*weights)/np.sum(weights)
    def weighted_std(x):
        w=np.array(x[1])**(-2)
        sigma=np.sqrt(1/np.sum(w))
        return sigma
    df=results

    float_cols=["w1m","w1p","w1efp","kpd","fpd","time","parameters"]
    for col in float_cols[:-2]:
            print(col)
            try:
                df[col]=df[col].apply(ast.literal_eval)
            except:
                df[col]=df[col].apply(convert_to_numpy_arrays)

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

    replace_dict={"MPGAN":"MPGAN","t_tf":"TGAN","t_pf":"PF","t_ipf":"IPF","t_apf":"APF","t_nf":"NF","t_ccnf":"NF(cc)","t_cnf":"NF(c)","t_tnf":"TNF","IN":"IN","t_tf_slow":"TGAN2"}
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

    order=["PF","IPF","APF","NF","NF(c)","NF(cc)","TNF","TGAN","MPGAN","IN"]
    print(df)
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
                mins=temp_index==temp_index.drop("IN").min()
            temp.loc[mins,col]="$\mathbf{"+temp.loc[mins,col].astype(str).str.replace("$","")+"}$"
        temp=temp.reset_index()[["model","w1m","w1p","w1efp","kpd","fpd","time","parameters"]]
        temp.columns=["model","$W_1^M (\times 10^{3})$","$W_1^P (\times 10^{3})$","$W_1^{EFP}(\times 10^{5})$","$\texttt{KPD}(\times 10^{4})$","$\texttt{FPD}$","Time $\mu s$", "# Parameters"]
        text=temp.to_latex(index=False,escape=False)
        parton="Gluon" if p=="g" else "Light Quark" if p=="q" else "Top Quark"
        tex+="\multirow{9}{*}{"+parton+"} & "+text.split("FPD")[1].split("\\bottomrule")[0].replace("\\\\","\\\\&").replace("\\midrule","").replace("  ","")[:-2]+"\cline{1-8}"
        tex+="\n"
    print(tex)

def plot(name,fake,true,boxcox,groups):
    true=true[:,:,:3]
    m_f,m_t=mass(fake),mass(true)
    mins=torch.quantile(true.reshape(-1,3),0.0,dim=0)
    maxs=torch.quantile(true.reshape(-1,3),1,dim=0)
    mins[0]=-0.5
    maxs[0]=0.5
    fake=torch.clamp(fake,min=mins,max=maxs)
    true=torch.clamp(true,min=mins,max=maxs)
    mins=torch.cat((mins,torch.tensor(m_t.min()).unsqueeze(0)))
    maxs=torch.cat((maxs,torch.quantile(m_t,1,dim=0).unsqueeze(0)))*1
    print(mins,maxs)
    m_f=torch.clamp(m_f,min=torch.quantile(m_t,0),max=torch.quantile(m_t,1))
    m_t=torch.clamp(m_t,min=torch.quantile(m_t,0,dim=0),max=torch.quantile(m_t,1,dim=0))
    hists=get_hists([50,50,50,50],mins-0.01,maxs+0.01,calo=model.name=="calo")
    masks=torch.cat(model.masks)

    for var in range(3):
        hists["hists_real"][var].fill(true.reshape(-1,3)[(true.reshape(-1,3)!=0).all(1)][:,var].cpu().numpy())
        hists["hists_fake"][var].fill(fake.reshape(-1,3)[(fake.reshape(-1,3)!=0).all(1)][:,var].cpu().numpy())
    hists["hists_real"][3].fill(m_t.cpu().numpy())
    hists["hists_fake"][3].fill(m_f.cpu().numpy())
    replace_dict={"MPGAN":"MPGAN","t_tf":"TGAN","t_pf":"PF","t_ipf":"IPF","t_apf":"APF","t_nf":"NF","t_ccnf":"NF(cc)","t_cnf":"NF(c)","t_tnf":"TNF","IN":"IN","t_tf_slow":"TGAN2"}
    plot=plotting_thesis()
    try:
        save_name=name if not boxcox else "boxcox/"+name
        plot.plot_ratio(hists["hists_real"],hists["hists_fake"],weighted=False,leg=2,model_name=name,legend_name=replace_dict[name])
        if name.find("t_tf")!=-1 or name.find("t_tnf")!=-1:

            groups["GANs"].append(plot.plot_ratio_multiple(hists["hists_real"],hists["hists_fake"],weighted=False,leg=2,model_name=save_name,legend_name=replace_dict[name],group_name="GANs",groups=groups,boxcox=boxcox))
        elif name.find("t_pf")!=-1 or name.find("t_ipf")!=-1 or name.find("t_apf")!=-1:
            groups["PFs"].append(plot.plot_ratio_multiple(hists["hists_real"],hists["hists_fake"],weighted=False,leg=2,model_name=save_name,legend_name=replace_dict[name],group_name="PFs",groups=groups,boxcox=boxcox))
        elif name.find("t_nf")!=-1 or name.find("t_ccnf")!=-1 or name.find("t_cnf")!=-1:
            groups["NFs"].append(plot.plot_ratio_multiple(hists["hists_real"],hists["hists_fake"],weighted=False,leg=2,model_name=save_name,legend_name=replace_dict[name],group_name="NFs",groups=groups,boxcox=boxcox))

        #plot.plot_ratio(hists["hists_real"],hists["hists_fake"],weighted=False,leg=2,model_name=save_name,legend_name=replace_dict[name])
        plot.plot_corr(true,fake,save_name,leg=-1)
    except:
        traceback.print_exc()
        print(name)
    return groups

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def jetnet_eval(model_dict,data_module,time_dict,param_dict,boxcox,debug=False):

    results=pd.DataFrame()
    true=data_module.real_test
    train=data_module.train
    groups={"GANs":[], "NFs":[],"PFs":[]}

    mpgan=torch.tensor(np.load("/gpfs/dust/maxwell/user/kaechben/thesis/jetnet30/MPGAN/t.npy")[:50000]).cuda().float()
    #epic= np.load("/home/kaechben/EPiC-GAN/{}_epic.npy".format("t"))[..., [1, 2, 0]]
    for name,temp,num_param in zip(model_dict.keys(),model_dict.values(),param_dict.values()):
        if debug:
            results_temp={"name": ["t"], "model": [name], "w1m": [0], "w1p": [0],  "w1efp": [0], "kpd": [0], "fpd": [0],"time":[0],"parameters":[num_param]}
        else:
            groups=plot(name,temp,true,boxcox,groups)
            w1m_ = w1m(temp[:, :, :3], true[:, :, :3], num_eval_samples=len(true))
            kpd_real = get_fpd_kpd_jet_features(true, efp_jobs=20)
            kpd_fake = get_fpd_kpd_jet_features(temp[: len(true),:,:3], efp_jobs=20)
            kpd_ = kpd(kpd_real, kpd_fake)
            fpd_ = fpd(kpd_real, kpd_fake,)
            i = 0
            w_dist_list = []
            w1efp_ =w1efp(temp[: len(true)], true[:, :, :3], num_eval_samples=len(true), efp_jobs=20)
            w1p_ = w1p(temp[: len(true),:,:3], true[:, :, :3], num_eval_samples=len(true))
            results_temp = {"name": ["t"], "model": [name], "w1m": [w1m_], "w1p": [w1p_],  "w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_],"time":[time_dict[name]],"parameters":[num_param]}#"w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_]
        results=pd.concat((results,pd.DataFrame(results_temp).set_index("name",drop=True)),axis=0)
        torch.save(temp,"/beegfs/desy/user/kaechben/thesis/jetnet30/{}.pt".format(name))
    for name,temp in zip(["MPGAN","IN"],[mpgan,train]):
        if debug:
            results_temp={"name": ["t"], "model": [name], "w1m": [0], "w1p": [0],  "w1efp": [0], "kpd": [0], "fpd": [0],"time":[0],"parameters":[0]}
        else:
            w1m_ = w1m(temp[:, :, :3], true[:, :, :3], num_eval_samples=len(true))
            w1efp_ =w1efp(temp[: len(true)], true[:, :, :3], num_eval_samples=len(true), efp_jobs=20)
            w1p_ = w1p(temp[: len(true),:,:3], true[:, :, :3], num_eval_samples=len(true))
            kpd_real = get_fpd_kpd_jet_features(true, efp_jobs=20)
            kpd_fake = get_fpd_kpd_jet_features(temp[: len(true),:,:3], efp_jobs=20)
            kpd_ = kpd(kpd_real, kpd_fake)
            fpd_ = fpd(kpd_real, kpd_fake,)
            results_temp = {"name": ["t"], "model": [name], "w1m": [w1m_], "w1p": [w1p_],  "w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_],"time":[35.7*1e-6 if name=="MPGAN" else -1],"parameters":[361123+355617]}#"w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_]
        results=pd.concat((results,pd.DataFrame(results_temp).set_index("name",drop=True)),axis=0)

    return results






torch.set_float32_matmul_precision('medium' )


boxcox=True
model_dict={}
time_dict={}
param_dict={}
name="jet"
ckpt_dir="./ckpts/"
ckpt_files = [f.split(".ckpt")[0] for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
data_dir="/beegfs/desy/user/kaechben/thesis/jetnet30/data_generated/"

if not boxcox:
    jetnet30=["t_ipf_std","t_pf_std","t_apf_std","t_tf_std","t_tnf_std","t_nf_std","t_cnf_std","t_ccnf_std",]#,
else:
    jetnet30=["t_ipf","t_pf","t_apf","t_tf","t_tnf","t_nf","t_cnf","t_ccnf",]#,
load=True

for model_name in jetnet30:

    print(model_name)
    state_dict=torch.load(ckpt_dir+model_name+".ckpt")
    config=state_dict["hyper_parameters"]
    config["model_name"]=model_name
    if model_name.find("_ipf")>-1 or model_name.find("_pf")>-1 or model_name.find("_apf")>-1:
        from fit.fit_pnf import PNF as model
    elif model_name.find("_nf")>-1 or model_name.find("_ccnf")>-1 or model_name.find("_cnf")>-1:
        from fit.fit_nf import NF as model
        config["pf"]=False
    elif model_name.find("_tnf")>-1:
        from fit.fit_tnf import TNF as model
    elif model_name.find("_tf")>-1:
        from fit.fit_tf import TF as model


    model=model.load_from_checkpoint(ckpt_dir+model_name+".ckpt",boxcox=boxcox)
    model_name=model_name.replace("_std","")

    config["batch_size"]=1000

    if boxcox:
        from utils.dataloader_jetnet import PointCloudDataloader
    else:
        from utils.dataloader_jetnet_std import PointCloudDataloader

    data_module = PointCloudDataloader(**config)
    data_module.setup("fit")
    model.bins=[100,100,100]
    model.n_dim = 3
    if boxcox:

        model.pt_scaler=data_module.scaler[1]
        model.scaler=data_module.scaler[0]
        print(model.pt_scaler)

    else:
        model.scaler=data_module.scaler
        print(model.scaler)

    # calculate data bounds
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
    model.scaler=model.scaler.to("cuda")
    model.scaler.std=model.scaler.std.cuda()
    model.scaled_mins=torch.tensor(data_module.mins).cuda()
    model.scaled_maxs=torch.tensor(data_module.maxs).cuda()

    train=data_module.train_dataloader().dataset.cuda()
    test=data_module.test_dataloader().dataset.cuda()
    if boxcox:
        std=torch.cat((model.scaler.inverse_transform(train[:,:,:-2]),test[:,:,:-2]),dim=0)
        pt=torch.cat((model.pt_scaler.inverse_transform(train[:,:,-2].unsqueeze(-1)),test[:,:,-2].unsqueeze(-1)),dim=0)
        data=torch.cat((std,pt),dim=2)

    else:
        data=torch.cat((model.scaler.inverse_transform(train[:,:,:-1]),test[:,:,:-1]),dim=0)
    mask=torch.cat((train[:,:,-1],test[:,:,-1]),dim=0)

    m=mass(data.cuda()).cpu()
    n=(~mask[:,:].bool()).float().sum(1).cpu()
    n_kde,m_kde=fit_kde(n,m)
    n,m=sample_kde(len(data),n_kde,m_kde)
    # Trainer setup and model validation
    trainer = pl.Trainer(devices=1,logger=None,accelerator="gpu")
    model.eval_metrics=False
    model.batch=[]
    model.masks=[]
    model.fake=[]
    model.conds=[]
    model=model.cuda()
    model.times_=[]
    model.load_datamodule(data_module)
    with torch.no_grad():
        trainer.test(model,dataloaders=data_module.test_dataloader())
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
    time_dict[model_name]=np.mean(np.array(model.times))/len(model_dict[model_name])
    param_dict[model_name]=count_trainable_params(model)
    if load:
        break
if load:
    with open(data_dir+"/model_dict.pkl", "rb") as f:
        model_dict = pickle.load(f)
    with open(data_dir+"/time_dict.pkl", "rb") as f:
        time_dict = pickle.load(f)
    with open(data_dir+"/param_dict.pkl", "rb") as f:
        param_dict = pickle.load(f)

if not load:
    with open(data_dir+"/model_dict.pkl", "wb") as f:
        pickle.dump(model_dict, f)
    with open(data_dir+"/time_dict.pkl", "wb") as f:
        pickle.dump(time_dict, f)
    with open(data_dir+"/param_dict.pkl", "wb") as f:
        pickle.dump(param_dict, f)

results=jetnet_eval(model_dict,data_module,time_dict,param_dict,debug=False,boxcox=boxcox)
results.to_csv("results.csv")
results=pd.read_csv("results.csv")
print_table(results)