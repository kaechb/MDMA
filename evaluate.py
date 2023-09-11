
import copy
import math
import pickle
import sys
import time
import traceback

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

import losses
import wandb
from fit import MDMA
from helpers import (MultiheadL2Attention, WeightNormalizedLinear, mass,
                     plotting_point_cloud)
#from preprocess import ScalerBase,DQ,LogitTransformer
from preprocess_new import *
from preprocess_new import (DQ, DQLinear, LogitTransformer, ScalerBaseNew,
                            SqrtTransformer)
import pandas as pd
# def df(results):
#      # df.index=["t"]#"q"

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
    # print_table.index=["t"]#"q"
    df=pd.DataFrame(results).copy()

    df["pmm"]=df["w1m"].apply(lambda x:x[1].mean())
    df["w1m"]=df["w1m"].apply(lambda x:x[0].mean())
    df["pmp"]=df["w1p"].apply(lambda x:x[1].mean())
    df["w1p"]=df["w1p"].apply(lambda x:x[0].mean())
    df["pme"]=df["w1efp"].apply(lambda x:x[1].mean())
    df["w1efp"]=df["w1efp"].apply(lambda x:x[0].mean())
    df["pmf"]=df["fpd"].apply(lambda x:x[1].mean())
    df["fpd"]=df["fpd"].apply(lambda x:x[0].mean())
    df["pmk"]=df["kpd"].apply(lambda x:x[1].mean())
    df["kpd"]=df["kpd"].apply(lambda x:x[0].mean())


    cols=["name","w1m","w1efp","w1m","pmm","pme","pmp","cov","mmd","fpd","kpd"]

    df=df.set_index("model",drop=True)
    # df=df[["w1m","w1p","w1efp","fpnd","cov","mmd","pmm","pmp","pme","model"]]

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
    print(df)
    def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
    count_parameters(model.gen_net)


    tex=""
    for p in ["t"]:
        temp=df.loc[["MDMA","EPiC","IN"],:]

        for col in df.drop("model",axis=1).columns:
            if col not in ["w1m","w1p","w1efp","fpd","kpd" ]:
                continue
            try:
                temp_index=temp[col].astype(str).str.replace("$","").str.split("\\").str[0].astype(float)
                mins=temp_index==temp_index.drop("IN",0).min()
            except:
                mins=0
            temp.loc[mins,col]="$\mathbf{"+temp.loc[mins,col].astype(str).str.replace("$","")+"}$"
        temp=temp[["model","w1m","w1p","w1efp","kpd","fpd"]]
        temp.columns=["model","$W_1^M (\times 10^{3})$","$W_1^P (\times 10^{3})$","$W_1^{EFP}(\times 10^{5})$","$KPD(\times 10^{4})$","$FPD(\times 10^{4})$"]
        text=temp.to_latex(index=False,escape=False)
        parton="Gluon" if p=="g" else "Light Quark" if p=="q" else "Top Quark"
        tex+="\multirow{3}{*}{"+parton+"} & "+text.split("KPD \\\\")[1].split("\\bottomrule")[0].replace("\\\\","\\\\&").replace("\\midrule","").replace("  ","")[:-2]+"\cline{1-8}"
        tex+="\n"
    print(tex)


def setup_model(config,data_module=None,model=False):
    if not model:
        model = MDMA(**config)
    if config["dataset"]=="calo":
        model.bins=[600,config["num_z"],config["num_alpha"],config["num_R"]]
        model.num_z, model.num_alpha, model.num_R = config["num_z"], config["num_alpha"], config["num_R"]
        model.E_loss=config["E_loss"]
        model.lambda_gp=config["lambda_gp"]
        model.lambda_response=config["lambda_response"]
        model.min_weighted_w1p=0.1
        model.min_w1p=0.1
        model.minE=0.01
        model.scaler = data_module.scaler
        model.n_dim = 4
        model.power_lambda = model.scaler.transfs[0].lambdas_[0]
        model.mean = model.scaler.transfs[0]._scaler.mean_[0]
        model.scale = model.scaler.transfs[0]._scaler.scale_[0]
        model.pos_mean = torch.tensor(model.scaler.transfs[1].steps[2][1].mean_).cuda()
        model.pos_scale = torch.tensor(model.scaler.transfs[1].steps[2][1].scale_).cuda()
        model.pos_max_scale = torch.tensor(model.scaler.transfs[1].steps[0][1].scale_).cuda()
        model.pos_min = torch.tensor(model.scaler.transfs[1].steps[0][1].min_).cuda()
        if config["dataset"]=="calo":
                    model.power_lambda=model.scaler.transfs[0].lambdas_[0]
                    model.mean=model.scaler.transfs[0]._scaler.mean_[0]
                    model.scale=model.scaler.transfs[0]._scaler.scale_[0]
    else:
        model.bins=[100,100,100]
        model.n_dim = 3
        model.scaler=data_module.scaler
        model.w1m_best=0.01
        model.min_pt=data_module.min_pt
        model.max_pt=data_module.max_pt
    model.i=0
    model.loss = losses.hinge if config["gan"] == "hinge" else losses.wasserstein if config["gan"] == "wgan" else losses.least_squares
    model.gp = config["gp"]
    model.d_loss_mean=None
    model.g_loss_mean=None
    if config["dataset"]=="jet":
        model.scaled_mins=torch.tensor(data_module.mins)
        model.scaled_maxs=torch.tensor(data_module.maxs)
    else:
        model.scaled_mins=torch.zeros(4).cuda()
        model.scaled_maxs=torch.tensor([1e9]+model.bins[1:]).cuda()
    model.swa=False
    return model

def jetnet_eval(model,data_module):
    with open("kde/" +"t" + "_kde.pkl", "rb") as f:
        kde = pickle.load(f)
    n=5
    results=pd.DataFrame()
    true=data_module.real_test
    val=data_module.test_set
    epic= np.load("/home/kaechben/EPiC-GAN/{}_epic.npy".format("t"))[..., [1, 2, 0]]

    with torch.no_grad():
        kde_sample = kde.resample( 50000 + 10000).T  # account for cases were kde_sample is not in [1,150]
        # z = torch.normal(torch.zeros((250000, 150,3), device="cpu"), torch.ones((250000, 150,3), device="cpu"))
        z = torch.normal(torch.zeros((50000, 150,3), device="cpu"), torch.ones((50000, 150,3), device="cpu"))
        n_sample = np.rint(kde_sample)
        n_sample = torch.tensor(n_sample[(n_sample >= 1) & (n_sample <= 150)]).cuda().long()
        indices = torch.arange(150, device="cuda")
        mask = indices.view(1, -1) < torch.tensor(n_sample).view(-1, 1)
        mask = ~mask.bool()[: len(z)]
        z[mask] = 0
        model.gen_net.cuda()
        cond=(n_sample/model.avg_n).unsqueeze(1).unsqueeze(1)[: len(z)]
        start = time.time()
        # fake = torch.cat([model.gen_net(z[i * 50000 : (i + 1) * 50000].cuda(), mask=mask[i * 50000 : (i + 1) * 50000].bool().cuda(),cond=cond[i * 50000 : (i + 1) * 50000]).cpu() for i in range(n)], dim=0)
        fake=model.gen_net(z.cuda(), mask=mask.bool().cuda(),cond=cond).cpu()
        fake[:,:,:] = model.relu(fake[:, :, :] - model.mins.cpu()) + model.mins.cpu()
        fake[:,:,:] = -model.relu(model.maxs.cpu() - fake[:, :, :]) + model.maxs.cpu()
        fake[mask] = 0  # set the masked values to zero
        fake_scaled = model.scaler.inverse_transform(fake).float()
        fake_scaled[mask]=0
        print("time pro jet {}".format((time.time() - start) / len(fake)))
        for temp,name in zip([fake_scaled,val,epic],["MDMA","IN","EPiC"]):

            w1m_ = w1m(temp[:, :, :3], true[:, :, :3], num_eval_samples=len(true),num_batches=16)

            kpd_real = get_fpd_kpd_jet_features(true, efp_jobs=20)
            kpd_fake = get_fpd_kpd_jet_features(temp[: len(true),:,:3], efp_jobs=20)
            kpd_ = kpd(kpd_real, kpd_fake)
            fpd_ = fpd(kpd_real, kpd_fake, min_samples=10000, max_samples=len(true))
            # kpd_=[np.array([0]),np.array([0.1])]
            # fpd_=[np.array([0]),np.array([0.1])]
            data_ms = mass(true).numpy()
            i = 0
            w_dist_list = []
            # for _ in range(n):
            #     gen_ms = mass(fake_scaled[i : i + len(true)]).numpy()
            #     i += len(true)
            #     w_dist_ms = wasserstein_distance(data_ms, gen_ms)
            #     w_dist_list.append(w_dist_ms)
            # w1m_2 = np.mean(np.array(w_dist_list))
            # w1m_2std = np.std(np.array(w_dist_list))

            w1efp_ = w1efp(temp[: len(true)], true[:, :, :3], num_eval_samples=len(true), efp_jobs=20)
            # w1efp_=[np.array([0]),np.array([0.1])]
            w1p_ = w1p(temp[: len(true),:,:3], true[:, :, :3], num_eval_samples=len(true))

            results_temp = {"name": ["t"], "model": [name], "w1m": [w1m_], "w1p": [w1p_],  "w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_]}#"w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_]
            results=pd.concat((results,pd.DataFrame(results_temp).set_index("name",drop=True)),axis=0)
        return results

if len(sys.argv)>1:
    NAME=sys.argv[1]
else:
    NAME="jet"
if NAME=="calo":
     from dataloader_calo import PointCloudDataloader
     ckpt="/gpfs/dust/maxwell/user/kaechben/calochallenge/MDMA_calo/83n3r9n7/checkpoints/epoch=294-w1p=0.00045-weighted_w1p=0.00057.ckpt"
else:
    from dataloader_jetnet import PointCloudDataloader
    name="jet"
    ckpt="./jet.ckpt"
    config=yaml.load(open("default_{}.yaml".format(NAME)),Loader=yaml.FullLoader)

torch.set_float32_matmul_precision('medium' )
data_module = PointCloudDataloader(**config)
data_module.setup("train")
state_dict=torch.load(ckpt,map_location="cpu")
config.update(**state_dict["hyper_parameters"])
config["L2"]=False
model =MDMA.load_from_checkpoint(ckpt,**config)
# model=MDMA.load_from_checkpoint(ckpt,strict=True)
model=setup_model(config,data_module,model)
model.swa=config["start_swa"]
model.load_datamodule(data_module)
#loop once through dataloader to find mins and maxs to clamp during training
data_module = PointCloudDataloader(**config)
data_module.setup("fit")
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
model.gen_net.avg_n=torch.cat(n,dim=0).float().cuda().mean()
model.dis_net.avg_n=torch.cat(n,dim=0).float().cuda().mean()
model.scaled_mins=data_module.mins
model.scaled_maxs=data_module.maxs
if NAME=="calo":
     pass
else:
    results=jetnet_eval(model,data_module)
    print_table(results)
#trainer.
