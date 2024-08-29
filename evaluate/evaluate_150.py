import torch
import os
import sys

# Get the current path of the notebook
notebook_path = os.getcwd()
import time
# Construct the path to the directory containing 'fit'

os.chdir(notebook_path)
import sys
# Add this parent directory to the system path
sys.path.insert(0, notebook_path)
from fit.fit_jet_fm import FM
import pytorch_lightning as pl
import torch

from callbacks.ema import EMA
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from utils.helpers import get_hists, mass
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
import ast
import pandas as pd
from jetnet.evaluation import w1m, w1efp, w1p, get_fpd_kpd_jet_features, fpd, kpd
from utils.helpers import plotting_thesis
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




def one_digit_formatter(x, pos):
    return f'{x:.1e}'



FONTSIZE=20

def calculate_data_bounds(dataloader, n_dim):
    """
    Calculates the minimum and maximum values across the dimensions from the dataloader.

    :param dataloader: The dataloader to process.
    :param n_dim: Number of dimensions.
    :return: Tuple of minimum values, maximum values, and count of non-masked data points.
    """
    mins = torch.ones(n_dim).unsqueeze(0)
    maxs = torch.ones(n_dim).unsqueeze(0)
    n_counts = []

    for data in dataloader:
        non_masked_data = data[0][~data[1]]
        mins = torch.min(torch.cat((mins, non_masked_data.min(0, keepdim=True)[0]), dim=0), dim=0)[0].unsqueeze(0)
        maxs = torch.max(torch.cat((maxs, non_masked_data.max(0, keepdim=True)[0]), dim=0), dim=0)[0].unsqueeze(0)
        n_counts.append((~data[1]).sum(1))

    return mins, maxs, n_counts
def make_plots(model_name,data_module, disco=False):
    import os

    if model_name!="IN" and model_name!="EPiC-GAN" and model_name!="EPiC-FM":
        ckptdir = "/ckpts/"
        ckpt = "{}.ckpt".format(model_name)
        # ckpt = "t_{}.ckpt".format(model_name)
        ckpt = f"/home/kaechben/MDMACalo/ckpts/{model_name}.ckpt"
        # Load state dictionary from checkpoint
        state_dict = torch.load(ckpt)
        import yaml
        config = state_dict["hyper_parameters"]
        print(config)
        config["ckpt"]=None
        config["sampler"]=False
        config["boxcox"]=boxcox
        print("boxcox",boxcox)
        model_name=model_name.replace("_ot","")
        # Choose the model class based on the model name

        if model_name.find("fm")==-1:
                from fit.fit import MDMA
                model=MDMA.load_from_checkpoint(ckpt,strict=False,boxcox=boxcox)
                config["model"]="MDMA"

                model.gen_net.avg_n = data_module.avg_n
                model.mins=data_module.std_mins.to("cuda")
                model.maxs=data_module.std_maxs.to("cuda")
        else:


            model=FM.load_from_checkpoint(ckpt,boxcox=boxcox,strict=False)
            model.mins=data_module.std_mins.to("cuda")
            model.maxs=data_module.std_maxs.to("cuda")
        model.w1m_best = 0.01
        # Initialize data module and set up model
        if boxcox:
            model.scaler = data_module.scaler[0].to("cuda")
            model.pt_scaler=data_module.scaler[1]
        else:
            model.scaler=data_module.scaler.to("cuda")
        model.scaled_mins = torch.tensor(data_module.mins).cuda()[:3]
        model.scaled_maxs = torch.tensor(data_module.maxs).cuda()[:3]
        model.min_pt = data_module.min_pt
        model.max_pt = data_module.max_pt
        mins, maxs, n_counts = calculate_data_bounds(data_module.train_dataloader(), config["n_dim"])
        model.maxs = maxs.cuda().reshape(-1)
        model.mins = mins.cuda().reshape(-1)
        model.batch=[]
        model.n_dim=3
        model.masks=[]
        model.fake=[]
        model.conds=[]
        model=model.cuda()
        model.name="jet"
        model.load_datamodule(data_module)
        trainer = pl.Trainer(
                devices=1,
                precision=32,
                logger=None,
                accelerator="gpu",
                callbacks=[EMA(0.999)] if boxcox else None,
                enable_progress_bar=False,
                default_root_dir="/gpfs/dust/maxwell/user/{}/{}".format(
                    os.environ["USER"], config["dataset"]
                ),
            )
        start=time.time()
        true=data_module.real_test[...,:-1]
        with torch.no_grad():
            trainer.test(model=model,dataloaders=data_module.test_dataloader())
        fake= torch.cat([torch.nn.functional.pad(batch, (0, 0, 0, model.hparams.n_part - batch.size(1))) for batch in model.fake],dim=0) if model.hparams.max else torch.cat(model.fake)
        sorted_indices = torch.argsort(fake[:,:,2], dim=1, descending=True)
        fake = torch.gather(fake, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, fake.shape[2]))
        mins=true.reshape(-1, 3).min()

        maxs=true.reshape(-1, 3).max()
        fake = torch.clamp(fake, min=mins, max=maxs*0.99)
        true = torch.clamp(true, min=mins, max=maxs*0.99)
        with open("/beegfs/desy/user/kaechben/thesis/eval_jetnet150/{}_jets.npy".format(model_name), "wb") as f:
            np.save(f,fake.cpu().numpy())
        total=(time.time()-start)/500000.
        if model_name.find("mdma_jet")!=-1:
            params=sum(p.numel() for p in model.gen_net.parameters() if p.requires_grad)
        else:
            params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:

        if model_name=="IN":
            fake=torch.from_numpy(data_module.train[...,:-1])
            total=0
            params=0
        elif model_name=="EPiC-FM":

            total=6075*1e-6
            params=561330
        else:
            params=424850
            total=62.5*1e-6


        model=None
    #plotter.plot_corr(true.numpy(), fake.numpy(), model_name, disco=disco,leg=-1)
    # true = torch.cat([torch.nn.functional.pad(batch, (0, 0, 0, model.hparams.n_part - batch.size(1))) for batch in model.batch],dim=0) if model.hparams.max else torch.cat(model.batch)

    return model,total,params

def calc_metrics(true,train,time_dict,param_dict,models):
    groups={"150":[]}

    for i,model_name in enumerate(models):#mdma_jet
        if model_name=="IN":
            fake=train
        else:
            fake=torch.from_numpy(np.load("/beegfs/desy/user/kaechben/thesis/eval_jetnet150/{}_jets.npy".format(model_name)))
        if model_name.find("EPiC-GAN")>-1:
            fake=fake[:,:,[1,2,0]]
        assert len(fake)==len(true)
        # model_name=model_name.replace("_std","")
        m_f, m_t = mass(fake), mass(true)
        # Apply clamping based on quantiles
        mins = torch.quantile(true.reshape(-1, 3), 0.0, dim=0)
        maxs = torch.quantile(true.reshape(-1, 3), 1, dim=0)
        mins[0]=-1
        maxs[0]=1
        fake = torch.clamp(fake, min=mins, max=maxs)
        true = torch.clamp(true, min=mins, max=maxs)
        m_f = torch.clamp(m_f, min=torch.quantile(m_t,0), max=torch.quantile(m_t,1))
        m_t = torch.clamp(m_t, min=torch.quantile(m_t,0), max=torch.quantile(m_t,1))
        w1m_score = w1m(fake, true)
        real_fpd = get_fpd_kpd_jet_features(true[:50000], efp_jobs=40)
        w1efp_score = w1efp(fake[:50000], true)
        w1p_score = w1p(fake[:50000], true)
        fake_fpd = get_fpd_kpd_jet_features(fake[:50000], efp_jobs=40)
        fpd_ = fpd(real_fpd, fake_fpd, )
        kpd_ = kpd(real_fpd, fake_fpd, )
        metrics = {"name": ["t"], "model": [model_name], "w1m": [w1m_score], "w1p": [w1p_score],  "w1efp": [w1efp_score], "kpd": [kpd_], "fpd": [fpd_],"time":[time_dict[model_name]],"parameters":[ param_dict[model_name]]}#"w1efp": [w1efp_], "kpd": [kpd_], "fpd": [fpd_]
        # Prepare histograms
        mins=torch.cat((mins,m_t.min().unsqueeze(0)))
        maxs=torch.cat((maxs,m_t.max().unsqueeze(0)))
        mins[0]=-1
        maxs[0]=1
        hists=get_hists([50,50,50,50],mins-0.01,maxs+0.01,calo=False)

        # Fill histograms
        for var in range(3):
            hists["hists_real"][var].fill(true.reshape(-1, 3)[(true.reshape(-1, 3) != 0).all(1)][:, var].cpu().numpy())
            hists["hists_fake"][var].fill(fake.reshape(-1, 3)[(fake.reshape(-1, 3) != 0).all(1)][:, var].cpu().numpy())
        hists["hists_real"][3].fill(m_t.cpu().numpy())
        hists["hists_fake"][3].fill(m_f.cpu().numpy())

        # Plotting
        plotter = plotting_thesis()
        replace_dict={"mdma_jet":"MDMA-GAN","mdma_jet":"MDMA-GAN","mdma_fm_jet":"MDMA-Flow","IN":"IN","EPiC-GAN":"EPiC-GAN","EPiC-FM":"EPiC-FM"}
        save_name=model_name

        if save_name.find("mdma")>-1:
            save_name=model_name



            groups["150"].append(plotter.plot_ratio_multiple(hists["hists_real"],hists["hists_fake"],weighted=False,leg=2,model_name=save_name,legend_name=replace_dict[model_name.replace("_std","")],group_name="150",groups=groups,n_part=150,boxcox=boxcox))

        df=pd.DataFrame(metrics)
        print(metrics)
        if i==0:
            results=df
        else:
            results=pd.concat([results,df],axis=0)
    return results
def create_mask(n, size=30):
    # Ensure n is a 1D tensor
    n = n.flatten()

    # Create a range tensor [0, 1, 2, ..., size-1]
    range_tensor = torch.arange(size).unsqueeze(0)

    # Compare range_tensor with n to create the mask
    mask = range_tensor >= n.unsqueeze(1)

    return mask


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

    replace_dict={"mdma_jet":"MDMA-GAN","mdma_jet":"MDMA-GAN","mdma_fm_jet":"MDMA-Flow","IN":"IN","EPiC-GAN":"EPiC-GAN","EPiC-FM":"EPiC-FM"}
    df.loc[:,"model"]=df["model"].apply(lambda x:replace_dict[x.replace("_std","")])
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

    order=["MDMA-GAN","MDMA-Flow","EPiC-GAN","EPiC-FM","IN"]
    print(df)
    df=df.loc[order,:]

    # def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print (count_parameters(model.gen_net))
    # df=df.set_index("model",drop=True)
    tex=""
    for p in ["t"]:
        temp=df
        for col in df.columns:
            if col not in ["w1m","w1p","w1efp","fpd","kpd","time","parameters"]:
                continue
            temp_index=temp[col].astype(str).str.replace("$","").str.replace("k","").str.split("\\").str[0].astype(float)
            mins=temp_index==temp_index.drop("IN").min()

            temp.loc[mins,col]="$\mathbf{"+temp.loc[mins,col].astype(str).str.replace("$","")+"}$"
        temp=temp.reset_index()[["model","w1m","w1p","w1efp","kpd","fpd","time","parameters"]]
        temp.columns=["model","$W_1^M (\times 10^{3})$","$W_1^P (\times 10^{3})$","$W_1^{EFP}(\times 10^{5})$","$\texttt{KPD}(\times 10^{4})$","$\texttt{FPD}$","Time $\mu s$", "#Parameters"]
        text=temp.to_latex(index=False,escape=False)
        parton="Gluon" if p=="g" else "Light Quark" if p=="q" else "Top Quark"
        tex+="\multirow{4}{*}{"+parton+"} & "+text.split("Parameters")[1].split("\\bottomrule")[0][4:].replace("\\\\","\\\\&").replace("\\midrule","").replace("  ","").replace("\\midrule","").replace("  ","")[:-2]+"\cline{1-9}"
        tex+="\n"
    print(tex)
import json
#,"mdma_fm_jet","EPiC-FM","IN","EPiC-GAN"
boxcox=True
if not boxcox:
    models=["mdma_jet_std","mdma_fm_jet_std","EPiC-FM","IN","EPiC-GAN"]
else:
    models=["mdma_jet","mdma_fm_jet","EPiC-FM","IN","EPiC-GAN"]
print("boxcox:",boxcox)
if boxcox:
    from utils.dataloader_jetnet import PointCloudDataloader
else:
    from utils.dataloader_jetnet_std import PointCloudDataloader
data_dir="/beegfs/desy/user/kaechben/thesis/jetnet150/data_generated/"
os.makedirs(data_dir,exist_ok=True)
data_module = PointCloudDataloader(batch_size=5000,n_part=150,n_dim=3,sampler=False,parton="t")
data_module.setup("fit")
torch.set_float32_matmul_precision("medium")
time_dict={}
param_dict={}
true=data_module.real_test[...,:-1]
train=torch.from_numpy(data_module.train[...,:-1])[:50000]

with open('params.json', 'r') as json_file:
    param_dict = json.load(json_file)
with open('times.json', 'r') as json_file:
    time_dict = json.load(json_file)
if False:


    for i,model_name in enumerate(models):#
        model,total,params=make_plots(model_name,data_module=data_module)
        time_dict[model_name]=total
        param_dict[model_name]=params
    with open('params.json', 'w') as json_file:
        json.dump(param_dict, json_file)
    with open('times.json', 'w') as json_file:
        json.dump(time_dict, json_file)

models=[m.replace("_ot","") for m in models]
results=calc_metrics(true,train,time_dict,param_dict,models)

results.to_csv("results150.csv")
results=pd.read_csv("results150.csv")
print_table(results)