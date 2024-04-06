import torch
import os
# Get the current path of the notebook
notebook_path = os.getcwd()

from fit.fit import MDMA
from fit.fit_jet_fm import FM
import pytorch_lightning as pl
import torch

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from utils.helpers import get_hists, mass, plotting_thesis
from utils.dataloader_calo import PointCloudDataloader
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from utils.preprocess import DQ, Cart, DQLinear, LogitTransformer, ScalerBase
from utils.preprocess import SqrtTransformer
from utils.preprocess import SqrtTransformer as LogTransformer
import time
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




def setup_model_with_data(model, data_module, config):
    """
    Sets up the model with the data module and configuration parameters.

    :param model: The model to be set up.
    :param data_module: The data module used for training and validation.
    :param config: Configuration dictionary.
    """
    model.bins = config["bins"]

    model.num_z, model.num_alpha, model.num_R = model.bins[1:]
    model.E_loss = config["E_loss"]
    model.lambda_gp = config["lambda_gp"]
    model.lambda_response = config["lambda_response"]
    model.min_weighted_w1p = 0.1
    model.min_w1p = 0.1
    model.minE = 0.01
    model.scaler = data_module.scaler
    model.n_dim = 4
    model.power_lambda = model.scaler.transfs[0].lambdas_[0]
    model.mean = model.scaler.transfs[0]._scaler.mean_[0]
    model.scale = model.scaler.transfs[0]._scaler.scale_[0]
    model.pos_mean = torch.tensor(model.scaler.transfs[1].steps[2][1].mean_).cuda()
    model.pos_scale = torch.tensor(
        model.scaler.transfs[1].steps[2][1].scale_
    ).cuda()
    model.pos_max_scale = torch.tensor(
        model.scaler.transfs[1].steps[0][1].scale_
    ).cuda()
    model.pos_min = torch.tensor(model.scaler.transfs[1].steps[0][1].min_).cuda()
    model.power_lambda = model.scaler.transfs[0].lambdas_[0]
    model.mean = model.scaler.transfs[0]._scaler.mean_[0]
    model.scale = model.scaler.transfs[0]._scaler.scale_[0]

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
    min_response,max_response=torch.tensor([100]).cuda(),torch.tensor([0]).cuda()
    for data in dataloader:
        non_masked_data = data[0][~data[1]]
        response=data[0][:,:,0].sum(1).reshape(-1)/((data[2][:,:,0].reshape(-1)+10).exp())
        min_response=torch.min(min_response,torch.tensor([response.min().item()]).cuda())
        max_response=torch.max(max_response,torch.tensor([response.max().item()]).cuda())
        mins = torch.min(torch.cat((mins, non_masked_data.min(0, keepdim=True)[0]), dim=0), dim=0)[0].unsqueeze(0)
        maxs = torch.max(torch.cat((maxs, non_masked_data.max(0, keepdim=True)[0]), dim=0), dim=0)[0].unsqueeze(0)
        n_counts.append((~data[1]).sum(1))

    return mins, maxs, n_counts, min_response, max_response


def one_digit_formatter(x, pos):
    return f'{x:.1e}'

from utils.dataloader_calo import PointCloudDataloader

FONTSIZE=20

def make_plots(model_name,data_module, disco=False,raw=False,groups={}):
    import os
    ckptdir = "./ckpts/"

    if model_name.find("fm") >-1:
        ckpt = "{}_small.ckpt".format(model_name)
    else:
        ckpt = "{}.ckpt".format(model_name)

    # ckpt = "t_{}.ckpt".format(model_name)
    ckpt = ckptdir + ckpt
    # Load state dictionary from checkpoint
    state_dict = torch.load(ckpt)
    import yaml

    # with open(parent_path+"/hparams/default_{}.yaml".format(model_name.replace("_slow","")), 'r') as file:
    #     default = yaml.safe_load(file)

    config = state_dict["hyper_parameters"]
    # default.update(**config)

# Load the model

    torch.set_float32_matmul_precision("medium")
    config["scaler_path"]="/home/kaechben/MDMACalo/"
    # Initialize data module and set up model
    config["batch_size"]=100


    config["max"]=False
    model = MDMA.load_from_checkpoint(ckpt,raw=raw) if config["model"]!="FM" else FM.load_from_checkpoint(ckpt,raw=raw)
    model.eval_metrics=False

# Assuming `model` is defined elsewhere in your code

    setup_model_with_data(model, data_module, config)

    mins,maxs,n,min_response,max_response=calculate_data_bounds(data_module.val_dataloader(), model.n_dim)

    # n_max=n.max()
    # n=n.float().cpu()
    # n_kde=fit_kde(n)
    # n=sample_kde(len(n),n_kde)

    model.eval_metrics=False
    model.batch=[]
    model.masks=[]
    model.fake=[]
    model.conds=[]
    model=model.cuda()
    if config["model"]=="FM":
        pass
    else:
        model.gen_net.avg_n=data_module.avg_n
    model.mins=mins.cuda()
    model.maxs=maxs.cuda()
    model.scaled_mins=mins.cuda()
    model.scaled_maxs=maxs.cuda()
    model.load_datamodule(data_module)
    model.hparams.raw=raw
    hists=get_hists(config["bins"],mins*1.1,maxs*1.1,calo=True,min_response=min_response,max_response=max_response)
    model.times=[]
    trainer = pl.Trainer(
                devices=1,
                precision=32,
                accelerator="gpu",
                logger=False,
                enable_progress_bar=False,
                default_root_dir="/gpfs/dust/maxwell/user/{}/{}".format(
                    os.environ["USER"], config["dataset"]
                ),
            )
    with torch.no_grad():
        i=0
        trainer.test(model=model, dataloaders=data_module.test_dataloader())
        print(len(model.batch))


    # Plotting
    total = np.mean(np.array(model.times))
    try:
        if "gen_net" in vars(model.vars):
            params=sum(p.numel() for p in model.gen_net.parameters() if p.requires_grad)
        else:
            params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    except:

        params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    params
    plotter = plotting_thesis()
    if raw:
        model_name+="_raw"
    groups["unweighted"].append(plotter.plot_calo_multiple(model.hists_real, model.hists_fake, weighted=False, leg=2-int(raw), model_name=model_name,legend_name="MDMA-GAN" if model_name.find("fm") == -1 else "MDMA-Flow",groups=groups,group_name="unweighted",raw=raw))

    groups["weighted"].append(plotter.plot_calo_multiple(model.weighted_hists_real, model.weighted_hists_fake, weighted=True, leg=2-int(raw), model_name=model_name,legend_name="MDMA-GAN" if model_name.find("fm") == -1 else "MDMA-Flow",groups=groups,group_name="weighted",raw=raw))
    print(1,groups["responses"])
    fig=plotter.plot_response_multiple(model.response_real,model.response_fake,model_name=model_name,legend_name="MDMA-GAN" if model_name.find("fm") == -1 else "MDMA-Flow",groups=groups,group_name="responses",raw=raw)
    print(fig)
    groups["responses"].append(fig)
    print(2,groups["responses"])
    print("saved plots",model_name)
    if model.hparams.dataset=="calo":
        torch.save(model.fake,"/beegfs/desy/user/kaechben/data_generated/calochallenge_{}_{}.pt".format(model_name,"big" if model.hparams.bins[1]==50 else "middle"))
        torch.save(model.batch,"/beegfs/desy/user/kaechben/data_generated/calochallenge_reals_{}_{}.pt".format(model_name,"big" if model.hparams.bins[1]==50 else "middle"))
        torch.save(model.masks,"/beegfs/desy/user/kaechben/data_generated/calochallenge_masks_{}_{}.pt".format(model_name,"big" if model.hparams.bins[1]==50 else "middle"))
        torch.save(model.conds,"/beegfs/desy/user/kaechben/data_generated/calochallenge_conds_{}_{}.pt".format(model_name,"big" if model.hparams.bins[1]==50 else "middle"))
        torch.save(model.times,"/beegfs/desy/user/kaechben/data_generated/calochallenge_times_{}_{}.pt".format(model_name,"big" if model.hparams.bins[1]==50 else "middle"))
        #plotter.plot_corr(true.numpy(), fake.numpy(), model_name, disco=disco,leg=-1)
        # true = torch.cat([torch.nn.functional.pad(batch, (0, 0, 0, model.hparams.n_part - batch.size(1))) for batch in model.batch],dim=0) if model.hparams.max else torch.cat(model.batch)

    return model,total,params,groups


def create_mask(n, size=30):
    # Ensure n is a 1D tensor
    n = n.flatten()

    # Create a range tensor [0, 1, 2, ..., size-1]
    range_tensor = torch.arange(size).unsqueeze(0)

    # Compare range_tensor with n to create the mask
    mask = range_tensor >= n.unsqueeze(1)

    return mask




torch.set_float32_matmul_precision("medium")
import json
import yaml
raw=True
groups={"weighted":[],"unweighted":[],"responses":[]}
config = config = yaml.load(
        open("hparams/default_calo.yaml"), Loader=yaml.FullLoader
    )
data_module = PointCloudDataloader(**config)
data_module.setup("train")
with open('params_calo.json', 'r') as json_file:
    param_dict = json.load(json_file)
with open('times_calo.json', 'r') as json_file:
    time_dict = json.load(json_file)
if True:
    time_dict={}
    param_dict={}
    for i,model_name in enumerate(["mdma_calo","mdma_fm_calo"]):#"mdma_fm_calo",
        model,total,params,groups=make_plots(model_name,data_module,raw=raw, groups=groups)
        time_dict[model_name]=total
        param_dict[model_name]=params
with open('params_calo.json', 'w') as json_file:
    json.dump(param_dict, json_file)
with open('times_calo.json', 'w') as json_file:
    json.dump(time_dict, json_file)


