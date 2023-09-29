import datetime
import os
import sys
import time
import traceback

import utils.losses as losses
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
import yaml
from fit.fit import MDMA
from fit.fit_nf import NF
from fit.fit_tnf import TNF
from fit.fit_pnf import PNF
from utils.helpers import *
from models.models import Disc, Gen
from preprocess_new import Cart, DQLinear, LogitTransformer, ScalerBaseNew
from preprocess_new import SqrtTransformer
from preprocess_new import SqrtTransformer as LogTransformer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import (CometLogger, TensorBoardLogger,
                                       WandbLogger)
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF
from torch.nn.utils.weight_norm import WeightNorm
from torch.optim.swa_utils import SWALR, AveragedModel
from tqdm import tqdm

if len(sys.argv)>1:
    NAME=sys.argv[1]
else:
    NAME="jet_tnf"
def fixmepls1(model):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                delattr(module, hook.name)
def fixmepls2(model):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook(module, None)

def lcm(a, b):
    return (a * b) // math.gcd(a, b)

def setup_model(config, data_module=None, model=False):
    """
    Set up the model object based on the provided configuration and data module.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.
        data_module (object, optional): An object containing the data module for the model. Defaults to None.
        model (bool, optional): A flag indicating whether a model object is provided. Defaults to False.

    Returns:
        object: The configured model object.
    """
    if not model and config["model"]=="MDMA":
            model = MDMA(**config)
    elif not model and config["model"]=="NF":
        model = NF(**config)
    elif not model and config["model"]=="TNF":
        model = TNF(**config)
    elif not model and config["model"]=="PNF":
        model = PNF(**config)

    if config["model"]!="NF":
            model.loss = losses.hinge if config["gan"] == "hinge" else losses.wasserstein if config["gan"] == "wgan" else losses.least_squares
            model.gp = config["gp"]
            model.d_loss_mean = None
            model.g_loss_mean = None


    if config["dataset"] == "calo":
        model.bins = [600, config["num_z"], config["num_alpha"], config["num_R"]]
        model.num_z, model.num_alpha, model.num_R = config["num_z"], config["num_alpha"], config["num_R"]
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
        model.pos_scale = torch.tensor(model.scaler.transfs[1].steps[2][1].scale_).cuda()
        model.pos_max_scale = torch.tensor(model.scaler.transfs[1].steps[0][1].scale_).cuda()
        model.pos_min = torch.tensor(model.scaler.transfs[1].steps[0][1].min_).cuda()
        model.power_lambda = model.scaler.transfs[0].lambdas_[0]
        model.mean = model.scaler.transfs[0]._scaler.mean_[0]
        model.scale = model.scaler.transfs[0]._scaler.scale_[0]
    else:
        model.bins = [100, 100, 100]
        model.n_dim = 3
        model.scaler = data_module.scaler
        model.w1m_best = 0.01
        model.min_pt = data_module.min_pt
        model.max_pt = data_module.max_pt
    model.i = 0

    if config["dataset"] == "jet":
        model.scaled_mins = torch.tensor(data_module.mins).cuda()
        model.scaled_maxs = torch.tensor(data_module.maxs).cuda()
        model.scaler.to("cuda")
    else:
        model.scaled_mins = torch.zeros(4).cuda()
        model.scaled_maxs = torch.tensor([1e9] + model.bins[1:]).cuda()
    model.swa = False
    return model

def train(config, logger,data_module,trainer,ckpt=False):
    # torch.set_float32_matmul_precision('medium' )
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - ckpt: path to checkpoint if used
    # Callbacks to use during the training, we  checkpoint our models
    print("This is run: ", logger.experiment.name)
    print("config:", config)
    swa=config["start_swa"]
    if not ckpt:
        model = setup_model(config,data_module,model=False)


    else:
        state_dict=torch.load(ckpt,map_location="cuda")
        config.update(**state_dict["hyper_parameters"])
        config["swa"]=swa
        if config["model"]=="MDMA":
            model=MDMA(**config).load_from_checkpoint(ckpt,**config,strict=True)
        elif config["model"]=="NF":
            model=NF(**config).load_from_checkpoint(ckpt,**config,strict=True)
        elif config["model"]=="TNF":
            model=TNF(**config).load_from_checkpoint(ckpt,**config,strict=True)
        model=setup_model(config,data_module,model)

        if model.swa: # this is a bit of a hack to get the SWA to work when loading it when there is a spectral/weight norm
            for param in model.parameters():
                param.data = param.data.detach()
            fixmepls1(model.gen_net)
            model.gen_net_averaged = AveragedModel(model.gen_net)
            fixmepls2(model.gen_net)
            fixmepls2(model.gen_net_averaged)

    model.load_datamodule(data_module)
    #loop once through dataloader to find mins and maxs to clamp during training
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
    if config["model"]=="MDMA":
        model.gen_net.avg_n=torch.cat(n,dim=0).float().cuda().mean()
        model.dis_net.avg_n=torch.cat(n,dim=0).float().cuda().mean()
        if model.swa:
            model.gen_net_averaged._modules["module"].avg_n=model.gen_net.avg_n
    if ckpt :
        trainer.fit(model,datamodule=data_module,ckpt_path=ckpt)
    else:
        trainer.fit(model,datamodule=data_module,)

if __name__ == "__main__":


    config=yaml.load(open("hparams/default_{}.yaml".format(NAME)),Loader=yaml.FullLoader)
    #set up WandB logger
    logger = WandbLogger(
        save_dir="/gpfs/dust/maxwell/user/{}/calochallenge".format(os.environ["USER"]),
        sync_tensorboard=False,
        project="MDMA_"+NAME)
    # update config with hyperparameters from sweep
    logger.experiment.log_code(".")
    if len(logger.experiment.config.keys()) > 0:
        ckpt=None
        config.update(**logger.experiment.config)
    if config["dataset"]=="calo":
        from utils.dataloader_calo import PointCloudDataloader
    else:
        from utils.dataloader_jetnet import PointCloudDataloader
    ckpt=config["ckpt"]
    data_module = PointCloudDataloader(**config)
    data_module.setup("train")
    if config["dataset"]=="jet":
        callbacks =[ModelCheckpoint(monitor="w1m", save_top_k=2, mode="min",filename="{epoch}-{w1m:.5f}-{fpd:.5f}",every_n_epochs=1,),pl.callbacks.LearningRateMonitor(logging_interval="step"),ModelCheckpoint(monitor="fpd", save_top_k=2, mode="min",filename="{epoch}-{w1m:.5f}-{fpd:.5f}",every_n_epochs=1,),pl.callbacks.LearningRateMonitor(logging_interval="step")]
    else:
        callbacks =[ModelCheckpoint(monitor="w1p", save_top_k=2, mode="min",filename="{epoch}-{w1p:.5f}-{weighted_w1p:.5f}",every_n_epochs=1,)]
    trainer = pl.Trainer(
        devices=1,
        precision=16 if config["amp"] else 32,
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=100,
        max_epochs=20000,
        callbacks=callbacks,
        val_check_interval=100 if ckpt and config["start_swa"] else 2000 if config["model"]=="NF" else 10000,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=2,
        enable_progress_bar=False,
        default_root_dir="/gpfs/dust/maxwell/user/{}/{}".format(os.environ["USER"],config["dataset"]),
    )
    train(config,logger=logger,data_module=data_module,trainer=trainer,ckpt=ckpt)