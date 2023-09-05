import datetime
import os
import sys
import time
import traceback

import wandb



import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF
import torch
import losses
# from pytorch_lightning.plugins.environments import SLURMEnvironment
from helpers import *
from fit import MDMA
from tqdm import tqdm
# from plotting import plotting

from preprocess_new import ScalerBaseNew,DQLinear,LogitTransformer, SqrtTransformer,Cart
from preprocess_new import SqrtTransformer as LogTransformer
from models import Gen, Disc
from torch.optim.swa_utils import AveragedModel, SWALR
# from comet_ml import Experiment
from torch.nn.utils.weight_norm import WeightNorm
import yaml
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

def setup_model(config,data_module=None,model=False):
    if not model:
        model = MDMA(**config)
    if config["name"]=="calo":
        model.bins=[600,config.num_z,config.num_alpha,config.num_R]
        model.num_z, model.num_alpha, model.num_R = config.num_z, config.num_alpha, config.num_R
        model.E_loss=config["E_loss"]
        model.pos_loss=config["pos_loss"]
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
        if model.name=="calo":
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
    model.loss = losses.hinge if config["gan"] == "hinge" else losses.wasserstein if config["gan"] == "wasserstein" else losses.least_squares
    model.gp = config["gp"]
    model.d_loss_mean=None
    model.g_loss_mean=None
    model.scaled_mins=data_module.mins
    model.scaled_maxs=data_module.maxs
    model.swa=False
    return model

def train(config, logger,data_module,trainer,ckpt=False):
    torch.set_float32_matmul_precision('medium' )
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - ckpt: path to checkpoint if used
    # Callbacks to use during the training, we  checkpoint our models
    print("This is run: ", logger.experiment.name)
    print("config:", config)
    if not ckpt:
        model = setup_model(config,data_module)
    else:
        state_dict=torch.load(ckpt,map_location="cpu")
        config.update(**state_dict["hyper_parameters"])
        config["lr"]*=0.01
        model=MDMA(**config).load_from_checkpoint(ckpt,**config,strict=True)

        model=setup_model(config,data_module,model)
        model.swa=config["start_swa"]
        if model.swa:
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
    model.gen_net.avg_n=torch.cat(n,dim=0).float().cuda().mean()
    model.dis_net.avg_n=torch.cat(n,dim=0).float().cuda().mean()
    if model.swa:
        model.gen_net_averaged._modules["module"].avg_n=model.gen_net.avg_n
    if ckpt :
        trainer.fit(model,datamodule=data_module)
    else:
        trainer.fit(model,datamodule=data_module,)

if __name__ == "__main__":

    config=yaml.load(open("default_jet.yaml"),Loader=yaml.FullLoader)
    config["dataset"]="jetnet"
    #set up WandB logger
    logger = WandbLogger(
        save_dir="/gpfs/dust/maxwell/user/{}/calochallenge".format(os.environ["USER"]),
        sync_tensorboard=False,
        project="MDMA_"+"calo" if config["name"]=="calochallenge" else "jet")
    # update config with hyperparameters from sweep
    logger.experiment.log_code(".")
    ckpt=config["ckpt"]
    if len(logger.experiment.config.keys()) > 0:
        ckpt=None
        config.update(**logger.experiment.config)
    if config["dataset"]=="calo":
        from dataloader_calo import PointCloudDataloader
    else:
        from dataloader_jetnet import PointCloudDataloader
    data_module = PointCloudDataloader(**config)
    data_module.setup("train")
    callbacks =[LearningRateMonitor(),ModelCheckpoint(monitor="w1m", save_top_k=2, mode="min",filename="{epoch}-{w1m:.5f}-{fpd:.5f}",every_n_epochs=1,),pl.callbacks.LearningRateMonitor(logging_interval="step")]
        # ModelCheckpoint(monitor="weighted_z", save_top_k=1, mode="min",filename="{epoch}-w1p_{w1p:.5f}-weighted_w1p_{weighted_w1p:.5f}-weighted_z_{weighted_z:.5f}",every_n_epochs=1,),
        # ModelCheckpoint(monitor="weighted_w1p", save_top_k=1, mode="min",filename="min_weighted_{epoch}-{w1p:.5f}-{weighted_w1p:.5f}",every_n_epochs=1,),pl.callbacks.LearningRateMonitor(logging_interval="step")
        # ,ModelCheckpoint(monitor= 'features_E', mode= 'min',save_top_k=1,filename="minE_{features_E:.7f}-{w1p:.5f}", every_n_train_steps= 0, every_n_epochs= 1, train_time_interval= None)]
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=300,
        max_epochs=20000,
        callbacks=callbacks,
        # progress_bar_refresh_rate=0,
        val_check_interval=10 if ckpt and config["start_swa"] else 10000,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=2,
        enable_progress_bar=False,
        default_root_dir="/gpfs/dust/maxwell/user/{}/{}".format(os.environ["USER"],config["dataset"]),
        # reload_dataloaders_every_n_epochs=0,#,config["val_check"] if not config["smart_batching"] else 0,
        #profiler="pytorch"
    )
    train(config,logger=logger,data_module=data_module,trainer=trainer,ckpt=ckpt)  # ckpt=ckpt,load_ckpt=ckptroot=root,