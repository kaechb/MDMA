import datetime
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF
from torch.nn.utils.weight_norm import WeightNorm
from tqdm import tqdm

import utils.losses as losses
import wandb
from callbacks import EMA, EMAModelCheckpoint
from fit.fit import MDMA
from fit.fit_jet_fm import FM
from fit.fit_nf import NF
from fit.fit_pnf import PNF
from fit.fit_tf import TF
from fit.fit_tnf import TNF
from models.model import Disc, Gen
from utils.preprocess import DQ, Cart, DQLinear, LogitTransformer, ScalerBase
from utils.preprocess import SqrtTransformer
from utils.preprocess import SqrtTransformer as LogTransformer
from utils.helpers import *

if len(sys.argv) > 1:
    NAME = sys.argv[1]
else:
    NAME = "jet_tnf"

def setup_scaler_calo(config, data_module,model):
        model.scaler=data_module.scaler
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
        return model

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


    if not model and config["model"] == "MDMA":
        model = MDMA(**config)
    elif not model and config["model"] == "NF":
        model = NF(**config)
    elif not model and config["model"] == "TNF":
        model = TNF(**config)
    elif not model and config["model"] == "TF":
        model = TF(**config)
    elif not model and config["model"] == "PNF":
        model = PNF(**config)
    elif not model and config["model"] == "FM":
        model = FM(**config)
    elif model:
        pass
    else:
        raise ValueError("model not found")
    if "gan" in config.keys():
        model.loss = losses.wasserstein if config["gan"] == "wgan" else losses.least_squares
        model.gp = config["gp"]
        model.d_loss_mean = None
        model.g_loss_mean = None

    if config["dataset"] == "calo":
        model.bins = config["bins"]
        model.num_z, model.num_alpha, model.num_R = model.bins[1:]
        model.E_loss = config["E_loss"]
        model.lambda_gp = config["lambda_gp"]
        model.lambda_response = config["lambda_response"]
        model.min_weighted_w1p = 0.1
        model.min_w1p = 0.1
        model.minE = 0.01
        model.n_dim = 4
        model=setup_scaler_calo(config, data_module,model)
        model.E_loss_mean = None

    else:

        model.bins = [100, 100, 100]
        model.n_dim = 3
        if config["boxcox"]:

            model.scaler = data_module.scaler[0].to("cuda")
            model.pt_scaler= data_module.scaler[1]
        else:
            model.scaler=data_module.scaler.to("cuda")
        model.w1m_best = 0.01

    if config["dataset"] == "jet":
        model.scaled_mins = torch.tensor(data_module.mins).cuda()
        model.scaled_maxs = torch.tensor(data_module.maxs).cuda()
    else:
        model.scaled_mins = torch.zeros(4).cuda()
        model.scaled_maxs = torch.tensor([1e9] + model.bins[1:]).cuda()

    return model


def train(config, logger, data_module, ckpt=False):
    # torch.set_float32_matmul_precision('medium' )
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - ckpt: path to checkpoint if used
    # Callbacks to use during the training, we  checkpoint our models
    print("This is run: ", logger.experiment.name)
    print("config:", config)
    torch.set_float32_matmul_precision("medium")
    mins = torch.ones(config["n_dim"]).unsqueeze(0)
    maxs = torch.ones(config["n_dim"]).unsqueeze(0)
    n = []
    num_batches=0
    for i in data_module.train_dataloader():
        mins = torch.min(
            torch.cat((mins, i[0][~i[1]].min(0, keepdim=True)[0]), dim=0), dim=0
        )[0].unsqueeze(0)
        maxs = torch.max(
            torch.cat((maxs, i[0][~i[1]].max(0, keepdim=True)[0]), dim=0), dim=0
        )[0].unsqueeze(0)
        n.append((~i[1]).sum(1))
        num_batches+=1
        config["avg_n"] = torch.cat(n, dim=0).float().mean()
    config["num_batches"]=num_batches
    if not ckpt:
        model = setup_model(config, data_module, model=False)
    else:
        if config["model"] == "MDMA":
            if config["dataset"] == "jet" and config["n_part"] == 150:
                model = MDMA.load_from_checkpoint(ckpt,**config)

            elif (config["dataset"] == "calo" and "middle" in config.keys() and config["middle"] == False):
                model = MDMA.load_from_checkpoint(
                    ckpt,strict=True, bins=config["bins"],lr=config["lr"],max=config["max"],middle=config["middle"],num_batches=num_batches
                )
                model=setup_scaler_calo(config, data_module,model)

        elif config["model"] == "FM":
            # state_dict=setup_state_dict(state_dict["state_dict"])
            if config["dataset"] == "jet" and config["n_part"] == 150:
                model = FM.load_from_checkpoint(ckpt, **config)
            else:
                model = FM.load_from_checkpoint(ckpt, bins=config["bins"],lr=config["lr"],exact=config["exact"],middle=config["middle"],max=config["max"],num_batches=num_batches)
        elif config["model"] == "NF":
            model = NF.load_from_checkpoint(ckpt, lambda_m=config["lambda_m"],mass_loss=config["mass_loss"],boxcox=config["boxcox"])
            print("model.hparams.boxcox",model.hparams.boxcox)

            model.hparams.boxcox=config["boxcox"]
        elif config["model"] == "PNF":
            model = PNF.load_from_checkpoint(ckpt, adversarial=config["adversarial"],strict=False)
        elif config["model"] == "TNF":
            model = TNF.load_from_checkpoint(ckpt,)
        elif config["model"] == "TF":
            model = TF.load_from_checkpoint(ckpt,)
        model = setup_model(config, data_module, model)
        print(model.scaler)

    model.load_datamodule(data_module)

    # loop once through dataloader to find mins and maxs to clamp during training

    model.maxs = maxs.cuda()
    model.mins = mins.cuda()
    model.avg_n = torch.cat(n, dim=0).float().cuda().mean()
    if config["model"] == "MDMA":
        model.gen_net.avg_n = torch.cat(n, dim=0).float().cuda().mean()
        model.dis_net.avg_n  = torch.cat(n, dim=0).float().cuda().mean()
    trainer = pl.Trainer(
        devices=1,
        precision=32,
        # accumulate_grad_batches=5 if config["model"]=="FM" and config["dataset"]=="calo" and config["middle"]==False else 1,
        accelerator="gpu",
        logger=logger,
        gradient_clip_val=0.5 if config["model"] == "FM" else None,
        log_every_n_steps=100,
        max_epochs=config["max_epochs"]+100,
        callbacks=callbacks,
        val_check_interval=(
            10000 if config["dataset"] == "calo" and config["middle"]==False  and config["model"] == "FM" else 15000
            if (config["dataset"] == "calo") and config["model"] == "FM"
            else 50000 if config["dataset"] == "calo" else None
        ),
        check_val_every_n_epoch=(
            1
            if config["model"] == "PNF" or config["model"]== "NF" and config["ckpt"]
            else (
                10  if (config["ckpt"] == "" and config["dataset"] == "jet")
                else 50 if (config["dataset"] == "jet") else None
            )
        ),
        num_sanity_val_steps=1,
        limit_val_batches=50,
        enable_progress_bar=False,
        default_root_dir="/beegfs/desy/user/{}/{}".format(
            os.environ["USER"], config["dataset"]
        ),
    )
    if ckpt and "continue" in config.keys() and config["continue"]:
        print("continuing training")
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt)
    else:
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    print(NAME)
    config = yaml.load(
        open("hparams/default_{}.yaml".format(NAME)), Loader=yaml.FullLoader
    )
    # set up WandB logger
    logger = WandbLogger(
        save_dir="/beegfs/desy/user/{}/calochallenge".format(os.environ["USER"]),
        sync_tensorboard=False,
        project="MDMA_" + NAME,
    )
    # update config with hyperparameters from sweep
    logger.experiment.log_code(".")
    if len(logger.experiment.config.keys()) > 0:
        config = yaml.load(
            open("hparams/default_{}.yaml".format(logger.experiment.config["name"])),
            Loader=yaml.FullLoader,
        )

        config.update(**logger.experiment.config)
    ckpt = config["ckpt"] if "ckpt" in config.keys() else False

    if config["dataset"] == "calo":
        from utils.dataloader_calo import PointCloudDataloader
    else:
        if config["boxcox"]:
            from utils.dataloader_jetnet import PointCloudDataloader
        else:
            from utils.dataloader_jetnet_std import PointCloudDataloader

    data_module = PointCloudDataloader(**config)
    data_module.setup("train")
    if config["dataset"] == "jet":
        callbacks = [
            ModelCheckpoint(
                monitor="w1m",
                save_top_k=2,
                mode="min",
                filename="{epoch}-{w1m:.5f}-{fpd:.5f}",
                every_n_epochs=1,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="fpd",
                save_top_k=2,
                mode="min",
                filename="{epoch}-{w1m:.5f}-{fpd:.5f}",
                every_n_epochs=1,
            ),

        ]
    else:
        callbacks = [
            ModelCheckpoint(
                monitor="w1p",
                save_top_k=2,
                mode="min",
                filename="{epoch}-{w1p:.5f}-{weighted_w1p:.5f}",
                every_n_epochs=1,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="weighted_z",
                save_top_k=2,
                mode="min",
                filename="{epoch}-{w1p:.5f}-{weighted_w1p:.5f}-{weighted_z:.5f}",
                every_n_epochs=1,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]

    if config["ema"]:
        callbacks.append(EMA(**config))
        callbacks.append(
            EMAModelCheckpoint(
                save_last=True,
                save_top_k=2,
                monitor="step",
                mode="max",
                filename="{epoch}_{step}",
                every_n_epochs=50,
                save_on_train_epoch_end=False,
            )
        )

    train(config, logger=logger, data_module=data_module, ckpt=ckpt)
