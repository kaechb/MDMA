import sys

sys.path.insert(1, "/home/kaechben/ProGamer")
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF
from tqdm import tqdm

from fit import MDMA
from helpers import *
from jetnet_dataloader import JetNetDataloader


def train(config):
    torch.set_float32_matmul_precision("medium")
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - ckpt: path to checkpoint if used
    # Callbacks to use during the training, we  checkpoint our models

    model = MDMA(config=config, **config)
    model.ckpt = None
    data_module = JetNetDataloader(config)
    data_module.setup("validation")
    model.data_module = data_module
    callbacks = [ModelCheckpoint(monitor="w1m", save_top_k=3, mode="min", filename="{epoch}-{w1m:.5f}-{fpd:.7f}", every_n_epochs=1), pl.callbacks.LearningRateMonitor(logging_interval="step")]
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=None,
        log_every_n_steps=100,
        max_epochs=3000,
        callbacks=callbacks,
        val_check_interval=2000,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=1,
        enable_progress_bar=False,
        default_root_dir="/beegfs/desy/user/kaechben/MF2",
    )
    # This calls the fit function which trains the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":

    config = {
        "batch_size": 128,
        "ckpt": False,
        "dropout_gen": 0,
        "dropout": 0.1,
        "freq": 1,
        "gan": "ls",
        "heads_gen": 8,
        "heads": 4,
        "hidden_gen": 48,
        "hidden": 64,
        "l_dim_gen": 16,
        "l_dim": 16,
        "l_dim": 16,
        "lr_d": 0.0001,
        "lr_g": 0.0001,
        "mean_field_loss": True,
        "n_dim": 3,
        "n_part": 150,
        "n_start": 150,
        "name": "MF2",
        "new": False,
        "num_layers_gen": 7,
        "num_layers": 2,
        "opt": "AdamW",
        "parton": "q",
        "stop_mean": True,
    }
    train(config)  # load_ckpt=ckptroot=root,
