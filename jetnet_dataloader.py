import os
from random import shuffle

import jetnet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import (DataLoader, Dataset, IterableDataset, Sampler,
                              TensorDataset)

from helpers import *


def custom_collate(data): #(2)
        # x=torch.cat(torch.unsqueeze(data,0),)

        data=torch.stack(data)
        n=(~data[:,:,-1].bool()).sum(1).max()
        return data[:,:int(n)]
class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native
        functions. The module does not expect the tensors to be of any specific shape;
         as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)#/5

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values):
        return (values * self.std) + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def to(self, dev):
        self.std = self.std.to(dev)
        self.mean = self.mean.to(dev)
        return self

class JetNetDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
   """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_dim = config["n_dim"]
        self.n_part = config["n_part"]
        self.batch_size = config["batch_size"]
        self.n_start = config["n_start"]


    def setup(self, stage ,n=None ):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        data=jetnet.datasets.JetNet.getData(jet_type=self.config["parton"],split="train",num_particles=self.n_part,data_dir="/beegfs/desy/user/kaechben/datasets")[0]
        test_set=jetnet.datasets.JetNet.getData(jet_type=self.config["parton"],split="valid",num_particles=self.n_part,data_dir="/beegfs/desy/user/kaechben/datasets")[0]
        data=torch.tensor(data)[:,:,:]
        test_set=torch.tensor(test_set)
        self.data=torch.cat((data,test_set),dim=0)
        if self.n_part>30:
            self.data[:,:,-1]=~self.data[:,:,-1].bool()
        self.n = self.data[:,:,-1].sum(axis=1)
        masks=(self.data[:,:,-1]).bool()
        self.scalers=[]
        self.scaler=StandardScaler()
        temp=self.data[:,:,:-1].reshape(-1,self.n_dim)
        temp[masks.reshape(-1)==0]=self.scaler.fit_transform(temp[masks.reshape(-1)==0,:])
        self.data[:,:,:-1]=temp.reshape(-1,self.n_part,self.n_dim)
        self.data[:,:,-1]=masks
        self.min_pt = torch.min(self.data[~masks][:,2])
        self.test_set = self.data[-len(test_set):].float()
        self.data = self.data[:-len(test_set)].float()

    def train_dataloader(self):
            return DataLoader(self.data[:,:150], batch_size=self.batch_size, shuffle=True,  drop_last=True,num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.test_set[:,:150], batch_size=self.batch_size*10, drop_last=False,num_workers=16 )



if __name__=="__main__":
    config = {
        "n_part": 150,
        "n_start": 150,
        "n_dim": 3,
        "batch_size": 1024,
        "parton": "t",
        "smart_batching":True
     }
    x=JetNetDataloader(config)
    x.setup("train")
    for i in x.val_dataloader():
        print(i.shape)