import os
from random import shuffle

import jetnet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import (DataLoader, Dataset, IterableDataset, Sampler,
                              TensorDataset,BatchSampler)

from utils.helpers import *


def custom_collate(data,avg_n,max=True): #(2)
        # x=torch.cat(torch.unsqueeze(data,0),)
        data=torch.stack(data)
        mask=data[:,:,-1].bool()
        n=(~mask).sum(1).float()
        if max:
            data=data[:,:int(n.max()),:3]
            mask=mask[:,:int(n.max())]
        return data[:,:,:3],mask,n.float().reshape(-1,1,1)/avg_n


class BucketBatchSampler(BatchSampler):
    def __init__(self, data_source, n,batch_size, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.n=n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last


    def __iter__(self):
        indices = list(range(len(self.data_source)))
        # Sort sequences by length
        indices = sorted(indices, key=lambda idx: self.n[idx].item())
        # Create batches based on the sorted indices
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        if self.drop_last or len(batches[-1]) ==0:
            batches = batches[:-1]
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size



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

class PointCloudDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
   """

    def __init__(self,parton,n_dim,n_part,batch_size,pretrain,sampler,max, **kwargs):
        super().__init__()

        self.parton = parton
        self.n_dim = n_dim
        self.n_part = n_part
        self.batch_size = batch_size
        self.pretrain=pretrain
        self.sampler=sampler
        self.max=max



    def setup(self, stage ,n=None ):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        if not self.pretrain:
            data=jetnet.datasets.JetNet.getData(jet_type=self.parton,split="train",num_particles=self.n_part,data_dir="/gpfs/dust/maxwell/user/kaechben/datasets")[0]
            test_set=jetnet.datasets.JetNet.getData(jet_type=self.parton,split="train",num_particles=self.n_part,data_dir="/gpfs/dust/maxwell/user/kaechben/datasets")[0][:50000]
            self.real_test=torch.cat([torch.tensor(jetnet.datasets.JetNet.getData(jet_type=self.parton,split="test",num_particles=self.n_part,data_dir="/gpfs/dust/maxwell/user/kaechben/datasets")[0]),torch.tensor(jetnet.datasets.JetNet.getData(jet_type=self.parton,split="valid",num_particles=self.n_part,data_dir="/gpfs/dust/maxwell/user/kaechben/datasets")[0])])[:50000]
        else:
            data=torch.cat([torch.tensor(jetnet.datasets.JetNet.getData(jet_type=p,split="train",num_particles=self.n_part,data_dir="/gpfs/dust/maxwell/user/kaechben/datasets")[0][:]) for p in ["q","g","w","z"]])
            test_set=torch.cat([torch.Tensor(jetnet.datasets.JetNet.getData(jet_type=p,split="train",num_particles=self.n_part,data_dir="/gpfs/dust/maxwell/user/kaechben/datasets")[0][:12500]) for p in ["q","g","w","z"]])
        data=torch.tensor(data)[:,:,:]
        test_set=torch.tensor(test_set)

        data[:,:,-1]=~data[:,:,-1].bool()
        test_set[:,:,-1]=~test_set[:,:,-1].bool()
        self.real_test[:,:,-1]=~self.real_test[:,:,-1].bool()
        self.n = data[:,:,-1].sum(axis=1)
        self.val_n = test_set[:,:,-1].sum(axis=1)
        self.avg_n = torch.mean(self.n)


        self.mins=np.zeros(4)
        self.maxs=np.zeros(4)
        self.mins[:3]=data.reshape(-1,4).min(0)[0].numpy()[:3]
        self.maxs[:3]=data.reshape(-1,4).max(0)[0].numpy()[:3]
        self.maxs[3]=mass(data[:,:,:3]).max().numpy()
        self.scalers=[]
        self.scaler=StandardScaler()
        temp=data[:,:,:-1].reshape(-1,self.n_dim)
        temp[~(data[:,:,-1]).bool().reshape(-1)]=self.scaler.fit_transform(temp[~(data[:,:,-1]).bool().reshape(-1)])
        print(self.scaler.std)
        data[:,:,:-1]=temp.reshape(-1,self.n_part,self.n_dim)
        self.data = data.float()

        self.test_set =test_set.float()
        self.unscaled_real_test=self.real_test.clone()
        if self.n_part==30:
            temp=self.test_set[:,:,:-1].reshape(-1,self.n_dim)
            temp[~(self.test_set[:,:,-1]).bool().reshape(-1)]=self.scaler.transform(temp[~(self.test_set[:,:,-1]).bool().reshape(-1)])
            self.test_set[:,:,:-1]=temp.reshape(-1,self.n_part,self.n_dim)
            temp=self.real_test[:,:,:-1].reshape(-1,self.n_dim)
            temp[~(self.real_test[:,:,-1]).bool().reshape(-1)]=self.scaler.transform(temp[~(self.real_test[:,:,-1]).bool().reshape(-1)])
            self.real_test[:,:,:-1]=temp.reshape(-1,self.n_part,self.n_dim)

        self.min_pt = self.data.reshape(-1,4)[:,2].min().item()
        self.max_pt = self.data.reshape(-1,4)[:,2].max().item()
        self.train_iterator = BucketBatchSampler(
                                self.data,
                                self.n,
                                batch_size = self.batch_size,
                                drop_last=True,
                                shuffle=True
                                )


    def train_dataloader(self):
        if self.sampler:
            return DataLoader(self.data[:,:150],num_workers=16, collate_fn=lambda x:custom_collate(x,self.avg_n,self.max), batch_sampler=self.train_iterator )
        else:
            return DataLoader(self.data[:,:150],num_workers=16, collate_fn=lambda x:custom_collate(x,self.avg_n,self.max),shuffle=True,drop_last=True,batch_size=self.batch_size )

    def val_dataloader(self):
        return DataLoader(self.test_set[:,:150], batch_size=self.batch_size*100, drop_last=False,num_workers=16,collate_fn=lambda x:custom_collate(x,self.avg_n))

    def test_dataloader(self):
        return DataLoader(self.real_test[:,:150], batch_size=self.batch_size*100, drop_last=False,num_workers=16,collate_fn=lambda x:custom_collate(x,self.avg_n))



if __name__=="__main__":
    config = {
        "n_part": 150,
        "n_dim": 3,
        "batch_size": 1024,
        "parton": "t",
        "smart_batching":True,
        "pretrain":True,
        "sampler":True
     }
    x=PointCloudDataloader(parton="t",n_dim=3,n_part=150,batch_size=128,pretrain=False,sampler=True,max=False)
    x.setup("train")
    for i in x.train_dataloader():
        print((~i[1].bool()).sum(1))
        print(i[0].shape)