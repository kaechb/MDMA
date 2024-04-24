import copy

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   StandardScaler)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset

#from preprocess import ScalerBase,DQ,LogitTransformer
from utils.preprocess import (DQ, DQLinear, LogitTransformer, ScalerBase,
                            SqrtTransformer)

from utils.helpers import PowerLawModel, Nflow


class BatchIterator:
    def __init__(self, data_source, max_tokens_per_batch, batch_size, drop_last=False, shuffle=True):
        self.data_source = data_source
        self.max_tokens_per_batch = max_tokens_per_batch
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batches = self._create_batches()

    def _get_group_key(self, length, range_size=200):
        """Determine the group key for a given length."""
        return length // range_size * range_size

    def _create_batches(self):
        """Create batches based on similar size grouping with tolerance."""
        length_groups = {}
        for idx in range(len(self.data_source)):
            sample_len = len(self.data_source[idx])
            group_key = self._get_group_key(sample_len, 200)  # Grouping by 200 tokens range
            length_groups.setdefault(group_key, []).append(idx)

        # Shuffle within each length group
        for group_key in length_groups:
            np.random.shuffle(length_groups[group_key])

        # Form batches
        batches = []
        batch = []
        batch_tokens = 0
        for group_key in sorted(length_groups.keys()):
            for idx in length_groups[group_key]:
                sample_len = len(self.data_source[idx])
                if batch_tokens + sample_len > self.max_tokens_per_batch or len(batch) >= self.batch_size:
                    batches.append(batch)
                    batch = []
                    batch_tokens = 0
                batch.append(idx)
                batch_tokens += sample_len
        if not self.drop_last and len(batch) > 0:
            batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)

        return batches

    def __iter__(self):
        """Return the iterator itself."""
        self.batch_index = 0
        return self

    def __next__(self):
        """Return the next batch."""
        if self.batch_index < len(self.batches):
            batch = self.batches[self.batch_index]
            self.batch_index += 1
            return batch
        else:
            raise StopIteration

    def __len__(self):
        """Return the number of batches."""
        return len(self.batches)

class CustomDataset(Dataset):
    def __init__(self, data, E,n_test=None):
        assert len(data) == len(E), "The lengths of data and E are not equal"
        self.data = data
        self.E = E
        self.n_test=n_test

    def __getitem__(self, index):
        if self.n_test is not None:
            return self.data[index], self.E[index],self.n_test[index]
        return self.data[index], self.E[index]

    def __len__(self):
        return len(self.data)




def pad_collate_fn(batch,avg_n):
    if len(batch[0])==2:
        batch,E=zip(*batch)
        n_test=None
    else:
        batch,E,n_test=zip(*batch)
    max_len = max(len(sample) for sample in batch)
    padded_batch =pad_sequence(batch, batch_first=True, padding_value=0.0)[:,:,:4].float()
    mask = ~(torch.arange(max_len).expand(len(batch), max_len) < torch.tensor([len(sample) for sample in batch]).unsqueeze(1))
    E=(torch.stack(E).log()-10).float()
    cond=torch.cat((E.unsqueeze(-1).unsqueeze(-1),(~mask).int().sum(1).float().unsqueeze(-1).unsqueeze(-1)/avg_n),dim=-1)
    if n_test is not None:
        return padded_batch,mask,cond,n_test
    return padded_batch,mask,cond



class PointCloudDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
   """

    def __init__(self,name,batch_size,max=False,scaler_path="./",middle=True,max_tokens=500_000,**kwargs):
        self.name=name
        self.batch_size=batch_size
        self.max=max
        self.max_tokens=max_tokens
        self.new=True if name.find("new")>-1 else False
        self.scaler_path=scaler_path
        self.dataset="middle" if middle else "big"
        self.mins=torch.tensor([[-3.3306, -8.5344, -6.3496, -5.8907]])#these are the minimums of the scaled data, this is used for plotting
        self.maxs=torch.tensor([[3.7937, 9.5630, 6.3497, 6.5893]])#these are the maxmums of the scaled data, this is used for plotting
        self.avg_n=1587.93468#these are the average number of particles per cloud, used to scale the condition
        self.n_kde=None
        self.m_kde=None
        bins=5
        # self.n_flow = Nflow()
        # self.n_flow.load_state_dict(torch.load("ckpts/n_flow_{}.pt".format("middle" if middle else "big")))

        # self.n_flow.load_state_dict(torch.load("ckpts/n_flow_{}.pt".format("middle" if middle else "big")))
        self.n_reg = PowerLawModel(coeffs=torch.zeros(6))
        self.n_reg.load_state_dict(torch.load("ckpts/n_reg_{}.pt".format("middle" if middle else "big")))
        print(self.n_reg.state_dict())
        self.n_std=self.n_reg.n_std
        self.n_mean=self.n_reg.n_mean

        super().__init__()

    def setup(self, stage ):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        self.data=torch.load(f"/beegfs/desy/user/kaechben/calochallenge/pc_train_{self.dataset}.pt")
        self.E=self.data["Egen"]
        self.data=self.data["E_z_alpha_r"]
        self.val_data=torch.load(f"/beegfs/desy/user/kaechben/calochallenge/pc_test_{self.dataset}.pt")
        self.test_E=self.val_data["Egen"]
        self.test_data=self.val_data["E_z_alpha_r"]
        self.val_E=self.val_data["Egen"]
        self.val_data=self.val_data["E_z_alpha_r"]
        print(len(self.val_data))


        self.scaler = ScalerBase(
                        transfs=[PowerTransformer(method="box-cox", standardize=True),
                            Pipeline([('dequantization', DQLinear(name=self.name)),('minmax_scaler', MinMaxScaler(feature_range=(1e-5, 1-1e-5))),('logit_transformer', LogitTransformer()),("standard_scaler",StandardScaler())]),
                        ],
                        featurenames=["E", "z", "alpha", "r"],
                        name=self.dataset,
                        data_dir=self.scaler_path,
                        overwrite=False)
        del self.scaler.transfs[1].steps[0]
        self.mins=torch.ones(4).unsqueeze(0)
        self.maxs=torch.ones(4).unsqueeze(0)
        self.n_test=self.sample_n(torch.tensor(self.test_E).reshape(-1))

        self.train_iterator = BatchIterator(
                            self.data,
                            batch_size = self.batch_size,
                            max_tokens_per_batch=self.max_tokens//2.5,
                            drop_last=True,
                            shuffle=True
                            )
        self.val_iterator = BatchIterator(
                            self.val_data,
                            batch_size = self.batch_size,
                            max_tokens_per_batch=self.max_tokens//3,
                            drop_last=False,
                            shuffle=True
                            )
        self.test_iterator = BatchIterator(
                            self.test_data,
                            batch_size = self.batch_size,
                                max_tokens_per_batch=self.max_tokens//2,
                                drop_last=False,
                                shuffle=False
                                )

        self.train_dl = DataLoader(CustomDataset(self.data,self.E), batch_sampler=self.train_iterator,collate_fn=lambda x: pad_collate_fn(x,self.avg_n),num_workers=16)
        self.val_dl = DataLoader(CustomDataset(self.val_data,self.val_E), batch_sampler=self.val_iterator,collate_fn=lambda x: pad_collate_fn(x,self.avg_n),num_workers=16)
        self.test_dl = DataLoader(CustomDataset(self.test_data,self.test_E,self.n_test), batch_sampler=self.test_iterator,collate_fn=lambda x: pad_collate_fn(x,self.avg_n),num_workers=40)

    def trafo(self,x,y):
        y=(y+torch.rand_like(y))
        E=(x+10).exp()
        y=(y/E**(1/2))#.log()
        self.n_mean,self.n_std=y.mean(),y.std()
        y=(y-self.n_mean)/self.n_std
        return x,y
    def inv_trafo(self,x,y,n_mean,n_std):
        y=y*n_std+n_mean
        E=(x+10).exp()
        y=y*E**(1/2)
        y=y.floor()
        return E,y

    def sample_n(self,cond):
        '''This is a helper function that samples the number of hits from the conditional distribution p(n|E). It uses the results from a 5th order polynomial regression to remove outliers the flow samples'''
        with torch.no_grad():

            n_pred = self.n_reg(cond.reshape(-1,1).log()-10)
            _,n_pred=self.inv_trafo(cond.reshape(-1).log()-10,n_pred.reshape(-1),self.n_mean,self.n_std)
            print(n_pred.sum())
            # n = n_flow(cond.reshape(-1,1).log()-10)
            # n=(n*n_flow.n_std+n_flow.n_mean).exp()
            # n_pred=(n_pred*n_flow.n_std+n_flow.n_mean).exp()
            # residual= ((n - n_pred)).abs()
            # outliers= residual/n_pred.sqrt()>30
            # n[outliers]=n_pred[outliers]
            # residuals=torch.abs(n-n_pred)
            # outliers=residuals/n_pred.sqrt()>5
            # n[outliers]=n_pred[outliers]
            return n_pred

    def train_dataloader(self):
        return self.train_dl# DataLoader(self.data, batch_size=10, shuffle=False, num_workers=1, drop_last=False,collate_fn=point_cloud_collate_fn)

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl
if __name__=="__main__":

    loader=PointCloudDataloader("middle",64,max=True,augmentation=False,scaler_path="./")
    loader.setup("train")

    mins=torch.ones(4).unsqueeze(0)
    responses=[]
    import matplotlib.pyplot as plt
    for i in loader.train_dataloader():
        batch,mask,cond=i
        batch[~mask]=loader.scaler.inverse_transform(batch[~mask])
        batch[mask]=0
        response=batch[:,:,0].sum(1)/((cond[:,0,0]+10).exp())
        responses.append(response)
    plt.hist(torch.cat(responses).detach().numpy(),bins=100)
    plt.savefig("response.png")
    responses=[]
    plt.close()
    for i in loader.val_dataloader():
        batch,mask,cond=i
        response=batch[:,:,0].sum(1)/((cond[:,0,0]+10).exp())
        responses.append(response)
    plt.hist(torch.cat(responses).detach().numpy(),bins=100)
    plt.savefig("response_val.png")