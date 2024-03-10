
import os
import time
import traceback
from collections import OrderedDict
import time
from typing import Any, Mapping

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import nflows as nf
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hist import Hist
from jetnet.evaluation import fpd, get_fpd_kpd_jet_features, kpd, w1m
from nflows.flows import base
from nflows.flows.base import Flow
from nflows.nn import nets
from nflows.transforms.autoregressive import *
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import *
from nflows.utils.torchutils import create_random_binary_mask
from pytorch_lightning.loggers import TensorBoardLogger
#from src.models.coimport torchdiffeqmponents.diffusion import VPDiffusionSchedule
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.nn import functional as FF
from torch.optim.lr_scheduler import (ExponentialLR, OneCycleLR,
                                      ReduceLROnPlateau)
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher, TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher)
from torchdyn.core import NeuralODE
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.fm_models import MDMA
from utils.helpers import get_hists, mass, plotting_point_cloud, sample_kde, create_mask
from copy import deepcopy
from torch.cuda import amp


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model,mask,cond,use_t=True):
        super().__init__()
        self.model = model
        self.use_t = use_t
        self.mask = mask
        self.cond = cond

    def forward(self, t, x, *args, **kwargs):
        if self.use_t:
            return self.model( t.repeat(x.shape[0])[:, None],x,mask=self.mask,cond=self.cond)
        else:
            print("not using time, you sure?")
            return self.model(x=x)




class FM(pl.LightningModule):


    def __init__(self,**hparams):

        '''This initializes the model and its hyperparameters'''
        super().__init__()
        self.save_hyperparameters()

        self.logprobs=[]
        self.counter=0
        self.name=hparams["name"]
        self.n_part=self.hparams.n_part
        self.times=[]
        #This is the Normalizing flow model to be used later, it uses as many
        #coupling_layers as given in the config

        self.net = MDMA(**self.hparams).to("cuda")

        self.FM = TargetConditionalFlowMatcher(sigma=0.) if self.hparams.exact=="target" else ConditionalFlowMatcher(sigma=0.) if not self.hparams.exact else ExactOptimalTransportConditionalFlowMatcher()




    def load_datamodule(self,data_module):
        '''needed for lightning training to work, it just sets the dataloader for training and validation'''
        self.data_module=data_module

    def on_after_backward(self) -> None:
        '''This is a genious little hook, sometimes my model dies, i have no clue why. This saves the training from crashing and continues'''
        valid_gradients = False
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print("not valid grads",self.counter)
            self.zero_grad()
            self.counter+=1
            if self.counter>50:
                raise ValueError('5 nangrads in a row')
        else:
            self.counter=0
    def forward(
        self,
        t: Tensor,
        x: Tensor,
        mask: Tensor = None,
        cond: Tensor = None,
    ) -> Tensor:

        return self.net(t, x, mask,cond)

    def on_validation_epoch_start(self, *args, **kwargs):
        # set up the histograms for the validation step
        # also create a list of the real and fake samples and conditions + masks
        self.net.train()
        # self.scaler.to(self.device)
        hists=get_hists(self.hparams.bins,self.scaled_mins.reshape(-1)*1.1,self.scaled_maxs.reshape(-1)*1.1,calo=self.hparams.dataset=="calo")
        self.hists_real,self.hists_fake=hists["hists_real"],hists["hists_fake"]
        if self.hparams.dataset=="calo":
            self.weighted_hists_real,self.weighted_hists_fake=hists["weighted_hists_real"],hists["weighted_hists_fake"]
            self.response_real,self.response_fake=hists["response_real"],hists["response_fake"]
        self.fake =[]

        self.batch = []
        self.conds = []
        self.masks = []

    def sample(self,batch,mask,cond,t_stop=1,num_samples=200):
        '''This is a helper function that samples from the flow (i.e. generates a new sample)
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
            because calculating the mass is a non linear transformation and does not commute with the mass calculation'''
        with torch.no_grad():
            x0 = torch.randn(batch.shape).to(self.device)
            wrapped_cnf = torch_wrapper(model=self.net, mask=mask, cond=cond)
            node=NeuralODE(lambda t,x,args: wrapped_cnf(t,x,mask,cond), solver="midpoint")

            samples = node.trajectory(x0, t_span=torch.linspace(0, t_stop, num_samples).to(self.device),)

            return samples[-1,:,:]

    def sampleandscale(self,batch,mask=None,cond=None,t_stop=1,num_samples=200,scale=True):
        '''This is a helper function that samples from the flow (i.e. generates a new sample)
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
            because calculating the mass is a non linear transformation and does not commute with the mass calculation'''
        print(self.device)
        fake=self.sample(batch,mask,cond,num_samples=num_samples)
        if self.hparams.dataset=="jet":
            if self.hparams.boxcox:
                std_fake=fake[:,:,:2]
                pt_fake=fake[:,:,-1:]
                std_fake= self.scaler.inverse_transform(std_fake)
                pt_fake= self.pt_scaler.inverse_transform(pt_fake)
                fake=torch.cat([std_fake,pt_fake],dim=2)
            else:
                fake= self.scaler.inverse_transform(fake)

        else:
            fake= self.scaler.inverse_transform(fake)
            fake[:,:,2]=(fake[:,:,2]+torch.randint(0,self.hparams.bins[2], size=(fake.shape[0],1),device=fake.device).expand(-1,mask.shape[1]))%self.hparams.bins[2]



        fake[mask]=0
        return fake,None
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:

                    is_changed = True
            else:

                is_changed = True
        if is_changed:
            checkpoint.pop("optimizer_states", None)
    def configure_optimizers(self):



        opt_g = torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weightdecay)
        # if self.hparams.ckpt:
        #     state_dict=torch.load(self.hparams.ckpt,map_location="cuda")
        #     opt_g.load_state_dict(state_dict["optimizer_states"][0])

        lr_scheduler= LinearWarmupCosineAnnealingLR(opt_g, warmup_epochs=self.trainer.max_epochs//10,max_epochs=self.trainer.max_epochs)
        return {"optimizer":opt_g,"lr_scheduler":lr_scheduler}#({'optimizer': opt_g, 'frequency': 1, 'scheduler':None if not self.lr_schedule else scheduler})




    def training_step(self, batch):
        """training loop of the model, here all the data is passed forward to a gaussian
            This is the important part what is happening here. This is all the training we do """

        batch,mask, cond= batch[0],batch[1], batch[2]
        x0,x1 =torch.randn_like(batch), batch
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0,x1)


        vt = self.net(t, xt,mask=mask,cond=cond).cuda()
        # zeroing errors from padded particles
        vt[mask]=0
        ut[mask]=0
        loss = torch.mean((vt - ut) ** 2)


        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mean",vt[~mask].mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/std",vt[~mask].std(), on_step=False, on_epoch=True, prog_bar=False)
        if loss>10:
            return None
        return {"loss": loss}



    def validation_step(self, batch, batch_idx):
        '''This calculates some important metrics on the hold out set (checking for overtraining)'''

        with torch.no_grad():
            if batch[0].shape[1]>0:
                self._log_dict = {}
                batch, mask,cond = batch[0], batch[1], batch[2]
                batch[mask]=0
                self.w1ps = []
                start=time.time()
                fake,_ = self.sampleandscale(batch=batch, mask=mask,cond=cond)
                self.times.append(time.time()-start)

                # self.conds.append(cond.cpu())


                if self.hparams.dataset=="jet":
                    batch=batch.reshape(-1,batch.shape[1],self.n_dim)
                    fake=fake.reshape(-1,batch.shape[1],self.n_dim)

                    self.batch.append(batch.cpu())
                    self.fake.append(fake.cpu())

                    self.masks.append(mask.cpu())
                    for i in range(self.n_dim):
                        self.hists_real[i].fill(batch[~mask.squeeze(-1)][:, i].cpu().numpy())
                        self.hists_fake[i].fill(fake[~mask.squeeze(-1)][:, i].cpu().numpy())
                    self.hists_real[-1].fill(mass(batch).cpu().numpy())
                    self.hists_fake[-1].fill(mass(fake[:len(batch)]).cpu().numpy())
                else:
                    response_real=(batch[:,:,0].sum(1).reshape(-1)/ (cond[:, 0,0] + 10).exp()).cpu().numpy().reshape(-1)
                    response_fake=(fake[:,:,0].sum(1).reshape(-1)/ (cond[:, 0,0] + 10).exp()).cpu().numpy().reshape(-1)

                    for i in range(self.n_dim):
                        self.hists_real[i].fill(batch[~mask.squeeze(-1)][:, i].cpu().long().numpy())
                        self.hists_fake[i].fill(fake[~mask.squeeze(-1)][:, i].cpu().long().numpy())
                        if i>=1:
                            self.weighted_hists_real[i-1].fill(batch[~mask.squeeze(-1)][:, i].cpu().long().numpy(),weight=batch[~mask.squeeze(-1)][:, 0].cpu().numpy())
                            self.weighted_hists_fake[i-1].fill(fake[~mask.squeeze(-1)][:, i].cpu().long().numpy(),weight=fake[~mask.squeeze(-1)][:, 0].cpu().numpy())
                    self.response_real.fill(response_real)
                    self.response_fake.fill(response_fake)


    def on_validation_epoch_end(self):
        if self.hparams.dataset=="jet":
            self.jetnet_evaluation()
        else:
            self.calo_evaluation()

    def jetnet_evaluation(self):
        if len(self.batch)==0:
            self.log("w1m", 0, on_step=False, prog_bar=False, logger=True, on_epoch=True)
            self.log("fpd", 0, on_step=False, prog_bar=False, logger=True, on_epoch=True)
        else:
            real = torch.cat([torch.nn.functional.pad(batch, (0, 0, 0, self.hparams.n_part - batch.size(1))) for batch in self.batch],dim=0) if self.hparams.max else torch.cat(self.batch)
            fake= torch.cat([torch.nn.functional.pad(batch, (0, 0, 0, self.hparams.n_part - batch.size(1))) for batch in self.fake],dim=0) if self.hparams.max else torch.cat(self.fake)
            max_size = max(mask.size(1) for mask in self.masks)

            # Pad each mask to have 'max_size' elements in the second dimension
            masks = torch.cat([torch.nn.functional.pad(mask, (0, self.hparams.n_part - mask.size(1)),value=1) for mask in self.masks])

            if not hasattr(self,"true_fpd") or not hasattr(self,"full"):
                self.true_fpd = get_fpd_kpd_jet_features(real, efp_jobs=1)
                self.full = True
                self.min_fpd = 0.01
            """calculates some metrics and logs them"""

            # calculate w1m 10 times to stabilize for ckpting
            w1m_ = w1m(real,fake,num_batches=16,num_eval_samples=25000)[0]

            fpd_log = 10
            self.log("w1m", w1m_, on_step=False, prog_bar=False, logger=True, on_epoch=True)
            if w1m_ < self.w1m_best and w1m_<0.001:  #
                fake_fpd = get_fpd_kpd_jet_features(fake[:50000], efp_jobs=1)
                fpd_ = fpd(self.true_fpd, fake_fpd, max_samples=len(real))
                fpd_log = fpd_[0]
                self.log("actual_fpd", fpd_[0], on_step=False, prog_bar=False, logger=True)
            if w1m_ < self.w1m_best:  # only log images if w1m is better than before because it takes a lot of time
                self.w1m_best = w1m_
                self.log("best_w1m", w1m_, on_step=False, prog_bar=False, logger=True, on_epoch=True)
            self.plot = plotting_point_cloud(step=self.global_step, logger=self.logger)
            try:
                self.plot.plot_jet(self.hists_real,self.hists_fake)

                # self.plot.plot_scores(torch.cat(self.scores_real).numpy().reshape(-1), torch.cat(self.scores_fake.reshape(-1)).numpy(), False, self.global_step)

            except Exception as e:
                fig,ax=plt.subplots(1,4,figsize=[4*6.4, 6.4])

                for i in range(3):
                    _,bins,_=ax[i].hist(real[~masks.squeeze(-1)].cpu().numpy().reshape(-1,3)[:,i],bins=30,alpha=0.5,label="real")
                    ax[i].hist(fake[~masks.squeeze(-1)].cpu().numpy().reshape(-1,3)[:,i],bins=bins,alpha=0.5,label="fake")
                # ax[-1].hist(mass(real).cpu().numpy().reshape(-1),bins=30,alpha=0.5,label="real")
                # ax[-1].hist(mass(fake).cpu().numpy().reshape(-1),bins=30,alpha=0.5,label="fake")
                # ax[-1].legend()
                # self.logger.log_image("inclusive_debug", [fig],self.global_step)
                plt.close()
                traceback.print_exc()

            self.log("fpd", fpd_log, on_step=False, prog_bar=False, logger=True)



    def calo_evaluation(self,):
        w1ps = []
        weighted_w1ps = []
        plot=False
        if not hasattr(self, "min_w1p") or not hasattr(self, "min_z"):
            self.min_w1p = 10
            self.min_z=0.01
        for i in range(4):
            cdf_fake = self.hists_fake[i].values().cumsum()
            cdf_real = self.hists_real[i].values().cumsum()
            cdf_fake /= cdf_fake[-1]
            cdf_real /= cdf_real[-1]
            w1p = np.mean(np.abs(cdf_fake - cdf_real))
            w1ps.append(w1p)
            if i!=0:
                self.log("features/"+self.hparams.names[i], w1p, on_step=False, on_epoch=True)
                weighted_cdf_fake = self.hists_fake[i].values().cumsum()
                weighted_cdf_real = self.hists_real[i].values().cumsum()
                weighted_cdf_fake /= weighted_cdf_fake[-1]
                weighted_cdf_real /= weighted_cdf_real[-1]
                weighted_w1p = np.mean(np.abs(weighted_cdf_fake - weighted_cdf_real))
                weighted_w1ps.append(weighted_w1p)
                self.log("features/"+self.hparams.names[i] + "_weighted", weighted_w1p, on_step=False, on_epoch=True)
            if i==1:
                self.log("weighted_z", weighted_w1p, on_step=False, on_epoch=True)
                if weighted_w1p<self.min_z:
                    self.min_z=weighted_w1p
                    plot=True
            if i==0:
                self.log("features_E", w1p, on_step=False, on_epoch=True)
                if np.mean(w1p) < self.minE:
                    self.minE = w1p
                    plot=True
        self.log("w1p", np.mean(w1ps), on_step=False, on_epoch=True)
        self.log("weighted_w1p", np.mean(weighted_w1ps), on_step=False, on_epoch=True)
        try:
            if np.mean(w1ps) < self.min_w1p:
                    self.min_w1p = np.mean(w1ps)
                    plot=True
                    self.log("min_w1p", np.mean(w1ps), on_step=False, on_epoch=True)

            if np.mean(weighted_w1ps) < self.min_weighted_w1p:
                    self.min_weighted_w1p = np.mean(weighted_w1ps)
                    plot=True
                    self.log("min_weighted_w1p", np.mean(weighted_w1ps), on_step=False, on_epoch=True)

            if plot:
                self.plot = plotting_point_cloud(step=self.global_step, logger=self.logger)
                self.plot.plot_calo(self.hists_fake, self.hists_real, weighted=False)
                self.plot.plot_calo(self.weighted_hists_fake, self.weighted_hists_real, weighted=True)
                self.plot.plot_response(self.response_fake, self.response_real )
        except:
            traceback.print_exc()
    def on_test_epoch_start(self, *args, **kwargs):
        self.net.eval()
        self.times=[]
        self.n_kde=self.data_module.n_kde
        self.m_kde=self.data_module.m_kde
        hists=get_hists(self.hparams.bins,self.scaled_mins.reshape(-1)*1.1,self.scaled_maxs.reshape(-1)*1.1,calo=self.hparams.dataset=="calo")
        self.hists_real,self.hists_fake=hists["hists_real"],hists["hists_fake"]
        if self.hparams.dataset=="calo":
            self.weighted_hists_real,self.weighted_hists_fake=hists["weighted_hists_real"],hists["weighted_hists_fake"]
            self.response_real,self.response_fake=hists["response_real"],hists["response_fake"]
    def test_step(self, batch, batch_idx):
        '''This calculates some important metrics on the hold out set (checking for overtraining)'''

        with torch.no_grad():

            if batch[0].shape[1]>0:

                self._log_dict = {}
                batch, mask, cond = batch[0], batch[1], batch[2]


                batch[mask]=0
                if self.hparams.dataset=="jet":
                    n,_=sample_kde(len(batch)*10,self.n_kde,self.m_kde)
                    mask=create_mask(n,size=self.hparams.n_part).cuda()

                    mask=mask[:len(batch)].bool()
                    cond=(~mask).sum(1).float().reshape(-1,1,1)/self.data_module.n_mean

                self.w1ps = []
                start=time.time()
                fake=self.sampleandscale(batch=batch, mask=mask,cond=cond,num_samples=200)[0]
                # fake = self.sample(batch=batch, mask=mask,cond=cond,num_samples=2)

                # self.times.append((time.time()-start)/len(fake))

                if self.hparams.dataset=="calo":
                    maxs=torch.tensor([6499, self.hparams.bins[1]-1,self.hparams.bins[2]-1,self.hparams.bins[3]-1],device=self.device)
                    fake=torch.clamp(fake,torch.zeros_like(fake), maxs)
                    response_real=(batch[:,:,0].sum(1).reshape(-1)/ (cond[:,0,0] + 10).exp())
                    response_fake=(fake[:,:,0].sum(1).reshape(-1)/ (cond[:, 0,0] + 10).exp())
                    response_real=torch.clamp(response_real,0.,1.99).cpu().numpy().reshape(-1)
                    response_fake=torch.clamp(response_fake,0.,1.99).cpu().numpy().reshape(-1)
                    for i in range(self.n_dim):
                        self.hists_real[i].fill(batch[~mask.squeeze(-1)][:, i].cpu().long().numpy())
                        self.hists_fake[i].fill(fake[~mask.squeeze(-1)][:, i].cpu().long().numpy())
                        if i>=1:
                            self.weighted_hists_real[i-1].fill(batch[~mask.squeeze(-1)][:, i].cpu().long().numpy(),weight=batch[~mask.squeeze(-1)][:, 0].cpu().numpy())
                            self.weighted_hists_fake[i-1].fill(fake[~mask.squeeze(-1)][:, i].cpu().long().numpy(),weight=fake[~mask.squeeze(-1)][:, 0].cpu().numpy())
                    self.response_real.fill(response_real)
                    self.response_fake.fill(response_fake)


                batch=batch.reshape(-1,batch.shape[1],batch.shape[2])
                fake=fake.reshape(-1,batch.shape[1],batch.shape[2])
                self.batch.append(batch.cpu())
                self.fake.append(fake.cpu())
                self.masks.append(mask.cpu())
                self.conds.append(cond.cpu())