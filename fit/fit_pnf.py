
import os
import time
import traceback
from collections import OrderedDict

import hist
import mplhep as hep
import nflows as nf
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hist import Hist
from jetnet.evaluation import fpd, kpd, get_fpd_kpd_jet_features, w1m
from nflows.flows import base
from nflows.flows.base import Flow
from nflows.nn import nets
from nflows.transforms.autoregressive import *
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import *
from nflows.utils.torchutils import create_random_binary_mask
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.nn import functional as FF
from torch.optim.lr_scheduler import (ExponentialLR, OneCycleLR,
                                      ReduceLROnPlateau)
from models.flowmodels import Flow,Shape, TDisc

from utils.helpers import plotting_point_cloud,mass, get_hists
import matplotlib.pyplot as plt
from time import time

class PNF(pl.LightningModule):


    def __init__(self,**hparams):

        '''This initializes the model and its hyperparameters'''
        super().__init__()

        self.logprobs=[]
        self.save_hyperparameters()
        config=hparams
        config["n_part"]=1
        self.flow=Flow(**config).flow.cuda()
        if hparams["context_features"]>0:
            self.shape=Shape(**hparams).cuda()
        if hparams["adversarial"]:
            self.automatic_optimization=False
            self.dis_net=TDisc(**hparams).cuda()
        self.counter=0
        self.name=hparams["name"]
        self.times=[]
        self.n_part=self.hparams.n_part
        #This is the Normalizing flow model to be used later, it uses as many
        #coupling_layers as given in the config


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
            if self.counter>5:
                raise ValueError('5 nangrads in a row')
        else:
            self.counter=0


    def on_validation_epoch_start(self, *args, **kwargs):
        # set up the histograms for the validation step
        # also create a list of the real and fake samples and conditions + masks
        self.flow.train()
        if self.hparams.model=="MDMA":
            self.gen_net.train()
            self.dis_net.train()
        # self.scaler.to(self.device)
        hists=get_hists(self.hparams.bins,self.scaled_mins.reshape(-1)*1.1,self.scaled_maxs.reshape(-1)*1.1,calo=self.name=="calo")
        self.hists_real,self.hists_fake=hists["hists_real"],hists["hists_fake"]
        self.fake =[]
        self.batch = []
        self.conds = []
        self.masks = []

    def sampleandscale(self,batch,mask=None,cond=None,scale=False):
        '''This is a helper function that samples from the flow (i.e. generates a new sample)
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
            because calculating the mass is a non linear transformation and does not commute with the mass calculation'''
        fake=None
        while fake is None:
            try:

                if self.hparams.context_features>0:
                    fake=self.flow.sample(1,cond)
                else:
                    fake=self.flow.sample(len(batch)*self.n_part)
            except:
                pass

        #This make sure that everything is on the right device
        #Not here that this sample is conditioned on the mass of the current batch allowing the MSE
        #to be calculated later on
        if self.hparams.mass_loss:
            m_f=mass(self.scaler.inverse_transform(fake.reshape(-1,self.n_part,self.n_dim))).reshape(-1)
        else:
            m_f=None
        if scale:
            fake=self.scaler.inverse_transform(fake.reshape(-1,self.n_part,self.n_dim))
            fake[mask]=0


        return fake,m_f

    def configure_optimizers(self):


        if self.hparams.adversarial:
            opt_g = torch.optim.AdamW(list(self.flow.parameters())+list(self.shape.parameters()), lr=self.hparams.lr)
            opt_d = torch.optim.AdamW(self.dis_net.parameters(), lr=self.hparams.lr)
            opt_s = torch.optim.AdamW(self.shape.parameters(), lr=self.hparams.lr)
            return [opt_g,opt_d,opt_s], []
        else:
            if self.hparams.context_features>0:
                parameters=list(self.flow.parameters())+list(self.shape.parameters())
            else:
                parameters=self.flow.parameters()
            opt_g = torch.optim.AdamW(parameters, lr=self.hparams.lr)
        # self.opt_g=opt_g
        # if self.lr_schedule=="onecycle":
        #     scheduler = OneCycleLR(self.opt_g,max_lr=0.01,total_steps=self.max_steps)
        # elif self.lr_schedule=="exp":
        #     scheduler = ExponentialLR(self.opt_g,gamma=0.99)
        # elif self.lr_schedule=="smart":
            # scheduler = OneCycleLR(self.opt_g,"min")
            return opt_g#({'optimizer': opt_g, 'frequency': 1, 'scheduler':None if not self.lr_schedule else scheduler})




    def training_step(self, batch):
        """training loop of the model, here all the data is passed forward to a gaussian
            This is the important part what is happening here. This is all the training we do """
        batch,mask,c= batch[0],batch[1],batch[2]
        batch[mask]=torch.randn_like(batch[mask])*1e-4

        if self.hparams.context_features>=1:

            cond=self.shape(batch.reshape(-1,self.hparams.n_part,self.hparams.n_dim)).repeat_interleave(self.hparams.n_part,dim=0)
        elif self.hparams.context_features==0:
            cond=None
        batch=batch.reshape(-1,self.n_dim)
        # This is the mass constraint, which constrains the flow to generate events with a mass which is the same as the mass it has been conditioned on, we can choose to not calculate this when we work without mass constraint to make training faster
        ##Normalizing Flow loss Normalizing Flow loss

        if self.hparams.adversarial:
            opt_g,opt_d,opt_s=self.optimizers()
            opt_g.zero_grad()
            g_loss = -self.flow.log_prob(batch,cond if self.hparams.context_features else None).mean()/(self.n_dim)
            self.log("logprob", g_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            g_loss.backward()
            opt_g.step()
            with torch.no_grad():
                cond=self.shape(batch.reshape(-1,self.hparams.n_part,self.hparams.n_dim)).repeat_interleave(self.hparams.n_part,dim=0)
                fake,_=self.sampleandscale(batch=batch,mask=mask,cond=cond,scale=False)
            pred_real=self.dis_net(batch.reshape(-1,self.n_part,self.n_dim),mask=mask)
            pred_fake=self.dis_net(fake.reshape(-1,self.n_part,self.n_dim),mask=mask)
            opt_d.zero_grad()
            d_loss = self.loss(pred_real, pred_fake,critic=True)
            d_loss.backward()
            opt_d.step()
            self.log("d_loss", d_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            opt_s.zero_grad()
            cond=self.shape(batch.reshape(-1,self.hparams.n_part,self.hparams.n_dim)).repeat_interleave(self.hparams.n_part,dim=0)
            fake,_=self.sampleandscale(batch=batch,mask=mask,cond=cond,scale=False)
            pred_fake=self.dis_net(fake.reshape(-1,self.n_part,self.n_dim),mask=mask)
            g_loss = self.loss(None, pred_fake,critic=False)
            g_loss.backward()
            opt_s.step()
            self.log("shape_loss", g_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        else:
            g_loss = -self.flow.log_prob(batch,cond if self.hparams.context_features else None).mean()/(self.n_dim)
            self.log("logprob", g_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        #some conditions on when we want to actually add the mass loss to our training loss, if we dont add it, it is as it wouldnt exist

            return OrderedDict({"loss":g_loss})



    def validation_step(self, batch, batch_idx):
        '''This calculates some important metrics on the hold out set (checking for overtraining)'''
        with torch.no_grad():

            if batch[0].shape[1]>0:

                self._log_dict = {}

                batch, mask, cond = batch[0], batch[1], batch[2]
                if self.hparams.context_features>=1:
                    cond=self.shape(batch).repeat_interleave(self.hparams.n_part,dim=0)
                elif self.hparams.context_features==0:
                    cond=None
                scaled_batch=self.scaler.inverse_transform(batch)
                scaled_batch[mask]=0
                logprob=-self.flow.log_prob(batch.reshape(-1,self.n_dim),cond if self.hparams.context_features else None).mean()/(self.n_dim)
                self.log("val_logprob",logprob, logger=True)

                self.w1ps = []
                start=time()
                fake,mf = self.sampleandscale(batch=batch, mask=mask, cond=cond, scale=True)
                self.times.append(time()-start)
                batch=scaled_batch.reshape(-1,self.n_part,self.n_dim)
                fake=fake.reshape(-1,self.n_part,self.n_dim)
                self.batch.append(batch.cpu())
                self.fake.append(fake.cpu())
                self.masks.append(mask.cpu())
                if cond is not None:
                    self.conds.append(cond.cpu())
                for i in range(self.n_dim):
                    self.hists_real[i].fill(batch[~mask][:, i].cpu().numpy())
                    self.hists_fake[i].fill(fake[~mask][:, i].cpu().numpy())

                self.hists_real[-1].fill(mass(batch).cpu().numpy())
                self.hists_fake[-1].fill(mass(fake[:len(batch)]).cpu().numpy())


    def on_validation_epoch_end(self):
        self.jetnet_evaluation()

    def jetnet_evaluation(self):
        real = torch.cat([torch.nn.functional.pad(batch, (0, 0, 0, self.hparams.n_part - batch.size(1))) for batch in self.batch],dim=0) if self.hparams.max else torch.cat(self.batch)
        fake= torch.cat([torch.nn.functional.pad(batch, (0, 0, 0, self.hparams.n_part - batch.size(1))) for batch in self.fake],dim=0) if self.hparams.max else torch.cat(self.fake)
        print("sample lengths:",len(real),len(fake))
        if not hasattr(self,"true_fpd") or not hasattr(self,"full"):
            self.true_fpd = get_fpd_kpd_jet_features(real, efp_jobs=1)
            self.full = True
            self.min_fpd = 0.01
        """calculates some metrics and logs them"""
        # calculate w1m 10 times to stabilize for ckpting

        w1m_ = w1m(real,fake,num_batches=16,num_eval_samples=250000)[0]

        fpd_log = 10
        self.log("w1m", w1m_, on_step=False, prog_bar=False, logger=True, on_epoch=True)
        if w1m_ < self.w1m_best * 1.2:  #
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
                ax[i].hist(torch.cat(self.batch).cpu().numpy().reshape(-1,3)[:,i],bins=30,alpha=0.5,label="real")
                ax[i].hist(torch.cat(self.fake).cpu().numpy().reshape(-1,3)[:,i],bins=30,alpha=0.5,label="fake")
            ax[-1].hist(mass(torch.cat(self.batch)).cpu().numpy().reshape(-1),bins=30,alpha=0.5,label="real")
            ax[-1].hist(mass(torch.cat(self.fake)).cpu().numpy().reshape(-1),bins=30,alpha=0.5,label="fake")
            ax[-1].legend()
            self.logger.log_image("inclusive", [fig],self.global_step)



            plt.close()
            traceback.print_exc()

        self.log("fpd", fpd_log, on_step=False, prog_bar=False, logger=True)

