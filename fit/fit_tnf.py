import sys
import traceback

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import wasserstein_distance
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.nn.functional import leaky_relu, sigmoid
from torch.nn.utils.rnn import pad_sequence
from torch.optim.swa_utils import AveragedModel,SWALR
from utils.helpers import CosineWarmupScheduler
rng = np.random.default_rng()
import matplotlib.pyplot as plt
# from metrics import *

from models.flowmodels import Flow,TDisc,TGen
from utils.helpers import get_hists, plotting_point_cloud, mass


def response(fake,batch, cond, mask):
    response_real=(fake[:,:,0].sum(1).reshape(-1)/(cond[:,0]+10).exp()).cpu().numpy()
    response_fake=(batch[:,:,0].sum(1).reshape(-1)/(cond[:,0]+10).exp()).cpu().numpy()
    return response_real,response_fake

class TNF(pl.LightningModule):
    def __init__(self,**hparams):
        """This initializes the model and its hyperparameters, also some loss functions are defined here"""
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.gen_net = TGen(**hparams)
        self.dis_net = TDisc(**hparams)
        state_dict=torch.load(hparams["ckpt_flow"])["state_dict"]
        flow_state_dict = {k.replace('flow.', ''): v for k, v in state_dict.items() if 'flow' in k}
        flow=Flow(**torch.load(hparams["ckpt_flow"])["hyper_parameters"])
        flow.flow.load_state_dict(flow_state_dict)
        self.flow=flow.flow
        self.relu = torch.nn.ReLU()
        self.mse = nn.MSELoss()
        self.step = 0

        self.name=self.hparams.dataset
        self._log_dict = {}



    def on_validation_epoch_start(self, *args, **kwargs):
        # set up the histograms for the validation step
        # also create a list of the real and fake samples and conditions + masks
        self.gen_net.train()
        self.dis_net.train()
        # self.scaler.to(self.device)
        hists=get_hists(self.hparams.bins,self.scaled_mins.reshape(-1)*1.1,self.scaled_maxs.reshape(-1)*1.1,calo=self.name=="calo")
        self.hists_real,self.hists_fake=hists["hists_real"],hists["hists_fake"]
        if self.name=="calo":
            self.weighted_hists_real,self.weighted_hists_fake=hists["weighted_hists_real"], hists["weighted_hists_fake"]
            self.response_real, self.response_fake = hists["response_real"], hists["response_fake"]


        self.fake =[]
        self.batch = []
        self.conds = []
        self.masks = []


    def on_validation_epoch_end(self, *args, **kwargs):
        self.gen_net.train()
        self.dis_net.train()

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module

    def transform(self,x):
        # boxcox transform for energy for calochallenge
        x=(x**self.power_lambda-1)/self.power_lambda
        return (x-self.mean)/self.scale

    def inverse_transform(self,x):
        # inverse boxcox transform for energy for calochallenge
        return ((x*self.scale+self.mean)*self.power_lambda+1)**(1/self.power_lambda)

    def sampleandscale(self, batch, mask, scale=False):
        """Samples from the generator and optionally scales the output back to the original scale"""
        with torch.no_grad():
            z=self.flow.sample(len(batch)).reshape(-1,self.hparams.n_part,self.hparams.n_dim)
            z[mask] = 0  # Since mean field is initialized by sum, we need to set the masked values to zero
        if  not scale or not self.swa:
            fake = self.gen_net(z, mask=mask.clone(), )
        else:
            fake= self.gen_net_averaged(z, mask=mask, )
        fake[:,:,:] = self.relu(fake[:, :, :] - self.mins) + self.mins
        fake[:,:,:] = -self.relu(self.maxs - fake[:, :, :]) + self.maxs
        fake[mask] = 0  # set the masked values to zero
        if scale:

            fake_scaled = self.scaler.inverse_transform(fake).float()
            assert (fake_scaled.reshape(-1,self.hparams.n_dim).max(0)[0]<=self.scaled_maxs[:self.hparams.n_dim]).all()
            fake_scaled[mask] = 0  # set the masked values
            return fake_scaled
        else:
            return fake



    def _gradient_penalty(self, real_data, generated_data, mask, cond):
        """Calculates the gradient penalty loss for WGAN GP, interpolated events always have the same number hits/particles"""
        batch_size = real_data.size()[0]
        alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True)
        prob_interpolated, _ = self.dis_net(interpolated, mask=mask, cond=cond, weight=False)
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones_like(prob_interpolated), create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean()

    def configure_optimizers(self):
        """Sets the optimizer and the learning rate scheduler"""
        if self.hparams.opt == "Adam":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999), eps=1e-14)
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999), eps=1e-14)  #
        elif self.hparams.opt == "AdamW":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999), eps=1e-14)
            opt_d = torch.optim.AdamW(self.dis_net.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999), eps=1e-14, weight_decay=self.hparams.weightdecay)  #
        else:
            raise
        sched_d, sched_g = self.schedulers(opt_d, opt_g)
        if self.swa:
            sched_g=SWALR(opt_g, anneal_strategy="cos", anneal_epochs=5, swa_lr=self.hparams.swa_lr)
        return [opt_d, opt_g], [sched_d, sched_g]

    def schedulers(self, opt_d, opt_g):
        sched_d = CosineWarmupScheduler(opt_d, 2000, 3000 * 1000)
        sched_g = CosineWarmupScheduler(opt_g, 2000, 3000 * 1000)
        return sched_d, sched_g

    def train_disc(self, batch, mask, opt_d):
        """Trains the critic"""
        with torch.no_grad():
            fake= self.sampleandscale(batch=batch, mask=mask)
        batch[mask] = 0
        pred_real = self.dis_net(batch, mask=mask)  # mean_field is used for feature matching
        pred_fake = self.dis_net(fake.detach(), mask=mask)
        self._log_dict["Training/pred_real_mean"]=pred_real.mean()
        self._log_dict["Training/pred_fake_mean"]=pred_fake.mean()
        d_loss=self.loss(pred_real.reshape(-1),pred_fake.reshape(-1),critic=True)
        self.d_loss_mean = d_loss.detach() * 0.01 + 0.99 * self.d_loss_mean if not self.d_loss_mean is None else d_loss
        self._log_dict["Training/d_loss"] = self.d_loss_mean

        opt_d.zero_grad()
        self.dis_net.zero_grad()
        self.manual_backward(d_loss)
        # if self.i%self.hparams.N==0:
        opt_d.step()


    def train_gen(self, batch, mask, opt_g,):
        """Trains the generator"""
        fake = self.sampleandscale(batch=batch, mask=mask)
        pred = self.dis_net(fake, mask=mask)
        self._log_dict["Training/pred_fake_mean_gen"]=pred.mean()
        g_loss=self.loss(None,pred.reshape(-1),critic=False)
        self.g_loss_mean = g_loss.item() * 0.01 + 0.99 * self.g_loss_mean if not self.g_loss_mean is None else g_loss

        opt_g.zero_grad()
        self.gen_net.zero_grad()
        self.manual_backward(g_loss)
        #if self.i%self.hparams.N==0:
        opt_g.step()
        self._log_dict["Training/g_loss"] = self.g_loss_mean

    def training_step(self, batch):
        """simplistic training step, train discriminator and generator"""

        if batch[0].shape[1]>0:
            batch, mask, cond = batch[0], batch[1].bool(), batch[2]
            if self.hparams.noise:
                batch[:, :, :self.hparams.n_dim] *= torch.ones_like(batch[:, :, :self.hparams.n_dim]) + torch.normal(torch.zeros_like(batch[:, :, :self.hparams.n_dim]), torch.ones_like(batch[:, :, :self.hparams.n_dim]) * 1e-3)


            opt_d, opt_g = self.optimizers()
            sched_d, sched_g = self.lr_schedulers()
            self.train_disc(batch=batch, mask=mask, opt_d=opt_d)
            if self.step % (self.hparams.freq) == 0  :
                self.train_gen(batch=batch, mask=mask, opt_g=opt_g)

            if self.step % (100* self.hparams.freq) == 0:
                self.logger.log_metrics(self._log_dict, step=self.global_step)
            sched_d.step()
            sched_g.step()
            self.step += 1
            if self.swa:#and self.i%self.hparams.N==0 :
                self.gen_net_averaged.update_parameters(self.gen_net)

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        with torch.no_grad():
            if batch[0].shape[1]>0:
                self._log_dict = {}
                batch, mask, cond = batch[0], batch[1], batch[2]
                if self.hparams.dataset=="calo":
                    cond = cond.reshape(-1,1,2).float()
                else:
                    cond=cond.reshape(-1,1,1).float()
                self.w1ps = []
                fake = self.sampleandscale(batch=batch, mask=mask, scale=True)
                batch = self.scaler.inverse_transform(batch).float()
                assert (fake==fake).all()
                self.batch.append(batch.cpu())
                self.fake.append(fake.cpu())
                self.masks.append(mask.cpu())
                self.conds.append(cond.cpu())
                if self.name=="jet":
                    self.hists_real[-1].fill(mass(batch).cpu().numpy())
                    self.hists_fake[-1].fill(mass(fake[:len(batch)]).cpu().numpy())
                else:
                    response_real = (fake[:len(cond), :, 0].sum(1) / (cond[:, :,0] + 10).exp()).cpu().numpy().reshape(-1)
                    response_fake = (batch[:, :, 0].sum(1) / (cond[:, :,0] + 10).exp()).cpu().numpy().reshape(-1)
                    self.response_real.fill(response_real)
                    self.response_fake.fill(response_fake)
                    for i in range(self.hparams.n_dim):
                        self.hists_real[i].fill(batch[~mask][:,i].cpu().numpy())
                        self.hists_fake[i].fill(fake[:len(batch)][~mask][:,i].cpu().numpy())
                        if self.name=="calo" and i>0:
                            self.weighted_hists_fake[i-1].fill(fake[:len(batch)][~mask][:,i].cpu().numpy(),weight=fake[:len(batch)][~mask][:,0].cpu().numpy())
                            self.weighted_hists_real[i-1].fill(batch[~mask][:,i].cpu().numpy(),weight=batch[~mask][:,0].cpu().numpy())

            #self.fill_hist(real=batch, fake=fake)



    def on_validation_epoch_end(self):
        if self.hparams.dataset=="calo":
            self.calo_evaluation()
        else:
            self.jetnet_evaluation()

    def jetnet_evaluation(self):
        from jetnet.evaluation import fpd, kpd, get_fpd_kpd_jet_features, w1m
        # Concatenate along the first dimension
        real = torch.cat([torch.nn.functional.pad(batch, (0, 0, 0, 150 - batch.size(1))) for batch in self.batch],dim=0)

        fake= torch.cat([torch.nn.functional.pad(batch, (0, 0, 0, 150 - batch.size(1))) for batch in self.fake],dim=0)
        print("sample lengths:",len(real),len(fake))
        if not hasattr(self,"true_fpd") or not hasattr(self,"full"):
            self.true_fpd = get_fpd_kpd_jet_features(real, efp_jobs=1)
            self.full = True
            self.min_fpd = 0.01
        """calculates some metrics and logs them"""
        # calculate w1m 10 times to stabilize for ckpting

        w1m_ = w1m(real,fake,num_batches=16,num_eval_samples=250000)[0]
        print(w1m_)
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
                plt.close()
                traceback.print_exc()

        self.log("fpd", fpd_log, on_step=False, prog_bar=False, logger=True)
        if self.w1m_best<0.001 and self.hparams.start_swa and not hasattr(self,"gen_net_averaged"):
            self.gen_net_averaged=AveragedModel(self.gen_net)
            self.gen_net_averaged.avg_n=self.avg_n
            self.swa=True

