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

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
rng = np.random.default_rng()
import matplotlib.pyplot as plt
# from metrics import *
from models.flowmodels import TGen, TDisc
from utils.helpers import get_hists, plotting_point_cloud, mass, sample_kde, create_mask
import time
import pickle
def response(fake,batch, cond, mask):
    response_real=(fake[:,:,0].sum(1).reshape(-1)/(cond[:,0]+10).exp()).cpu().numpy()
    response_fake=(batch[:,:,0].sum(1).reshape(-1)/(cond[:,0]+10).exp()).cpu().numpy()
    return response_real,response_fake
def add_sn(m):
    if isinstance(m,nn.Linear):
        return torch.nn.utils.parametrizations.spectral_norm(m)
    else:
        return m

def add_wn(m):
    if isinstance(m,nn.Linear):
        return torch.nn.utils.parametrizations.weight_norm(m)
    else:
        return m
class TF(pl.LightningModule):
    def __init__(self,**hparams):
        """This initializes the model and its hyperparameters, also some loss functions are defined here"""
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.gen_net = TGen(**hparams)
        self.dis_net = TDisc(**hparams)

        self.relu = torch.nn.ReLU()
        self.mse = nn.MSELoss()
        self.step = 0

        self.name=self.hparams.dataset
        self._log_dict = {}
        self.times=[]
        if "spectralnorm" in hparams.keys() and hparams["spectralnorm"]:
            self.dis_net.apply(add_sn)
        elif"weightnorm" in hparams.keys() and hparams["spectralnorm"]:
            self.dis_net.apply(add_wn)
        self.dis_net=self.dis_net.cuda()
        with open("kde/" +"t" + "_kde.pkl", "rb") as f:
            self.kde=pickle.load(f)


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

    def sampleandscale(self, batch, mask, cond, scale=False,test=False):
        """Samples from the generator and optionally scales the output back to the original scale"""
        # Since mean field is initialized by sum, we need to set the masked values to zero
        with torch.no_grad():
            z = torch.normal(torch.zeros(mask.shape[0], mask.shape[1], self.n_dim, device=mask.device), torch.ones(mask.shape[0], mask.shape[1], self.n_dim, device=mask.device))
            z[mask] = 0  # Since mean field is initialized by sum, we need to set the masked values to zero

        fake = self.gen_net(z.cuda(), mask=mask.clone().cuda(), cond=cond)
        fake[:,:,:] = self.relu(fake[:, :, :] - self.mins) + self.mins
        fake[:,:,:] = -self.relu(self.maxs - fake[:, :, :]) + self.maxs
        fake[mask] = 0  # set the masked values to zero
        if scale:

            fake_scaled = self.scaler.inverse_transform(fake).float()

            fake_scaled[mask] = 0  # set the masked values
            return fake_scaled,None
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

        return [opt_d, opt_g], [sched_d, sched_g]

    def schedulers(self, opt_d, opt_g):
        sched_d= LinearWarmupCosineAnnealingLR(opt_d, warmup_epochs=self.hparams.num_batches*5,max_epochs=self.trainer.max_epochs*self.hparams.num_batches, warmup_start_lr=1e-6, eta_min=1e-6)
        sched_g= LinearWarmupCosineAnnealingLR(opt_g, warmup_epochs=self.hparams.num_batches*5,max_epochs=self.trainer.max_epochs*self.hparams.num_batches, warmup_start_lr=1e-6, eta_min=1e-6)

        return sched_d, sched_g

    def train_disc(self, batch, mask, opt_d, cond):
        """Trains the critic"""
        with torch.no_grad():
            fake= self.sampleandscale(batch=batch, mask=mask, cond=cond,scale=False)
        batch[mask] = 0
        pred_real, mean_field = self.dis_net(batch, mask=mask,  cond=cond,return_mf=True)  # mean_field is used for feature matching
        pred_fake,_ = self.dis_net(fake.detach(), mask=mask,  cond=cond)
        self._log_dict["Training/pred_real_mean"]=pred_real.mean()
        self._log_dict["Training/pred_fake_mean"]=pred_fake.mean()
        d_loss=self.loss(pred_real.reshape(-1),pred_fake.reshape(-1),critic=True)
        self.d_loss_mean = d_loss.detach() * 0.01 + 0.99 * self.d_loss_mean if not self.d_loss_mean is None else d_loss
        self._log_dict["Training/d_loss"] = self.d_loss_mean
        if self.hparams.gp:
            gp = self._gradient_penalty(batch, fake, mask=mask, cond=cond)
            d_loss += self.hparams.lambda_gp*gp
            self._log_dict["Training/gp"] = gp
        opt_d.zero_grad()
        self.dis_net.zero_grad()
        self.manual_backward(d_loss)
        # if self.i%self.hparams.N==0:
        opt_d.step()

        return mean_field

    def train_gen(self, batch, mask, opt_g, cond, mean_field=None):
        """Trains the generator"""
        fake = self.sampleandscale(batch=batch, mask=mask, cond=cond)
        pred, mean_field_gen = self.dis_net(fake, mask=mask,  cond=cond,return_mf=True)
        self._log_dict["Training/pred_fake_mean_gen"]=pred.mean()
        g_loss=self.loss(None,pred.reshape(-1),critic=False)
        self.g_loss_mean = g_loss.item() * 0.01 + 0.99 * self.g_loss_mean if not self.g_loss_mean is None else g_loss
        if self.hparams.mean_field_loss and self.step<50000 and self.hparams.fast:

            mean_field = ((mean_field_gen-mean_field.detach())**2).mean()

            self._log_dict["Training/mean_field"] = mean_field
            g_loss += mean_field
        if self.name=="calo" and self.E_loss:
            g_loss += self.calc_E_loss(fake, batch, mask, cond)
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

            if self.global_step > 500000 and self.hparams.stop_mean:
                self.hparams.mean_field_loss = False
            opt_d, opt_g = self.optimizers()
            sched_d, sched_g = self.lr_schedulers()
            mean_field= self.train_disc(batch=batch, mask=mask, opt_d=opt_d, cond=cond)
            if self.step % (self.hparams.freq) == 0  :
                self.train_gen(batch=batch, mask=mask, opt_g=opt_g, cond=cond, mean_field=mean_field)

            if self.step % (100* self.hparams.freq) == 0:
                self.logger.log_metrics(self._log_dict, step=self.global_step)
            sched_d.step()
            sched_g.step()
            self.step += 1


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
                start=time.time()
                fake,_ = self.sampleandscale(batch=batch, mask=mask, cond=cond, scale=True)
                self.times.append(start-time.time())
                assert (fake==fake).all()
                self.batch.append(batch.cpu())
                self.fake.append(fake.cpu())
                self.masks.append(mask.cpu())
                self.conds.append(cond.cpu())
                if self.name=="jet":
                    for i in range(3):
                        self.hists_real[i].fill(batch[~mask][:, i].cpu().numpy())
                        self.hists_fake[i].fill(fake[~mask][:, i].cpu().numpy())
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
        # self.gen_net.eval()
        self.n_kde=self.data_module.n_kde
        self.m_kde=self.data_module.m_kde
    def test_step(self, batch, batch_idx):
        '''This calculates some important metrics on the hold out set (checking for overtraining)'''

        with torch.no_grad():

            if batch[0].shape[1]>0:

                self._log_dict = {}
                batch, mask, cond = batch[0], batch[1], batch[2]
                batch[mask]=0
                n,_=sample_kde(len(batch)*10,self.n_kde,self.m_kde)
                mask=create_mask(n).cuda()
                start=time.time()
                mask=mask[:len(batch)]

                self.w1ps = []
                start=time.time()
                fake,mf = self.sampleandscale(batch=batch.cuda(), mask=mask.cuda(), cond=cond.cuda(), scale=True)
                fake[mask]=0
                self.times.append(time.time()-start)
                # batch=batch.reshape(-1,self.hparams.n_part,self.n_dim)
                # fake=fake.reshape(-1,self.hparams.n_part,self.n_dim)
                self.batch.append(batch.cpu())
                self.fake.append(fake.cpu())
                self.masks.append(mask.cpu())
                self.conds.append(cond.cpu())

