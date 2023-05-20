import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from jetnet.evaluation import fpd, fpnd, get_fpd_kpd_jet_features, w1efp, w1m, w1p

# from metrics import *
from models import Gen,Disc
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from helpers import CosineWarmupScheduler, center_jets_tensor, mass
from plots import *


class MDMA(pl.LightningModule):
    def __init__(self, **kwargs):
        """This initializes the model and its hyperparameters, also some loss functions are defined here"""
        super().__init__()
        self.automatic_optimization = False
        self.opt = kwargs["opt"]
        self.parton = kwargs["parton"]
        self.n_dim = kwargs["n_dim"]
        self.n_part = kwargs["n_part"]
        self.lr_g = kwargs["lr_g"]
        self.lr_d = kwargs["lr_d"]
        self.gan = kwargs["gan"]
        self.stop_mean = kwargs["stop_mean"]
        self.mean_field_loss = kwargs["mean_field_loss"]
        self.gen_net = Gen(**kwargs)
        self.dis_net = Disc(**kwargs)
        self.true_fpd = None
        self.save_hyperparameters()
        self.relu = torch.nn.ReLU()
        self.g_loss_mean = 0.5
        self.d_loss_mean = 0.5
        self.w1m_best = 1
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.head_start=0

    def on_validation_epoch_start(self, *args, **kwargs):
        """This is called at the beginning of the validation epoch, it sets the models to eval mode and resets the lists for the metrics"""
        self.gen_net.eval()
        self.dis_net.train()
        self.fake_scaled = []
        self.true_scaled = []
        self.scores_real = []
        self.scores_fake = []

    def on_validation_epoch_end(self, *args, **kwargs):
        """This is called at the end of the validation epoch, it calculates the metrics and logs them"""
        self.gen_net.train()
        self.dis_net.train()
        self.calc_log_metrics(torch.cat(self.fake_scaled, dim=0).numpy(), torch.cat(self.true_scaled, dim=0).numpy(), torch.cat(self.scores_real, dim=0).numpy(), torch.cat(self.scores_fake, dim=0).numpy())

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module


    def sampleandscale(self, batch, mask=None, scale=False):
        """Samples from the generator and optionally scales the output back to the original scale"""
        with torch.no_grad():
            if scale:
                #during evaluation sample 10 times as many events to stabilize w1m metric
                z = torch.normal(torch.zeros(batch.shape[0] * 10, batch.shape[1], batch.shape[2], device=batch.device), torch.ones(batch.shape[0] * 10, batch.shape[1], batch.shape[2], device=batch.device))
                mask = mask.repeat(10, 1).bool()

            else:
                z = torch.normal(torch.zeros(batch.shape[0], batch.shape[1], batch.shape[2], device=batch.device), torch.ones(batch.shape[0], batch.shape[1], batch.shape[2], device=batch.device))
        z[mask] = 0 #mean field is initialized by sum, so we need to set the masked values to zero
        fake = self.gen_net(z, mask=mask, weight=False)
        fake[mask] = 0
        if scale:
            self.data_module.scaler = self.data_module.scaler.to(batch.device)
            fake_scaled = self.data_module.scaler.inverse_transform(fake.clone())
            true = self.data_module.scaler.inverse_transform(batch.clone())
            fake_scaled[mask] = 0
            true[mask[:len(true)]] = 0
            return fake, fake_scaled, true
        else:
            return fake


    def configure_optimizers(self):
        """Sets the optimizer and the learning rate scheduler"""
        if self.opt == "Adam":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.lr_g, betas=(0., 0.999), eps=1e-14)
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.lr_d,betas=(0., 0.999), eps=1e-14)#
        elif self.opt == "AdamW":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.lr_g, betas=(0., 0.999), eps=1e-14)
            opt_d = torch.optim.AdamW(self.dis_net.parameters(), lr=self.lr_d,betas=(0., 0.999), eps=1e-14, weight_decay=0.01)#
        else:
            raise
        sched_d, sched_g = self.schedulers(opt_d, opt_g)
        return [opt_d, opt_g], [sched_d, sched_g]

    def schedulers(self, opt_d, opt_g):
        sched_d = CosineWarmupScheduler(opt_d, 2000, 3000 * 1000)#about  epochs*num_batches
        sched_g = CosineWarmupScheduler(opt_g, 2000, 3000 * 1000)
        return sched_d, sched_g

    def train_disc(self, batch, mask, opt_d):
        """Trains the discriminator"""
        with torch.no_grad():
            fake = self.sampleandscale(batch, mask, scale=False)
        opt_d.zero_grad()
        self.dis_net.zero_grad()
        batch[mask] = 0
        if self.mean_field_loss:
            pred_real, mean_field = self.dis_net(batch, mask=mask, weight=False)  # mean_field is used for feature matching
            pred_fake, _ = self.dis_net(fake.detach(), mask=mask, weight=False)
        else:
            mean_field = None
            pred_real, _= self.dis_net(batch, mask=mask, weight=False)
            pred_fake, _= self.dis_net(fake.detach(), mask=mask, weight=False)
        pred_fake = pred_fake.reshape(-1)
        pred_real = pred_real.reshape(-1)
        if self.gan == "ls":
            with torch.no_grad():
                target_fake = torch.zeros_like(pred_fake)
                target_real = torch.ones_like(pred_real)
            d_loss = self.mse(pred_fake, target_fake).mean() + self.mse(pred_real, target_real).mean()
            self.d_loss_mean = d_loss.detach().item() * 0.01 + 0.99 * self.d_loss_mean


        self.manual_backward(d_loss)
        opt_d.step()
        self.log("Training/d_loss", self.d_loss_mean, on_step=True, logger=True, prog_bar=False)
        return mean_field

    def train_gen(self, batch, mask, opt_g, mean_field=None):
        """Trains the generator"""
        opt_g.zero_grad()
        self.gen_net.zero_grad()
        fake = self.sampleandscale(batch, mask=mask, scale=False)
        fake[mask] = 0
        if mean_field is not None:
            pred, mean_field_gen = self.dis_net(fake, mask=mask, weight=False)
            mean_field = self.mse(mean_field_gen, mean_field.detach()).mean()
            self.log("Training/mean_field", mean_field.detach().item(), on_step=True, logger=True)
        else:
            pred, _ = self.dis_net(fake, mask=mask, weight=False)
        pred = pred.reshape(-1)
        if self.gan == "ls":
            target = torch.ones_like(pred)
            g_loss = 0.5 * self.mse(pred, target).mean()
        else:
            g_loss = self.bce(pred.reshape(-1), torch.ones_like(pred.reshape(-1))).mean()
        if self.g_loss_mean is None:
            self.g_loss_mean = g_loss
        self.g_loss_mean = g_loss.detach().item() * 0.01 + 0.99 * self.g_loss_mean
        if self.mean_field_loss and self.global_step > 1000:
            g_loss += mean_field
        self.manual_backward(g_loss)
        opt_g.step()
        self.log("Training/g_loss", self.g_loss_mean, on_step=True, logger=True)

    def training_step(self, batch):
        """simplistic training step, train discriminator and generator"""
        self.dis_net.eval()
        batch[:, :, :3] *= torch.ones_like(batch[:, :, :3]) + torch.normal(torch.zeros_like(batch[:, :, :3]), torch.ones_like(batch[:, :, :3]) * 1e-3)
        if not hasattr(self, "freq"):
            self.freq = 1
        if self.global_step > 100000 and self.stop_mean:
            self.mean_field_loss = False
        if self.global_step > 100000 and self.d_loss_mean > 0.499:
            self.freq = min(self.freq + 1, 5)
        if self.global_step > 100000 and self.d_loss_mean < 0.4:
            self.freq = 1
        if len(batch) == 1:
            return None
        mask = batch[:, : self.n_part, self.n_dim].bool()
        batch = batch[:, : self.n_part, : self.n_dim]
        opt_d, opt_g = self.optimizers()
        sched_d, sched_g = self.lr_schedulers()
        ### GAN PART
        mean_field = self.train_disc(batch, mask, opt_d)
        self.head_start += 1
        if (self.global_step % (self.freq) == 0 and self.head_start > 1000) or self.d_loss_mean < 0.3:
            self.train_gen(batch, mask, opt_g, mean_field)
        sched_d.step()
        sched_g.step()

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        with torch.no_grad():
            batch = batch
            mask = batch[:, : self.n_part, -1].bool()
            batch = batch[:, : self.n_part, : self.n_dim]
            gen, fake_scaled, true_scaled = self.sampleandscale(batch, mask, scale=True)
            scores_real = self.dis_net(batch, mask=mask[: len(batch)], weight=False)[0]
            scores_fake = self.dis_net(gen[: len(batch)], mask=mask[: len(batch)], weight=False)[0]
            self.scores_fake.append(scores_fake.cpu())
            self.scores_real.append(scores_real.cpu())
            self.true_scaled.append(true_scaled[: len(batch)].cpu())
            self.fake_scaled.append(fake_scaled.cpu())

    def calc_log_metrics(self, fake_scaled, true_scaled, scores_real, scores_fake):
        with torch.no_grad():
            if self.true_fpd is None:
                self.true_fpd = get_fpd_kpd_jet_features(true_scaled, efp_jobs=1)
                self.min_fpd = 0.01
            """calculates some metrics and logs them"""
            # calculate w1m 10 times to stabilize for ckpting
            w1m_ = []
            for i in range(10):
                w1m_.append(scipy.stats.wasserstein_distance(mass(fake_scaled[i * len(true_scaled) : (i + 1) * len(true_scaled)]), mass(true_scaled)))
            w1m_ = np.mean(w1m_)
            fpd_log = 10
            self.log("w1m", w1m_, on_step=False, prog_bar=False, logger=True, on_epoch=True)
            if w1m_ < self.w1m_best * 1.2:  #
                fake_fpd = get_fpd_kpd_jet_features(fake_scaled[: len(true_scaled)], efp_jobs=1)
                fpd_ = fpd(self.true_fpd, fake_fpd, max_samples=len(true_scaled), min_samples=15000)
                fpd_log = fpd_[0]
                self.log("actual_fpd", fpd_[0], on_step=False, prog_bar=False, logger=True)
            if w1m_ < self.w1m_best:  # only log images if w1m is better than before because it takes a lot of time
                self.w1m_best = w1m_
                self.log("best_w1m", w1m_, on_step=False, prog_bar=False, logger=True, on_epoch=True)
                self.plot = plotting_point_cloud(model=self, gen=fake_scaled[: len(true_scaled)], true=true_scaled, n_dim=self.n_dim, n_part=self.n_part, step=self.global_step, logger=None, n=self.n_part, p=self.parton)
                try:
                    self.plot.plot_scores(scores_real.reshape(-1), scores_fake.reshape(-1), False, self.global_step)
                    self.plot.plot_mass(save=None, bins=50)
                except Exception as e:
                    plt.close()
                    traceback.print_exc()

            self.log("fpd", fpd_log, on_step=False, prog_bar=False, logger=True)
