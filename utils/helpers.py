import copy
import datetime
import math
import os
import random
import time
import traceback
import warnings
from functools import partial
from math import sqrt
from pathlib import Path

import h5py
import hist
import joblib
import matplotlib
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
from hist import Hist
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from helpers import mass
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   StandardScaler)
from torch import nn, optim
from torch.nn import Parameter
from torch.nn import functional as FF
from torch.nn.utils import spectral_norm, weight_norm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm


def mass(p, canonical=False):
    if not torch.is_tensor(p):
        p=torch.tensor(p)
    if len(p.shape)==2:
        n_dim = p.shape[1]
        p = p.reshape(-1, n_dim // 3, 3)

    px = torch.cos(p[:, :, 1]) * p[:, :, 2]
    py = torch.sin(p[:, :, 1]) * p[:, :, 2]
    pz = torch.sinh(p[:, :, 0]) * p[:, :, 2]
    E = torch.sqrt(px**2 + py**2 + pz**2)
    E = E.sum(axis=1) ** 2

    p = px.sum(axis=1) ** 2 + py.sum(axis=1) ** 2 + pz.sum(axis=1) ** 2
    m2 = E - p

    return torch.sqrt(torch.max(m2, torch.zeros(len(E)).to(E.device)))



class TPReLU(nn.Module):

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(TPReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.bias = Parameter(torch.zeros(num_parameters))

    def forward(self, input):
        bias_resize = self.bias.view(1, self.num_parameters, *((1,) * (input.dim() - 2))).expand_as(input)
        return F.prelu(input - bias_resize, self.weight.clamp(0, 1)) + bias_resize

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_parameters) + ')'
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch >= self.max_num_iters:
            lr_factor*=self.max_num_iters/epoch
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class WeightNormalizedLinear(nn.Module):

    def __init__(self, in_features, out_features, scale=False, bias=False, init_factor=1, init_scale=1):
        super(WeightNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if scale:
            self.scale = Parameter(torch.Tensor(out_features).fill_(init_scale))
        else:
            self.register_parameter('scale', None)

        self.reset_parameters(init_factor)

    def reset_parameters(self, factor):
        stdv = 1. * factor / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def weight_norm(self):
        return self.weight.pow(2).sum(1).sqrt().add(1e-8)

    def norm_scale_bias(self, input):
        output = input.div(self.weight_norm().unsqueeze(0))
        if self.scale is not None:
            output = output.mul(self.scale.unsqueeze(0))
        if self.bias is not None:
            output = output.add(self.bias.unsqueeze(0))
        return output

    def forward(self, input):
        return self.norm_scale_bias(F.linear(input, self.weight))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

def masked_layer_norm(x, mask, eps = 1e-5):
    """
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L)]
    """
    mask = mask.float().unsqueeze(-1)  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  #(N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm



class plotting_point_cloud():
    '''This is a class that takes care of  plotting steps in the script,
        It is initialized with the following arguments:
        true=the simulated data, note that it needs to be scaled
        gen= Generated data , needs to be scaled
        step=The current step of the training, this is need for tensorboard
        model=the model that is trained, a bit of an overkill as it is only used to access the losses
        config=the config used for training
        logger=The logger used for tensorboard logging'''
    def __init__(self,step=None,logger=None,weight=1):

        self.step=step

        self.weight=weight
        self.fig_size1=[6.4, 6.4]
        self.fig_size2=[2*6.4, 6.4]
        self.fig_size3=[3*6.4, 6.4]
        self.fig_size4=[4*6.4, 6.4]
        self.alpha=0.3
        mpl.rcParams['lines.linewidth'] = 2
        font = { "size": 18}#"family": "normal",
        mpl.rc("font", **font)
        mpl.rc('lines', linewidth=2)
        sns.set_palette("Pastel1")
        if logger is not None:
            self.summary=logger
        else:
            self.summary = None

    def plot_calo(self,h_real,h_fake,weighted,leg=-1):
        #This creates a histogram of the inclusive distributions and calculates the mass of each jet
        #and creates a histogram of that
        #if save, the histograms are logged to tensorboard otherwise they are shown
        #if quantile, this also creates a histogram of a subsample of the generated data,
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i=0
        k=0
        fig,ax=plt.subplots(2,4 if not weighted else 3,gridspec_kw={'height_ratios': [4, 1]},figsize=self.fig_size4,sharex="col")

        cols=["E","z","alpha","R"]
        names=[r"$E$",r"$z$",r"$\alpha$",r"$R$"]
        if weighted:
            cols=["z","alpha","R"]
            names=[r"$z$",r"$\alpha$",r"$R$"]
        for v,name in zip(cols,names):
            main_ax_artists, sublot_ax_arists = h_real[k].plot_ratio(
                h_fake[k],
                ax_dict={"main_ax":ax[0,k],"ratio_ax":ax[1,k]},
                rp_ylabel=r"Ratio",
                bar_="blue",
                rp_num_label="Generated",
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar
            )
            ax[0,k].set_xlabel("")
            ax[0,k].patches[1].set_fill(True)
            ax[0,k].ticklabel_format(axis="y",style="scientific",scilimits=(-3,3),useMathText=True)
            ax[0,k].patches[1].set_fc(sns.color_palette()[1])
            ax[0,k].patches[1].set_edgecolor("black")
            ax[0,k].patches[1].set_alpha(self.alpha)
            ax[1,k].set_xlabel(name)
            ax[0,k].set_ylabel("Counts" )
            ax[1,k].set_ylabel("Ratio")
            ax[0,k].patches[0].set_lw(2)
            ax[0,k].get_legend().remove()
            xticks=[int(h_real[k].axes[0].edges[-1]//4*i) for i in range(0,int(5))]

            ax[1,k].set_xticks(np.array(xticks),np.array(xticks))
            ax[0,k].set_xticks(np.array(xticks))

            ax[1,k].set_ylim(0.75,1.25)
            k+=1
        if not weighted:
            ax[0,0].set_yscale("log")
        ax[0,leg].legend(loc="best",fontsize=18)
        handles, labels = ax[0,leg].get_legend_handles_labels()
        handles[1]=mpatches.Patch(color=sns.color_palette()[1], label='The red data')
        ax[0,leg].legend(handles, labels)
        plt.tight_layout(pad=0.2)
        self.summary.log_image("{}ratio".format("weighted " if weighted else "unweighted "), [fig],self.step)
        plt.close()

    def plot_jet(self, h_real, h_fake, leg=-1):
        # This creates a histogram of the inclusive distributions and calculates the mass of each jet
        # and creates a histogram of that
        # if save, the histograms are logged to tensorboard otherwise they are shown
        # if quantile, this also creates a histogram of a subsample of the generated data,
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i = 0
        k = 0
        fig, ax = plt.subplots(2, 4, gridspec_kw={"height_ratios": [4, 1]}, figsize=self.fig_size4)
        plt.suptitle("All Particles", fontsize=18)
        for v, name in zip(["eta", "phi", "pt", "m"], [r"$\eta^{\tt rel}$", r"$\phi^{\tt rel}$", r"$p_T^{\tt rel}$", r"$m^{\tt rel}$"]):


            main_ax_artists, sublot_ax_arists = h_fake[k].plot_ratio(
            h_real[k],
            ax_dict={"main_ax":ax[0,k],"ratio_ax":ax[1,k]},
            rp_ylabel=r"Ratio",
            bar_="blue",
            rp_num_label="Generated",
            rp_denom_label="Ground Truth",
            rp_uncert_draw_type="line",  # line or bar)
            )
            i += 1
            ax[0, k].set_xlabel("")
            ax[0, k].patches[1].set_fill(True)
            ax[0, k].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3), useMathText=True)
            ax[0, k].patches[1].set_fc(sns.color_palette()[1])
            ax[0, k].patches[1].set_edgecolor("black")
            ax[0, k].patches[1].set_alpha(self.alpha)
            ax[1, k].set_xlabel(name)
            ax[0, k].set_ylabel("Counts")
            ax[1, k].set_ylabel("Ratio")
            ax[0, k].patches[0].set_lw(2)
            ax[0, k].get_legend().remove()
            k += 1
        ax[0, leg].legend(loc="best", fontsize=18)
        handles, labels = ax[0, leg].get_legend_handles_labels()
        ax[0, -1].locator_params(nbins=4, axis="x")
        ax[1, -1].locator_params(nbins=4, axis="x")
        handles[1] = mpatches.Patch(color=sns.color_palette()[1], label="The red data")
        ax[0, leg].legend(handles, labels)
        plt.tight_layout(pad=1)
        # if not save==None:
        #     plt.savefig(save+".pdf",format="pdf")
        plt.tight_layout()
        try:
            self.summary.log_image("inclusive", [fig], self.step)
            plt.close()
        except:
            plt.show()


    def plot_scores(self,pred_real,pred_fake,train,step):
        fig, ax = plt.subplots()
        bins=30#np.linspace(0,1,10 if train else 100)
        ax.hist(pred_fake, label="Generated", bins=bins, histtype="step")
        if pred_real.any():
            ax.hist(pred_real, label="Ground Truth", bins=bins, histtype="stepfilled",alpha=self.alpha)
        ax.legend()
        ax.patches[0].set_lw(2)
        plt.ylabel("Counts")
        plt.xlabel("Critic Score")
        if self.summary:
            plt.tight_layout()
            if pred_real.any():
                self.summary.log_image("class_train" if train else "class_val", [fig],self.step)
            else:
                self.summary.log_image("class_gen", [fig],self.step)
            plt.close()
        else:
            plt.savefig("plots/scores_"+str(train)+".pdf",format="pdf")
            plt.show()

    def plot_response(self,h_real,h_fake):
        fig,ax=plt.subplots(2,sharex=True)
        h_real.plot_ratio(
                h_fake,
                ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                rp_ylabel=r"Ratio",
                bar_="blue",
                rp_num_label="Generated",
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar
            )
        ax[0].set_xlabel("")
        plt.ylabel("Counts")
        plt.xlabel("Response")
        ax[0].legend()
        if self.summary:
            plt.tight_layout()
            self.summary.log_image("response", [fig],self.step)
            plt.close()
        else:
            plt.savefig("plots/response.pdf",format="pdf")
            plt.show()

def get_hists(bins,mins,maxs,calo=False):
    hists={}
    hists["hists_real"] = []
    hists["hists_fake"] = []


    if calo:
        hists["weighted_hists_real"] = []
        hists["weighted_hists_fake"] = []
        hists["response_real"]=hist.Hist(hist.axis.Regular(100, 0.6, 1.1))
        hists["response_fake"]=hist.Hist(hist.axis.Regular(100, 0.6, 1.1))

        hists["hists_real"].append(hist.Hist(hist.axis.Regular(100, 0, 6500)))
        hists["hists_fake"].append(hist.Hist(hist.axis.Regular(100, 0, 6500)))

        for n in bins[1:]:
            hists["hists_real"].append(hist.Hist(hist.axis.Integer(0, n)))
            hists["hists_fake"].append(hist.Hist(hist.axis.Integer(0, n)))
        for n in bins[1:]:
            hists["weighted_hists_real"].append(hist.Hist(hist.axis.Integer(0, n)))
            hists["weighted_hists_fake"].append(hist.Hist(hist.axis.Integer(0, n)))
    else:
            for n,mi,ma in zip(bins,mins,maxs):
                hists["hists_real"].append(hist.Hist(hist.axis.Regular(n,mi, ma)))
                hists["hists_fake"].append(hist.Hist(hist.axis.Regular(n,mi, ma)))
    return hists


class MultiheadL2Attention(nn.Module):
    def __init__(self, embed_size, num_heads, kdim=None, vdim=None):
        super(MultiheadL2Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads


        # Ensure that the embedding size is a multiple of the number of heads
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size should be divisible by number of heads"

        # Input linear layers
        self.q_linear = spectral_norm(nn.Linear(embed_size, embed_size, bias=False))
        self.k_linear = spectral_norm(nn.Linear(embed_size if not kdim else kdim, embed_size,  bias=False))
        self.v_linear = spectral_norm(nn.Linear(embed_size if not vdim else vdim, embed_size, bias=False))

        # Output linear layer
        self.out = spectral_norm(nn.Linear(embed_size, embed_size, bias=False))

    def forward(self, Q, K, V, key_padding_mask=None, need_weights=False):
        batch_size = Q.shape[0]

        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)

        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        l2_distances = torch.norm(Q.unsqueeze(3) - K.unsqueeze(2), dim=-1)
        l2_distances = -l2_distances

        if key_padding_mask is not None:
            l2_distances = l2_distances.masked_fill(key_padding_mask == 1, float("-inf"))

        attention_scores = F.softmax(l2_distances, dim=-1)
        attention_output = torch.matmul(attention_scores, V)

        # Sum across the seq_length_K dimension to get the final representation of Q
        attention_output = attention_output.sum(dim=2)  # Sum across the K sequence dimension

        # Reshape for the final output linear layer
        attention_output = attention_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        return self.out(attention_output), None
