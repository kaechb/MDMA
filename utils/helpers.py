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
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from my_cmaps import cmap
from scipy.stats import boxcox
def create_mask(n, size=None):
    # Ensure n is a 1D tensor
    n = n.flatten()
    if size is None:
        size = n.max().int().item()
    # Create a range tensor [0, 1, 2, ..., size-1]
    range_tensor = torch.arange(size).unsqueeze(0).to(n.device)

    # Compare range_tensor with n to create the mask
    mask = range_tensor >= n.unsqueeze(1)

    return mask
def fit_kde(n,m,n_max=30):

    from sklearn.neighbors import KernelDensity
    kde=KernelDensity(bandwidth="scott").fit(n.unsqueeze(1).numpy())
    nhat=kde.sample(100000)
    nhat=nhat[nhat<n_max+1]
    mass_distributions = {int(i):m[n==i] for i in range(n.min().int(),n.max().int()+1)}
    mass_kdes ={}

    for i in mass_distributions.keys():
        try:
            mass_kdes[i]=KernelDensity(bandwidth=1e-3).fit(mass_distributions[i].unsqueeze(1).numpy())
        except:
            mass_kdes[i]=KernelDensity(bandwidth=1e-3).fit(mass_distributions[i+1].unsqueeze(1).numpy())



    return kde,mass_kdes
def sample_kde(n,n_kde,m_kde=False):
    #fit kde
    nhat=n_kde.sample(n)
    np.random.shuffle(nhat)
    if m_kde:
        nhat_hist,bins=np.histogram(nhat+0.01,bins=np.arange(0,31),density=False)
        nhat_hist,bins=torch.tensor(nhat_hist),torch.tensor(bins)
        nhat_hist,bins=nhat_hist[nhat_hist>0],bins[1:][nhat_hist>0]

        n_dict={int(i):j for i,j in zip(bins,nhat_hist)}
        ms=[torch.from_numpy(m_kde[int(i)].sample(n_dict[int(i)])) for i in bins]
        mhat=torch.cat(ms).numpy()
        np.random.shuffle(mhat)

        return torch.from_numpy(nhat),torch.from_numpy(mhat)
    else:
        return nhat

class plotting_thesis():
    '''This is a class that takes care of  plotting steps in the script,
        It is initialized with the following arguments:
        true=the simulated data, note that it needs to be scaled
        gen= Generated data , needs to be scaled
        step=The current step of the training, this is need for tensorboard
        model=the model that is trained, a bit of an overkill as it is only used to access the losses
        config=the config used for training
        logger=The logger used for tensorboard logging'''
    def __init__(self,step=None,logger=None,weight=1,big=False):

        self.step=step
        self.weight=weight
        self.big=big
        self.fig_size1=[6.4, 6.4]
        self.fig_size2=[2*6.4, 2*6.4]
        self.fig_size3=[3*6.4, 6.4]
        self.fig_size4=[4*6.4, 6.4]
        self.alpha=.3
        mpl.rcParams['lines.linewidth'] = 2
        font = { "size": 18}#"family": "normal",
        mpl.rc("font", **font)
        mpl.rc('lines', linewidth=2)
        # sns.set_palette("Pastel1")
        self.FONTSIZE=20
        if logger is not None:
            self.summary=logger
        else:
            self.summary = None

    def _adjust_legend(self, ax, leg):
        # Adjust the legend on the plot
        if leg >= 0:
            ax.legend(loc="best", fontsize=self.FONTSIZE-5)
            handles, labels = ax.get_legend_handles_labels()
            handles[0] = mpatches.Patch(color=sns.color_palette()[0], label="The red data", alpha=0.3)
            ax.legend(handles, labels)

    def plot_ratio_calo(self, h_real, h_fake, weighted=False, leg=-1, model_name="",legend_name=""):

        if legend_name=="":
            legend_name="Generated"

            # Main plot
        FONTSIZE=20
        # Plot variables and their names
        variables = [r"E",r"z",r"alpha",r"R"]
        names = [r"$E$ [MeV]",r"$z$",r"$\alpha$",r"$R$"]
        ticks=[[0,2000,4000,6000],[0,10,20,30,40],[0,5,10,15],[0,2,4,6,8]] if not self.big else [[0,2000,4000,6000],[0,10,20,30,40],[0,10,20,30,40],[0,4,8,12,16]]
        if weighted:
            FONTSIZE=FONTSIZE+3
            fig = plt.figure(figsize=self.fig_size3)

            outer_gs = gridspec.GridSpec(1,3, figure=fig,hspace=0.3,wspace=.3)
            variables=variables[1:]
            names=names[1:]
            ticks=ticks[1:]
        else:
            fig = plt.figure(figsize=self.fig_size2)
            outer_gs = gridspec.GridSpec(2, 2, figure=fig,hspace=0.3,wspace=.3)

        for k, (variable, name,ticks) in enumerate(zip(variables, names,ticks)):
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[k],
                                                        height_ratios=[4, 1], hspace=0.15
                                                        )

            # Create the main and ratio axes within the nested grid
            ax_ratio = fig.add_subplot(inner_gs[1], )
            ax_main = fig.add_subplot(inner_gs[0],sharex=ax_ratio)
            locator=MaxNLocator( nbins=6, prune="both")
            ax_main.yaxis.set_major_locator(locator)

            # Plotting logic
            h_real[k].plot_ratio(
                h_fake[k],
                ax_dict={"main_ax": ax_main, "ratio_ax": ax_ratio},
                rp_ylabel="Ratio",
                bar_="blue",
                rp_num_label="GEANT4",
                rp_denom_label=legend_name,
                rp_uncert_draw_type="line"
            )
            ax_ratio.set_xlabel(name, fontsize=FONTSIZE)
            ax_main.set_ylabel("Counts", fontsize=FONTSIZE)
            ax_ratio.set_ylabel("Ratio", fontsize=FONTSIZE)
            ax_main.get_legend().remove()
            ax_ratio.set_ylim(0.5, 1.5) if variable=="E" else ax_ratio.set_ylim(0.9, 1.1)
            ax_main.set_xlabel("")

            ax_main.patches[0].set_fill(True)
            ax_main.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
            ax_ratio.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)


            ax_main.set_xticks([])

            ax_ratio.set_xticks(ticks,labels=ticks)

            ax_main.patches[0].set_edgecolor("black")
            ax_main.patches[1].set_linewidth(2)
            ax_main.patches[0].set_alpha(self.alpha)
            ax_main.patches[0].set_lw(2)
            if k==0:
                ax_main.set_yscale("log")
            # ax_ratio.set_xlim(ax_main.get_xlim())
            ax_main.tick_params(axis='x', which='both', length=0, labelbottom=False)

            ax_ratio.tick_params(axis='x', which='both', labelbottom=True)
            if k==leg:
                self._adjust_legend(ax_main, leg)


        plt.tight_layout()
        weighted="_weighted" if weighted else ""
        plt.savefig(f"plots/calo/{model_name}{weighted}.pdf", format="pdf")
        plt.show()
        plt.close()

    def plot_response(self,h_real,h_fake,model_name,legend_name=""):
        if legend_name=="":
            legend_name="Generated"
        fig,ax=plt.subplots(2,sharex=True,height_ratios=[4,1],figsize=self.fig_size1)
        h_real.plot_ratio(
                h_fake,
                ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                rp_ylabel=r"Ratio",
                bar_="blue",
                rp_num_label="GEANT4",
                rp_denom_label=legend_name,
                rp_uncert_draw_type="line",  # line or bar
            )
        ax_main,ax_ratio=ax
        ax_main.set_xlabel("")
        ax_main.set_ylabel("Counts",fontsize=self.FONTSIZE)
        ax[1].set_xlabel("Response",fontsize=self.FONTSIZE)
        ax[1].set_ylabel("Ratio",fontsize=self.FONTSIZE)

        ax_main.legend()
        ax_ratio.set_ylim(0.5, 1.5)



        ax_main.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        ax_ratio.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        ax_main.patches[0].set_fill(True)
        ax_main.patches[0].set_alpha(self.alpha)

        # ax_main.patches[1].set_fc(sns.color_palette()[0])
        # ax_main.patches[0].set_fc(sns.color_palette()[1])
        ax_main.patches[0].set_edgecolor("black")
        ax_main.patches[1].set_linewidth(2)




        ax_main.patches[0].set_lw(2)
        # ax_ratio.set_xlim(ax_main.get_xlim())
        ax_main.tick_params(axis='x', which='both', length=0, labelbottom=False)

        ax_ratio.tick_params(axis='x', which='both', labelbottom=True)
        fig.tight_layout(pad=0.3)
        fig.savefig(f"plots/calo/response_{model_name}.pdf",format="pdf")
        fig.show()

    def plot_calo_multiple(self, h_real, h_fake, weighted=False, leg=-1, model_name="",legend_name="",group_name="",groups={},raw=False,sample_n=False):
        if legend_name=="":
            legend_name="Generated"

            # Main plot
        FONTSIZE=20
        # Plot variables and their names
        variables = [r"E",r"z",r"alpha",r"R"]
        names = [r"$E$ [MeV]",r"$z$",r"$\alpha$",r"$R$"]
        ticks=[[0,2000,4000,6000],[0,10,20,30,40],[0,5,10,15],[0,2,4,6,8]] if not self.big else [[0,2000,4000,6000],[0,10,20,30,40],[0,10,20,30,40],[0,4,8,12,16]]
        if weighted:
            variables=variables[1:]
            names=names[1:]
            ticks=ticks[1:]
        if groups[group_name] ==[] :

            if weighted:

                fig = plt.figure(figsize=self.fig_size3)
                FONTSIZE=FONTSIZE+3
                outer_gs = gridspec.GridSpec(1,3, figure=fig,hspace=0.3,wspace=.3)

            else:
                fig = plt.figure(figsize=self.fig_size2)

                outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=.3)
        else:
            # Assuming fig and outer_gs are provided for adding plots to existing axes
            fig=groups[group_name][0]
            print(fig.axes)
            pass

        current_cycler = plt.rcParams['axes.prop_cycle']

        # Convert to a list and access the second color
        colors = list(current_cycler)
        # Iterate through the provided variables and plot
        for idx, (variable, name) in enumerate(zip(variables, names)):
            if len(groups[group_name])>0:  # Check if this is a subsequent call

                print(group_name,fig.axes)
                if weighted and idx==3:
                    break
                ax_main = fig.axes[idx * 2]
                ax_ratio = fig.axes[idx * 2 + 1]


            else:
                inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[idx], height_ratios=[4, 1], hspace=0.15)
                ax_main = fig.add_subplot(inner_gs[0])
                ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)

            # Example plot commands, replace with your actual data and plotting
            if len(groups[group_name])==0:  # Check if this is a subsequent call

                hep.histplot(h_real[idx], ax=ax_main, label="GEANT4", histtype='fill', alpha=self.alpha)
            hep.histplot(h_fake[idx], ax=ax_main, label=legend_name, histtype='step')

            # Calculating ratio: Assume h_real and h_fake have compatible binning and can be directly divided
            ratio = h_fake[idx].values()/ h_real[idx].values()
            bins = h_real[idx].axes[0].edges  # Assuming 1D histograms and compatible binning

            # Plotting the ratio as a bar plot
            # Calculate bin centers and width for the bar plot

            for i in range(len(bins)-1):
                bin_start = bins[i]
                bin_end = bins[i+1]
                ratio_value = ratio[i]
                # Draw a horizontal line for each bin
                ax_ratio.hlines(ratio_value, bin_start, bin_end, lw=2,color=colors[1+len(groups[group_name])]['color'])
                if i < len(bins) - 2:  # Check to avoid index out of bounds
                    next_ratio_value = ratio[i+1]
            # Vertical line connecting to the next bin
                    ax_ratio.vlines(bin_end, ratio_value, next_ratio_value, lw=2,color=colors[1+len(groups[group_name])]['color'])
            #ax_ratio.bar(bin_centers, ratio, width=bin_widths, align='center', fill=False,linestyle=linestyles[self.counter],label=legend_name)

            # Setting labels and styles
            ax_ratio.set_xlabel(name, fontsize=FONTSIZE)
            ax_main.set_ylabel("Counts" if not weighted else "E [MeV]", fontsize=FONTSIZE)
            ax_ratio.set_ylabel("Ratio", fontsize=FONTSIZE)

            ax_ratio.set_ylim(0.5, 1.5) if variable=="E" else ax_ratio.set_ylim(0.9, 1.1)
            ax_main.set_xlabel("")

            ax_main.patches[0].set_fill(True)
            if groups[group_name]==[]:
                ax_main.tick_params(axis='x', which='both', length=0, labelbottom=False)

                ax_main.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
                ax_ratio.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
                ax_main.tick_params(axis='x', which='both', length=0, labelbottom=False)

                ax_ratio.tick_params(axis='x', which='both', labelbottom=True)
            ax_main.set_xticks([])
            ax_ratio.set_xticks(ticks[idx],labels=ticks[idx])

            ax_main.patches[0].set_edgecolor("black")
            ax_main.patches[1].set_linewidth(2)
            ax_main.patches[0].set_alpha(self.alpha)
            ax_main.patches[0].set_lw(2)
            if idx==0 and not weighted:
                ax_main.set_yscale("log")
            # ax_ratio.set_xlim(ax_main.get_xlim())

            if idx==leg:
                self._adjust_legend(ax_main, leg)


        fig.tight_layout()
        weighted="_weighted" if weighted else ""
        if self.big:
            weighted+="_big"
        weighted+="_raw" if raw else ""
        weighted+="_sample_n" if sample_n else ""
        path="plots/calo/"


        os.makedirs(path, exist_ok=True)

        fig.savefig(path+f"calo{weighted}.pdf", format="pdf")




        return fig

    def plot_response_multiple(self, h_real, h_fake, weighted=False, leg=-1, model_name="",legend_name="",group_name="",groups={},raw=False,sample_n=False):
        if legend_name=="":
            legend_name="Generated"

            # Main plot
        print(groups,group_name)
        current_cycler = plt.rcParams['axes.prop_cycle']

        # Convert to a list and access the second color
        colors = list(current_cycler)

        # Plot variables and their names


        # Create or reuse figure and gridspec
        if groups[group_name] ==[] :
            print("Creating new figure")

            fig,ax=plt.subplots(nrows=2,sharex=True,height_ratios=[4,1],figsize=self.fig_size1)
            ax_main = ax[0]
            ax_ratio = ax[1]

        else:
            # Assuming fig and outer_gs are provided for adding plots to existing axes
            fig=groups[group_name][0]
            ax_main = fig.axes[0]
            ax_ratio = fig.axes[ 1]

        # Iterate through the provided variables and plot
# Check if this is a subsequent call



        # Example plot commands, replace with your actual data and plotting
        if len(groups[group_name])==0:  # Check if this is a subsequent call

            hep.histplot(h_real, ax=ax_main, label="GEANT4", histtype='fill', alpha=self.alpha)
        hep.histplot(h_fake, ax=ax_main, label=legend_name, histtype='step')

        # Calculating ratio: Assume h_real and h_fake have compatible binning and can be directly divided
        ratio = h_fake.values()/ h_real.values()
        bins = h_real.axes[0].edges  # Assuming 1D histograms and compatible binning

        # Plotting the ratio as a bar plot
        # Calculate bin centers and width for the bar plot
        for i in range(len(bins)-1):
            bin_start = bins[i]
            bin_end = bins[i+1]
            ratio_value = ratio[i]
            # Draw a horizontal line for each bin
            ax_ratio.hlines(ratio_value, bin_start, bin_end, lw=2,color=colors[1+len(groups[group_name])]['color'])
            if i < len(bins) - 2:  # Check to avoid index out of bounds
                next_ratio_value = ratio[i+1]
        # Vertical line connecting to the next bin
                ax_ratio.vlines(bin_end, ratio_value, next_ratio_value, lw=2,color=colors[1+len(groups[group_name])]['color'])
        #ax_ratio.bar(bin_centers, ratio, width=bin_widths, align='center', fill=False,linestyle=linestyles[self.counter],label=legend_name)

            # Setting labels and styles

        ax_main.set_xlabel("")
        ax_main.set_ylabel("Counts",fontsize=self.FONTSIZE)
        ax_ratio.set_xlabel("Response",fontsize=self.FONTSIZE)
        ax_ratio.set_ylabel("Ratio",fontsize=self.FONTSIZE)
        ax_main.set_yscale("log")
        ax_main.legend()
        ax_ratio.set_ylim(0.5, 1.5)




        ax_main.patches[0].set_fill(True)
        ax_main.patches[0].set_alpha(self.alpha)

        ax_main.patches[0].set_edgecolor("black")
        ax_main.patches[1].set_linewidth(2)
        ax_main.patches[0].set_lw(2)
        if groups[group_name]==[]:
                ax_main.tick_params(axis='x', which='both', length=0, labelbottom=False)

                #ax_main.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
                #ax_ratio.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

        fig.tight_layout(pad=0.3)
        path="plots/calo/"
        path+="raw_" if raw else ""
        path+="sample_n" if sample_n else ""
        os.makedirs(path, exist_ok=True)
        if not self.big:
            fig.savefig(path+f"response.pdf", format="pdf")
        else:
            fig.savefig(path+f"response_big.pdf", format="pdf")
        return fig

    def plot_ratio(self,h_real,h_fake,weighted,leg=-1,model_name="",n_part=30,log=False,legend_name="",outer_gs=None,fig=None):
        i = 0
        k = 0
        FONTSIZE=20
        fig = plt.figure(figsize=self.fig_size2)
        if legend_name=="":
            legend_name="Generated"


            # Main plot

        # Plot variables and their names
        variables = ["eta", "phi", "pt", "m"]
        names = [r"$\eta^{\mathrm{rel}}$", r"$\phi^{\mathrm{rel}}$", r"$p_{\mathrm{T}}^{\mathrm{rel}}$", r"$m^{\mathrm{rel}}$"]

        if outer_gs==None:
            outer_gs = gridspec.GridSpec(2, 2, figure=fig,hspace=0.3,wspace=.3)

        for k, (variable, name) in enumerate(zip(variables, names)):
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[k],
                                                        height_ratios=[4, 1], hspace=0.15
                                                        )

            # Create the main and ratio axes within the nested grid
            ax_main = fig.add_subplot(inner_gs[0])
            ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
            ax_main.tick_params(axis='x', which='both', length=0, labelbottom=False)
            # Plotting logic
            h_real[k].plot_ratio(
                h_fake[k],
                ax_dict={"main_ax": ax_main, "ratio_ax": ax_ratio},
                rp_ylabel="Ratio",
                bar_="blue",
                rp_num_label="PYTHIA",
                rp_denom_label=legend_name,
                rp_uncert_draw_type="line"
            )
            ax_ratio.set_xlabel(name, fontsize=FONTSIZE)
            ax_main.set_ylabel("Counts", fontsize=FONTSIZE)
            ax_ratio.set_ylabel("Ratio", fontsize=FONTSIZE)
            ax_main.get_legend().remove()
            ax_ratio.set_ylim(0.5, 1.5)
            ax_ratio.set_yticks([0.5, 1, 1.5])
            ax_main.set_xlabel("")
            ax_main.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
            # ax_ratio.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
            # locator=MaxNLocator( nbins=6, prune="both")
            # ax_main.yaxis.set_major_locator(locator)
            ax_main.patches[0].set_fill(True)
            ax_main.patches[0].set_alpha(self.alpha)
            if k!=3:
            #if k==2 or log and k!=3:
                print("set log scale")
                ax_main.set_yscale("log")
            # ax_main.patches[1].set_fc(sns.color_palette()[0])
            # ax_main.patches[0].set_fc(sns.color_palette()[1])
            ax_main.patches[0].set_edgecolor("black")
            ax_main.patches[1].set_linewidth(2)
            # ax_main.patches[0].set_lw(2)
            ax_ratio.set_xlim(ax_main.get_xlim())
            if k==1:
                self._adjust_legend(ax_main, leg)


        # plt.suptitle("All Particles", fontsize=18)

       # plt.suptitle("Agreement between Ground Truth and Generated Data", fontsize=28, fontweight="bold")
        plt.tight_layout(pad=0.3)
        os.makedirs("plots/jetnet{}".format(n_part),exist_ok=True)
        # if not save==None:
        plt.savefig("plots/jetnet{}/{}_jetnet.pdf".format(n_part,model_name),format="pdf")

        plt.show()
        plt.close()

    def plot_ratio_multiple(self, h_real, h_fake, weighted, leg=-1, model_name="", n_part=30, log=False, legend_name="",group_name="",groups={},boxcox=False):
        FONTSIZE = 20
        linestyles=[":","--","-."]
        edgecolors=["black","darkgrey","grey"]
        current_cycler = plt.rcParams['axes.prop_cycle']

        # Convert to a list and access the second color
        colors = list(current_cycler)
        second_color = colors[1]['color']
        if legend_name == "":
            legend_name = "Generated"

        # Plot variables and their names
        variables = ["eta", "phi", "pt", "m"]
        names = [r"$\eta^{\mathrm{rel}}$", r"$\phi^{\mathrm{rel}}$", r"$p_{\mathrm{T}}^{\mathrm{rel}}$", r"$m^{\mathrm{rel}}$"]

        # Create or reuse figure and gridspec
        if groups[group_name] ==[] :

            fig = plt.figure(figsize=self.fig_size2)
            outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=.3)
        else:
            # Assuming fig and outer_gs are provided for adding plots to existing axes
            fig=groups[group_name][0]
            pass
        axes_list = []  # Store axes for reuse
        # Iterate through the provided variables and plot
        for idx, (variable, name) in enumerate(zip(variables, names)):
            if len(groups[group_name])>0:  # Check if this is a subsequent call
                ax_main = fig.axes[idx * 2]
                ax_ratio = fig.axes[idx * 2 + 1]
            else:
                inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[idx], height_ratios=[4, 1], hspace=0.15)
                ax_main = fig.add_subplot(inner_gs[0])
                ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
                axes_list.append((ax_main, ax_ratio))
            ax_main.tick_params(axis='x', which='both', length=0, labelbottom=False)

            # Example plot commands, replace with your actual data and plotting
            if len(groups[group_name])==0:  # Check if this is a subsequent call

                hep.histplot(h_real[idx], ax=ax_main, label="PYTHIA", histtype='fill', alpha=self.alpha)
            hep.histplot(h_fake[idx], ax=ax_main, label=legend_name, histtype='step')

            # Calculating ratio: Assume h_real and h_fake have compatible binning and can be directly divided
            ratio = h_fake[idx].values()/ h_real[idx].values()
            bins = h_real[idx].axes[0].edges  # Assuming 1D histograms and compatible binning

            # Plotting the ratio as a bar plot
            # Calculate bin centers and width for the bar plot

            for i in range(len(bins)-1):
                bin_start = bins[i]
                bin_end = bins[i+1]
                ratio_value = ratio[i]
                # Draw a horizontal line for each bin
                ax_ratio.hlines(ratio_value, bin_start, bin_end, lw=2,color=colors[1+len(groups[group_name])]['color'])
                if i < len(bins) - 2:  # Check to avoid index out of bounds
                    next_ratio_value = ratio[i+1]
            # Vertical line connecting to the next bin
                    ax_ratio.vlines(bin_end, ratio_value, next_ratio_value, lw=2,color=colors[1+len(groups[group_name])]['color'])
            #ax_ratio.bar(bin_centers, ratio, width=bin_widths, align='center', fill=False,linestyle=linestyles[self.counter],label=legend_name)

            # Setting labels and styles
            ax_ratio.set_xlabel(name, fontsize=FONTSIZE)
            ax_main.set_ylabel("Counts", fontsize=FONTSIZE)
            ax_ratio.set_ylabel("Ratio", fontsize=FONTSIZE)
            if len(groups[group_name])==0:
                ax_main.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
                ax_ratio.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
            ax_ratio.set_ylim(0.5, 1.5)  # Adjust as necessary
            ax_ratio.set_yticks([0.5, 1, 1.5])
            ax_main.set_xlabel("")
            if idx!=3:
                ax_main.set_yscale("log")
            if idx==2:
                self._adjust_legend(ax_main, leg)
        fig.tight_layout()
        # Get the current color cycle

        path="plots/jetnet{}".format(n_part)
        if boxcox:
            path+="/boxcox"
        os.makedirs(path, exist_ok=True)
        fig.savefig(path+"/{}_jetnet.pdf".format( group_name), format="pdf")
        return fig

    def plot_corr(self,real,fake,model,leg=-1):
        # Sample data: batch_size of 100, 30 particles, 3 features each


        def compute_correlation_matrix(tensor):
     # Sum over the angular and radial layers


            # Compute the correlation matrix across the batch dimension
            correlation_matrix = np.corrcoef(tensor, rowvar=False)


            return correlation_matrix
        diffs=[]
        lims=[]
        for name,data in zip(["Ground Truth","Generated"],[real,fake]):
            # Compute correlation for each batch and then average
            correlations = []

            sorted_data, indices = torch.sort(data[:,:,2], dim=1, descending=True)

            # Use the indices to reorder the data
            data = torch.gather(data, 1, indices.unsqueeze(-1).expand(-1, -1, 3)).numpy()
            for feature_idx in range(3):
                correlation_matrix = compute_correlation_matrix(data[:, :, feature_idx])
                np.fill_diagonal(correlation_matrix, np.nan)

                non_ones = correlation_matrix[correlation_matrix <= 0.99]
                if name == "Ground Truth":
                    lims.append((non_ones.min(), non_ones.max()))
                correlations.append(correlation_matrix)
                diffs.append(correlation_matrix)
            # Convert tensors to numpy arrays for plotting
            # Plot heatmaps
            fig, axes = plt.subplots(1, 3, figsize=self.fig_size3)
            # fig.suptitle("Correlations between Particles for {} Data".format(name), fontsize=28, fontweight="bold")
            sns.heatmap(correlations[0], ax=axes[0], cmap='coolwarm', cbar=False,vmin=lims[0][0],vmax=lims[0][1])
            axes[0].set_title(r'$\eta^{\text{rel}}$',fontsize=self.FONTSIZE+10, pad=10)

            sns.heatmap(correlations[1], ax=axes[1], cmap='coolwarm', cbar=False,vmin=lims[1][0],vmax=lims[1][1])
            axes[1].set_title(r'$\phi^{\text{rel}}$',fontsize=self.FONTSIZE+10, pad=10)

            cax3=sns.heatmap(correlations[2], ax=axes[2], cmap='coolwarm',cbar=False,vmin=lims[2][0],vmax=lims[2][1]
                             )
            axes[2].set_title(r'$p_{\mathrm{T}}^{\text{rel}}$',fontsize=self.FONTSIZE+10, pad=10)
            for ax in axes:
                ax.set_xticks([1,10,20,30],)
                ax.set_yticks([1,10,20,30])
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()


                ax.set_yticklabels([1,10,20,30])
                ax.set_xticklabels([1,10,20,30])

                ax.set_xlabel("Particles",fontsize=self.FONTSIZE+5)
                ax.set_ylabel("Particles",fontsize=self.FONTSIZE+5)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(cax3.collections[0], cax=cbar_ax)
            fig.tight_layout(rect=[0, 0, 0.9, 1])
            if name=="Ground Truth":
                fig.savefig("plots/jetnet30/corrGroundTruth.pdf",format="pdf")
            else:
                fig.savefig("plots/jetnet30/"+model+"corr.pdf",format="pdf")

            fig.show()
        # diff=[diffs[0]-diffs[3],diffs[1]-diffs[4],diffs[2]-diffs[5]]

        # fig, axes = plt.subplots(1, 3, figsize=self.fig_size3)
        # fig.suptitle(r"$\Delta$Ground Truth - Generated Data Correlations between Particles", fontsize=28,  fontweight="bold")
        # sns.heatmap(diff[0], ax=axes[0], cmap='coolwarm', cbar=False,vmin=-.1,vmax=.1)
        # axes[0].set_title(r'$\eta^{rel}$')

        # sns.heatmap(diff[1], ax=axes[1], cmap='coolwarm', cbar=False,vmin=-.1,vmax=.1)
        # axes[1].set_title(r'$\phi^{rel}$')

        # cax3=sns.heatmap(diff[2], ax=axes[2], cmap='coolwarm',cbar=False,vmin=-.1,vmax=.1)
        # axes[2].set_title(r'$p_T^{rel}$')
        # for ax in axes:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_xlabel("Particles")
        #     ax.set_ylabel("Particles")
        # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # fig.colorbar(cax3.collections[0], cax=cbar_ax)
        # plt.tight_layout(rect=[0, 0, 0.9, 1])
        # plt.savefig("plots/jetnet30/diff_"+model+name+".pdf",format="pdf")
        # plt.show()


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

    def plot_calo(self,h_real,h_fake,weighted,leg=-1,legend_name=""):
        if legend_name=="":
            legend_name="Generated"
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
                rp_num_label=legend_name,
                rp_denom_label="GEANT4",
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
        try:
            self.summary.log_image("{}ratio".format("weighted " if weighted else "unweighted "), [fig],self.step)
        except:
            plt.show()
        plt.close()

    def plot_jet(self, h_real, h_fake, leg=-1,ema=False):
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
            if k!=3:

                ax[0,k].set_yscale("log")
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
            self.summary.log_image("inclusive_ema" if ema else "inclusive", [fig], self.step)
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
        try:

            plt.tight_layout()
            self.summary.log_image("response", [fig],self.step)
            plt.close()
        except:
            plt.savefig("plots/response.pdf",format="pdf")
            plt.show()

def get_hists(bins,mins,maxs,calo=False,ema=False,min_response=0,max_response=10,simon=False):
    hists={}
    hists["hists_real"] = []
    hists["hists_fake"] = []
    if ema:
        hists["hists_fake_ema"] = []

    if calo:
        hists["weighted_hists_real"] = []
        hists["weighted_hists_fake"] = []
        hists["response_real"]=hist.Hist(hist.axis.Regular(bins[0], 0., 2,underflow=True,overflow=True,flow=True))
        hists["response_fake"]=hist.Hist(hist.axis.Regular(bins[0], 0., 2,underflow=True,overflow=True,flow=True))

        hists["hists_real"].append(hist.Hist(hist.axis.Regular(bins[0], 0, 6500)))
        hists["hists_fake"].append(hist.Hist(hist.axis.Regular(bins[0], 0, 6500)))

        for n in bins[1:]:
            hists["hists_real"].append(hist.Hist(hist.axis.Integer(0, n)))
            hists["hists_fake"].append(hist.Hist(hist.axis.Integer(0, n)))
        for n in bins[1:]:
            hists["weighted_hists_real"].append(hist.Hist(hist.axis.Integer(0, n)))
            hists["weighted_hists_fake"].append(hist.Hist(hist.axis.Integer(0, n)))
    else:
            i=0
            for n,mi,ma in zip(bins,mins,maxs):
                mi-=1e-5
                ma+=1e-5
                hists["hists_real"].append(hist.Hist(hist.axis.Regular(n,mi, ma,underflow=True,overflow=True,flow=True)))
                hists["hists_fake"].append(hist.Hist(hist.axis.Regular(n,mi, ma,underflow=True,overflow=True,flow=True)))

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

import torch
import torch.nn as nn

class BoxCoxTransformer(nn.Module):
    def __init__(self, lambda_param=None, epsilon=1e-6):
        """
        Box-Cox Transformer as a PyTorch module.

        Parameters:
        lambda_param (float): The lambda parameter for the Box-Cox transformation. If None, it will be estimated.
        epsilon (float): Small value to ensure positivity.
        """
        super(BoxCoxTransformer, self).__init__()
        # Lambda could be a learned parameter or set during fitting
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.mean=None
        self.std=None
    def fit(self, x):
        # Placeholder for a fit method if lambda is to be determined from data
        # For true fitting, statistical methods to find lambda should be used
        x_np = x.detach().cpu().numpy().reshape(-1)
        _, fitted_lambda = boxcox(x_np + self.epsilon)  # Adjust with epsilon to ensure positivity
        self.lambda_param = torch.tensor(fitted_lambda, dtype=torch.float32)


    def fit_transform(self, x):
        """
        Fit the Box-Cox transformation to the input data and apply the transformation.

        Parameters:
        x (Tensor): Input data to fit and transform.

        Returns:
        Tensor: Transformed data.
        """
        self.fit(x)
        return self.transform(x)
    def transform(self, x):
        """
        Apply the Box-Cox transformation to the input data.

        Parameters:
        x (Tensor): Input data to transform.

        Returns:
        Tensor: Transformed data.
        """

        x_adj = x + self.epsilon  # Ensure x > 0
        y = (torch.pow(x_adj, self.lambda_param) - 1) / self.lambda_param
        if self.mean is None:
            self.mean=y.mean()
            self.std=y.std()
        y= (y-self.mean)/self.std
        return y

    def inverse_transform(self, y):
        """
        Apply the inverse Box-Cox transformation to the input data.

        Parameters:
        y (Tensor): Input data to inverse transform.

        Returns:
        Tensor: Original data before transformation.
        """
        y = (y*self.std)+self.mean
        x_adj = (y * self.lambda_param) + 1
        x = torch.pow(x_adj, 1 / self.lambda_param) - self.epsilon
        return x

class Nflow(torch.nn.Module):
    def __init__(self,n_mean=0,n_std=1):
        super(Nflow,self).__init__()
        bins=5
        self.flow= zuko.flows.NICE(
            features=1,
            context=1,
            transforms=3,
            univariate=zuko.transforms.MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            hidden_features=[128, 128,128,128]
)
        self.n_mean=torch.nn.Parameter(torch.tensor(n_mean),requires_grad=False)
        self.n_std=torch.nn.Parameter(torch.tensor(n_std),requires_grad=False)
    def forward(self,x):
        return self.flow(x).sample()

class PowerLawModel(torch.nn.Module):
    def __init__(self,coeffs,n_mean=0.,n_std=1.):
        super(PowerLawModel, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(coeffs[0]),requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor(coeffs[1]),requires_grad=True)
        self.c = torch.nn.Parameter(torch.tensor(coeffs[2]),requires_grad=True)
        self.d = torch.nn.Parameter(torch.tensor(coeffs[3]),requires_grad=True)
        self.e = torch.nn.Parameter(torch.tensor(coeffs[4]),requires_grad=True)
        self.f = torch.nn.Parameter(torch.tensor(coeffs[5]),requires_grad=True)

        self.n_std=torch.nn.Parameter(torch.tensor(n_std),requires_grad=False)
        self.n_mean=torch.nn.Parameter(torch.tensor(n_mean),requires_grad=False)
    # def forward(self, x):
    def forward(self, x):
            return (self.a*x**5 +self.b * x**4 + self.c * x**3 + self.d * x**2 + self.e * x**1 + self.f)