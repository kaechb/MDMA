
import hist
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from hist import Hist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from torch import nn
from torch.nn import functional as FF


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



mpl.rcParams['lines.linewidth'] = 2

class set_pastel():
    def __init__(self):
        self.fig_size1=[6.4, 6.4]
        self.fig_size2=[2*6.4, 6.4]
        self.fig_size3=[3*6.4, 6.4]
        self.fig_size4=[4*6.4, 6.4]
        mpl.rcParams['lines.linewidth'] = 2
        font = { "size": 18}
        mpl.rc("font", **font)
        mpl.rc('lines', linewidth=2)
        sns.set_palette("Pastel1")
    def fwd(self,):
        pass


class plotting_point_cloud():
    '''This is a class that takes care of  plotting steps in the script,
        It is initialized with the following arguments:
        true=the simulated data, note that it needs to be scaled
        gen= Generated data , needs to be scaled
        step=The current step of the training, this is need for tensorboard
        model=the model that is trained, a bit of an overkill as it is only used to access the losses
        config=the config used for training
        logger=The logger used for tensorboard logging'''
    def __init__(self,true,gen,n_dim, n_part,p,step=None,model=None,logger=None,weight=1,nf=None,n=30):


        self.test_set=true
        self.step=step
        self.model=model
        self.gen=gen
        self.p=p
        self.n_part=n_part
        self.n=n
        self.n_dim=n_dim
        self.weight=weight
        self.fig_size1=[6.4, 6.4]
        self.fig_size2=[2*6.4, 6.4]
        self.fig_size3=[3*6.4, 6.4]
        self.fig_size4=[4*6.4, 6.4]
        self.alpha=0.3
        mpl.rcParams['lines.linewidth'] = 2
        font = { "size": 18}
        mpl.rc("font", **font)
        mpl.rc('lines', linewidth=2)


        sns.set_palette("Pastel1")
        if logger is not None:
            self.summary=logger
        else:
            self.summary = None
    def plot_marginals(self,ith=0,title=None,save=None,leg=-1):
        #This plots the marginal distribution for simulation and generation
        #Note that this is the data the model sees during training as input to model in the NF
        #This is the distribution of one of [eta,phi,pt] of one particle of the n particles per jet: for example the pt of the 3rd particle
        #if save, the histograms are logged to tensorboard otherwise they are shown

        # plt.switch_backend('agg')
        name,label=["eta","phi","pt"],['${{\eta}}^{{\\tt rel}}_{{{}}}$'.format(ith+1),"${{\phi}}^{{\\tt rel}}_{{{}}}$".format(ith+1),"${{p^{{\\tt rel}}_{{T,{}}}}}$".format(ith+1)]
        fig,ax=plt.subplots(2,3,gridspec_kw={'height_ratios': [3, 1]},figsize=self.fig_size3)
        particles=[3*ith,3*ith+1,3*ith+2]
        pre=""
        if ith!=0:
            pre="$2^{{\rm rd}} $ "
        elif ith==1:
            pre="$2^{{\rm nd}}$ "
        plt.suptitle(pre+"Hardest Particle")
        k=0
        for i in particles:
            ax_temp=ax[:,k]
            a=np.quantile(self.test_set[:,i],0)
            b=np.quantile(self.test_set[:,i],1)
            h=hist.Hist(hist.axis.Regular(15,a,b,label=label[i%3],underflow=True,overflow=True))
            h2=hist.Hist(hist.axis.Regular(15,a,b,label=label[i%3],underflow=True,overflow=True))
            h2.fill(self.test_set[:,i])
            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax_temp[0],"ratio_ax":ax_temp[1]},
                rp_ylabel=r"Ratio",
#                 rp_xlabel=label[i%3],
                rp_num_label="Generated",
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar
            )
            ax_temp[0].set_xlabel("")
            ax_temp[0].ticklabel_format(axis="y",style="scientific",scilimits=(-3,3),useMathText=True)
            ax_temp[1].set_ylim(0.5,2)
            ax_temp[0].set_xlim(a,b)
            ax_temp[1].set_xlim(a,b)
            ax_temp[1].set_xlabel(label[i%3])
            ax_temp[0].set_ylabel("Counts")
            ax_temp[1].set_ylabel("Ratio")
            ax[0,k].patches[1].set_fill(True)
            ax[0,k].patches[0].set_lw(2)
            ax[0,k].patches[1].set_fc(sns.color_palette()[1])

            ax[0,k].get_legend().remove()
            #plt.tight_layout(pad=2)
            k+=1
        ax[0,leg].legend(loc="best",fontsize=18)
        handles, labels = ax[0,leg].get_legend_handles_labels()

        handles[1]=mpatches.Patch(color=sns.color_palette()[1], label='The red data')
        ax[0,leg].legend(handles, labels)
        if not self.summary:

             plt.savefig("plots/marginals.pdf",format="pdf")
             plt.close()

        else:
            plt.tight_layout()
            self.summary.log_image("marginals", [fig],self.step)
            plt.close()




    def plot_mass(self,save=None,quantile=False,bins=15,plot_vline=False,title="",leg=-1):
        #This creates a histogram of the inclusive distributions and calculates the mass of each jet
        #and creates a histogram of that
        #if save, the histograms are logged to tensorboard otherwise they are shown
        #if quantile, this also creates a histogram of a subsample of the generated data,
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i=0
        k=0
        fig,ax=plt.subplots(2,4,gridspec_kw={'height_ratios': [4, 1]},figsize=self.fig_size4)
        plt.suptitle("All Particles",fontsize=18)
        with torch.no_grad():
            m_t=mass(self.test_set).numpy()
            m=mass(self.gen).numpy()
        print("max mass: ",m.max())
        for v,name in zip(["eta","phi","pt","m"],[r"$\eta^{\tt rel}$",r"$\phi^{\tt rel}$",r"$p_T^{\tt rel}$",r"$m^{\tt rel}$"]):

            if v!="m":
                a=min(np.quantile(self.gen.reshape(-1,3)[:,i],0.001),np.quantile(self.test_set.reshape(-1,3)[:,i],0.001))
                b=max(np.quantile(self.gen.reshape(-1,3)[:,i],0.999),np.quantile(self.test_set.reshape(-1,3)[:,i],0.999))
                temp=self.test_set[:i]
                h=hist.Hist(hist.axis.Regular(bins,a,b))
                h2=hist.Hist(hist.axis.Regular(bins,a,b))
                h.fill(self.gen.reshape(-1,3)[self.gen.reshape(-1,3)[:,i]!=0,i])
                h2.fill(self.test_set.reshape(-1,3)[self.test_set.reshape(-1,3)[:,i]!=0,i])
                i+=1

            else:
                a=min(np.quantile(m_t,0.001),np.quantile(m,0.001))
                b=max(np.quantile(m_t,0.999),np.quantile(m,0.999))
                a=np.quantile(m_t,0.001)
                b=np.quantile(m_t,0.999)
                h=hist.Hist(hist.axis.Regular(bins,a,b))
                h2=hist.Hist(hist.axis.Regular(bins,a,b))
                #bins = h.axes[0].edges

                h.fill(m)#,weight=1/self.weight)
                h2.fill(m_t)
                temp=m_t

            #hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )

            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0,k],"ratio_ax":ax[1,k]},
                rp_ylabel=r"Ratio",
                bar_="blue",
                rp_num_label="Generated",
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar
            )
            ax[0,k].set_xlabel("")


            # ax[0,k].patches[1].set_fc("orange")
            # ax[0,k].patches[1].set_alpha(0.5)
#                 if quantile and v=="m" and plot_vline:
#                     ax[0,k].hist(m[m_t<np.quantile(m_t,0.1)],histtype='step',bins=bins,alpha=1,color="red",label="10% quantile gen",hatch="/")
#                     ax[0,k].vlines(np.quantile(m_t,0.1),0,np.max(h[:]),color="red",label='10% quantile train')

            #ax[0,k].hist(temp,bins=bins,color="orange",alpha=0.5)
            ax[0,k].patches[1].set_fill(True)
            ax[0,k].ticklabel_format(axis="y",style="scientific",scilimits=(-3,3),useMathText=True)

            ax[0,k].patches[1].set_fc(sns.color_palette()[1])
            ax[0,k].patches[1].set_edgecolor("black")

            ax[0,k].patches[1].set_alpha(self.alpha)

            ax[1,k].set_ylim(0.25,2)
            ax[0,k].set_xlim(a,b)
            ax[1,k].set_xlabel(name)
            ax[1,k].set_xlim(a,b)
            ax[0,k].set_ylabel("Counts" )
            ax[1,k].set_ylabel("Ratio")
            ax[0,k].patches[0].set_lw(2)
            ax[0,k].get_legend().remove()
            k+=1
        ax[0,leg].legend(loc="best",fontsize=18)
        handles, labels = ax[0,leg].get_legend_handles_labels()
        ax[0,-1].locator_params(nbins=4,axis="x")
        ax[1,-1].locator_params(nbins=4,axis="x")
        handles[1]=mpatches.Patch(color=sns.color_palette()[1], label='The red data')
        ax[0,leg].legend(handles, labels)
        plt.tight_layout(pad=1)
        # if not save==None:
        #     plt.savefig(save+".pdf",format="pdf")
        if self.summary:

            plt.tight_layout()
            self.summary.log_image("inclusive", [fig],self.step)
            plt.close()
        else:
            plt.savefig("plots/inclusive_"+self.p+".pdf",format="pdf")
            plt.close()


    def plot_correlations(self, save=True):
        # Plots correlations between all particles for i=0 eta,i=1 phi,i=2 pt
        self.plot_corr(i=0, save=save)
        self.plot_corr(i=1, save=save)
        self.plot_corr(i=2, save=save)

    def plot_corr(
        self, i=0, names=["$\eta^{rel}$", "$\phi^{rel}$", "$p_T$"], save=True
    ):
        if i == 2:
            c = 1
        else:
            c = 0.25
        df_g = pd.DataFrame(self.gen[:, : self.n_dim][:, range(i, self.n*3, 3)])
        df_h = pd.DataFrame(self.test_set[:, : self.n_dim][:, range(i, self.n*3, 3)])

        fig, ax = plt.subplots(ncols=2, figsize=(15, 7.5))
        corr_g = ax[0].matshow(df_g.corr())
        corr_g.set_clim(-c, c)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(corr_g, cax=cax)

        corr_h = ax[1].matshow(df_h.corr())
        corr_h.set_clim(-c, c)
        divider = make_axes_locatable(ax[1])

        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(corr_h, cax=cax2)
        plt.suptitle("{} Correlation between Particles".format(names[i]), fontsize=38)
        ax[0].set_title("Flow Generated", fontsize=34)
        ax[1].set_title("MC Simulated", fontsize=28)
        ax[0].set_xlabel("Particles", fontsize=28,fontweight="bold")
        ax[0].set_ylabel("Particles", fontsize=28,fontweight="bold")
        ax[1].set_xlabel("Particles", fontsize=28,fontweight="bold")
        ax[1].set_ylabel("Particles", fontsize=28,fontweight="bold")
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        title = ["corr_eta", "corr_phi", "corr_pt"]
        if self.summary:
            plt.tight_layout()
            self.summary.self.summary.log_image("corr", [fig],step=self.step)
            plt.close()
        else:
            plt.savefig("plots/"+title[i]+".pdf",format="pdf")
            plt.close()

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
            plt.close()



    def plot_mom(self,step):
        fig, ax = plt.subplots()
        bins=np.linspace(0.7,1.4,30)
        ax.hist(self.gen.reshape(len(self.gen),self.n,3)[:,:,2].sum(1), label="Generated", bins=bins, histtype="step",alpha=1)

        ax.hist(self.test_set.reshape(len(self.test_set),self.n,3)[:,:,2].sum(1), label="Ground Truth", bins=bins, histtype="stepfilled",alpha=self.alpha)

        ax.legend()
        plt.ylabel("Counts")
        plt.xlabel("$\sum p_T^{rel}$")
        if self.summary:

            plt.tight_layout()
            self.summary.log_image("momentum sum", [fig],self.step)
            plt.close()
        else:
            plt.savefig("plots/momentum.pdf",format="pdf")
            plt.close()

