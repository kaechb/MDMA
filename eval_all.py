import os
import sys
import time

import jetnet
import torch

from fit import MDMA
from helpers import center_jets_tensor, mass
from jetnet_dataloader import JetNetDataloader
from plots import mass

name = "best" if len(sys.argv) == 1 else sys.argv[1]
import numpy as np
import pandas as pd
from jetnet.evaluation import fpd, get_fpd_kpd_jet_features, kpd, w1efp, w1m, w1p
from scipy.stats import wasserstein_distance

results_df = pd.DataFrame()
import pickle

if name == "best":
    ckpt_dict = {"g": "ckpts/g.ckpt", "q": "ckpts/q.ckpt", "t": "ckpts/t.ckpt", "w": "ckpts/w.ckpt", "z": "ckpts/z.ckpt"}

n = 20  # determines oversampling factor


for parton in [
    "q",
    "t",
    "z",
    "w",
    "g",
]:  # ,
    with torch.no_grad():
        print(parton)
        torch.set_float32_matmul_precision("medium")
        true = torch.tensor(jetnet.datasets.JetNet.getData(jet_type=parton, split="test", num_particles=150, data_dir="/beegfs/desy/user/kaechben/datasets")[0]).float()  # model.data_module.scaler.inverse_transform(true)

        mask = ~(true[:, :, 3].bool())
        true = true[:, :, :3]

        if name not in ["epic", "in"]:
            config = {"n_part": 150, "n_dim": 3, "batch_size": 1024, "parton": parton, "smart_batching": True, "n_start": 150}
            data_module = JetNetDataloader(config,)
            data_module.setup("train")
            min_=data_module.scaler.transform(true[~mask[:len(true)]])[:,2].min()
            max_=data_module.scaler.transform(true[~mask[:len(true)]])[:,2].max()
            model = MDMA.load_from_checkpoint(ckpt_dict[parton]).eval()
            model.to("cuda")
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_gen = count_parameters(model.gen_net)
            n_dis = count_parameters(model.dis_net)
            z = torch.normal(torch.zeros(n * true.shape[0], true.shape[1], true.shape[2]), torch.ones(n * true.shape[0], true.shape[1], true.shape[2]))
            m = []
            with open("kde/" + config["parton"] + "_kde.pkl", "rb") as f:
                kde = pickle.load(f)
            # sample particle multiplicity per jet
            kde_sample = kde.resample(n * len(mask) + 10000).T  # account for cases were kde_sample is not in [1,150]
            n_sample = np.rint(kde_sample)
            n_sample = n_sample[(n_sample >= 1) & (n_sample <= 150)]
            indices = torch.arange(150, device=mask.device)
            mask = indices.view(1, -1) < torch.tensor(n_sample).view(-1, 1)
            mask = ~mask.bool()[: len(z)]
            z[mask[: len(z)]] = 0
            model.gen_net.cuda()

            start = time.time()
            fake = torch.cat([model.gen_net(z[i * len(true) : (i + 1) * len(true)].cuda(), mask=mask[i *len(true) : (i + 1) * len(true)].bool().cuda()).cpu() for i in range(n)], dim=0)
            print("time pro jet {}".format((time.time() - start) / len(fake)))
            torch.clamp(fake[:,:,2],min_,max_,out=fake[:,:,2])

            model.data_module = data_module
            model.data_module.scaler = model.data_module.scaler.to(true.device)
            fake_scaled = model.data_module.scaler.inverse_transform(fake.clone())
            fake_scaled[mask[:len(fake_scaled)]] = 0

        elif name == "epic":
            n_gen = -1
            n_dis = -1
            true[mask] = 0
            true = center_jets_tensor(true[:, :, :3][..., [2, 0, 1]])[..., [1, 2, 0]]
            true[mask] = 0
            if parton in ["z", "w"]:
                continue

            fake_scaled = np.load("/home/kaechben/EPiC-GAN/{}_epic.npy".format(parton))[..., [1, 2, 0]]
        elif name == "in":
            n_gen = -1
            n_dis = -1
            n = 4
            fake_scaled = torch.tensor(jetnet.datasets.JetNet.getData(jet_type=parton, split="train", num_particles=150, data_dir="/beegfs/desy/user/kaechben/datasets")[0]).float()[:, :, :3]
        w1m_ = w1m(fake_scaled[:, :, :3], true[:, :, :3], num_eval_samples=len(true))

        kpd_real = get_fpd_kpd_jet_features(true, efp_jobs=20)
        kpd_fake = get_fpd_kpd_jet_features(fake_scaled[: len(true)], efp_jobs=20)

        kpd_ = kpd(kpd_real, kpd_fake)
        fpd_ = fpd(kpd_real, kpd_fake, min_samples=10000, max_samples=len(true))
        data_ms = mass(true).numpy()
        i = 0
        w_dist_list = []
        for _ in range(n):
            gen_ms = mass(fake_scaled[i : i + len(true)]).numpy()
            i += len(true)
            w_dist_ms = wasserstein_distance(data_ms, gen_ms)
            w_dist_list.append(w_dist_ms)
        w1m_2 = np.mean(np.array(w_dist_list))
        w1m_2std = np.std(np.array(w_dist_list))

        w1efp_ = w1efp(fake_scaled[: len(true)], true[:, :, :3], num_eval_samples=len(true), efp_jobs=20)
        w1p_ = w1p(fake_scaled, true[:, :, :3], num_eval_samples=len(true))

        config = {"name": [parton], "model": [name], "w1m": [w1m_], "w1p": [w1p_], "w1efp": [w1efp_], "ngen": [n_gen], "ndis": [n_dis], "w1m_2": [(w1m_2, w1m_2std)], "kpd": [kpd_], "fpd": [fpd_]}
        print(config)
        results_df = pd.concat([results_df, pd.DataFrame(config)], ignore_index=True)
print(results_df)
results_df.to_csv("results/{}.csv".format(name))
