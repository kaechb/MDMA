import sys

import torch

from fit import MDMA
from jetnet_dataloader import JetNetDataloader

parton = "q" if len(sys.argv) == 1 else sys.argv[1]
n=10000 if len(sys.argv) == 1 else int(sys.argv[2])
batchsize=3000 if len(sys.argv) == 1 else int(sys.argv[3])
outdir = "out/" if len(sys.argv) == 1 else sys.argv[4]
import pickle

import numpy as np

device="cuda" if torch.cuda.is_available() else "cpu"

ckpt_dict = {"g": "ckpts/g.ckpt", "q": "ckpts/q.ckpt", "t": "ckpts/t.ckpt", "w": "ckpts/w.ckpt", "z": "ckpts/z.ckpt"}
with torch.no_grad():
    print("generating {} jets for {}-initiated jets with batch_size {} on {}".format(n,parton,batchsize,device))
    torch.set_float32_matmul_precision("medium")

    config = {"n_part": 150, "n_dim": 3, "batch_size": 1024, "parton": parton, "smart_batching": True, "n_start": 150}
    data_module = JetNetDataloader(config,)
    data_module.setup("train")
    model = MDMA.load_from_checkpoint(ckpt_dict[parton]).eval()
    model.to(device)
    z = torch.normal(torch.zeros(n, 150, 3), torch.ones(n, 150, 3))
    m = []
    with open("kde/" + config["parton"] + "_kde.pkl", "rb") as f:
        kde = pickle.load(f)
    # sample particle multiplicity per jet
    kde_sample = kde.resample(int(n*1.1)).T  # account for cases were kde_sample is not in [1,150]
    n_sample = np.rint(kde_sample)
    n_sample = n_sample[(n_sample >= 1) & (n_sample <= 150)]
    indices = torch.arange(150, device=device)
    mask = indices.view(1, -1) < torch.tensor(n_sample).to(device).view(-1, 1)
    mask = ~mask.bool()[: len(z)]
    z[mask] = 0
    model.gen_net.cuda()

    fake = torch.cat([model.gen_net(z[i * batchsize : (i + 1) * batchsize].cuda(), mask=mask[i * batchsize : (i + 1) * batchsize].bool().cuda()).cpu() for i in range(n//batchsize+1)], dim=0)
    model.data_module = data_module
    model.data_module.scaler = model.data_module.scaler.to("cpu")
    fake_scaled = model.data_module.scaler.inverse_transform(fake.clone())
    fake_scaled[mask[:len(fake_scaled)]] = 0
    torch.save(fake_scaled,outdir+parton+"_"+str(n)+".pt")

