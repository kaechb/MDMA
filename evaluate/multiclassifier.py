import sys
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import os
from my_cmaps import cmap
from matplotlib.colors import LinearSegmentedColormap
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pytorch_lightning as pl
import torch.nn.functional as F
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)

# Add the parent directory to sys.path
sys.path.append(parent_directory)
from models.multiclass import Disc
import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
import pytorch_lightning as pl
import random
from pytorch_lightning.loggers import WandbLogger
datadir="/beegfs/desy/user/kaechben/thesis/jetnet30"
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import jetnet
from pytorch_lightning.callbacks import EarlyStopping

import torch.utils.data as data_utils
class MyDataModule(pl.LightningDataModule):
    def __init__(self, datadir,names,files, test_split=0.2,val_split=0.1):
        super().__init__()
        self.datadir = datadir
        self.names=names
        self.files=files
        self.test_split = test_split
        self.val_split = val_split

    def prepare_data(self):
        pass

    def transform(self, x):
        return (x-self.mean)/self.std

    def inverse_transform(self, x):
        return x*self.std+self.mean

    def setup(self, stage=None):
        datasets = []
        data_dict = {}
        real_test=torch.cat([torch.tensor(jetnet.datasets.JetNet.getData(jet_type="t",split="test",num_particles=30,data_dir="/gpfs/dust/maxwell/user/kaechben/datasets")[0]),torch.tensor(jetnet.datasets.JetNet.getData(jet_type="t",split="valid",num_particles=30,data_dir="/gpfs/dust/maxwell/user/kaechben/datasets")[0])])[:50000]


        self.real_test=real_test[:,:,:3]
        self.mask=self.real_test[:,:,-1]==0
        self.cond=(~self.mask).float().sum(1).unsqueeze(1).unsqueeze(1)
        self.avg_n=self.cond.mean()
        self.cond/=self.avg_n
        self.std=self.real_test[~self.mask].std(0)
        self.mean=self.real_test[~self.mask].mean(0)
        self.real_test=self.transform(self.real_test.float())

        for i, file in enumerate(self.files):
            data = torch.load(os.path.join(self.datadir, file))
            data_dict[file] = data

            inputs = self.transform(data.float())
            mask = data[:, :, 2] == 0
            cond = (~mask).float().sum(1).unsqueeze(1).unsqueeze(1)/self.avg_n
            labels = torch.tensor([i] * len(inputs))
            dataset = torch.utils.data.TensorDataset(inputs, labels, mask, cond)
            datasets.append(dataset)
        combined_dataset = ConcatDataset(datasets)

        # Split the dataset into train and test sets
        test_size = int(len(combined_dataset) * self.test_split)
        val_size = int(len(combined_dataset) * self.val_split)
        train_size = len(combined_dataset) - test_size - val_size
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size, test_size])


        self.train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=15)
        self.test_dl = DataLoader(test_dataset, batch_size=128, shuffle=False,num_workers=15)
        self.val_dl = DataLoader(valid_dataset, batch_size=128, shuffle=False,num_workers=15)

    def train_dataloader(self):
        return self.train_dl


    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

class MultiClassifier(pl.LightningModule):
    def __init__(self, datadir, num_classes,names,files):
        super().__init__()
        self.names=names
        self.files=files
        self.datadir = datadir
        self.num_classes = num_classes
        self.model = Disc(n_dim=3, l_dim=128, hidden=512, num_layers=4,heads= 32,dropout= 0.2,cond_dim= 1,weightnorm= False,cloudnorm= False, out_dim=num_classes,glu=False)
        self.criterion = nn.CrossEntropyLoss()

    def load_datamodule(self, datamodule):
        self.datamodule = datamodule
        self.model.avg_n = self.datamodule.avg_n

    def forward(self, inputs, mask, cond):
        return self.model(inputs, mask, cond)[0]

    def training_step(self, batch, batch_idx):
        inputs, labels, mask, cond = batch

        outputs = self.forward(inputs, mask, cond).squeeze(1)
        loss = self.criterion(outputs, labels)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0001)

        return optimizer


    def on_validation_epoch_start(self):
        self.model.eval()

    def validation_step(self, batch, batch_idx):
        inputs, labels, mask, cond = batch

        outputs = self.forward(inputs, mask, cond).squeeze(1)
        loss = self.criterion(outputs, labels)

        self.log('val_loss', loss)
        accuracy = (outputs.argmax(1) == labels).float().mean()
        self.log('val_acc', accuracy)
        return loss

    def on_test_start(self):
        self.preds=[]
        self.logits=[]
        self.labels=[]
        self.model.eval()
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs, labels, mask, cond = batch

            outputs = F.softmax(self.forward(inputs, mask, cond).squeeze(1))
            loss = self.criterion(outputs, labels)
            self.log('test_loss', loss)

            _, predicted = torch.max(outputs, 1)
            self.preds.append(predicted)
            self.labels.append(labels)
            self.logits.append(outputs)
        return loss

    def on_test_end(self):
        self.preds=torch.cat(self.preds)
        self.logits=torch.cat(self.logits)
        self.labels=torch.cat(self.labels)
        self.model.eval()
        cm = confusion_matrix(self.labels.cpu().numpy(), self.preds.cpu().numpy())

        # Plot the confusion matrix
        fig=plt.figure(figsize=(6.4*3, 6.4*3))
        plt.imshow(cm, cmap='Blues')
        plt.xlabel('Predicted Labels', fontsize=20*3)
        plt.ylabel('True Labels', fontsize=20*3)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j]/cm.sum(1)[i], '.2f'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black", fontdict={"size":30})
        plt.xticks(np.arange(len(self.names)), self.names, rotation=90, fontsize=15*3)
        plt.yticks(np.arange(len(self.names)), self.names, fontsize=15*3)
        plt.tight_layout()
        self.logger.log_image("confusion_matrix", [fig])
        import os
        os.makedirs("plots/jetnet30",exist_ok=True)
        plt.savefig("plots/jetnet30/confusion_matrix.png")
        plt.close()
        with torch.no_grad():
            self.model.cpu()
            outputs = self.forward(self.datamodule.real_test.cpu(), self.datamodule.mask.cpu(), self.datamodule.cond.cpu())
            predicted = torch.nn.functional.softmax(outputs.squeeze(1)).mean(0)
            fig=plt.figure(figsize=(6.4,6.4))
            plt.ylabel(r"$p(\rm data)$",fontsize=20)
            plt.bar(np.arange(len(self.names)),predicted.cpu().numpy())
            plt.xticks(np.arange(len(self.names)), labels=self.names, rotation=45)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.tight_layout()
            self.logger.log_image("predicted_all", [fig])
            plt.savefig("plots/jetnet30/predicted_all.png")

files=os.listdir(datadir)

names=["ipf","apf","pf","nf","cnf","ccnf","tnf","tf"]
files=["t_"+x+".pt" for x in names]
replace_dict={"tf":"TGAN","ccnf":"NF(cc)","cnf":"NF(c)","ipf":"IPF","nf":"NF","apf":"APF","pf":"PF","tnf":"TNF","tNF":"TNF"}
names=[replace_dict[name] for name in names]

# Define the model
logger = WandbLogger(
        save_dir="/beegfs/desy/user/kaechben/thesis/eval_jetnet30/",
        sync_tensorboard=False,
        project="jetnet30_eval"
    )
# Define the Lightning Trainer
trainer = pl.Trainer(
 # Set the number of GPUs to use
    logger=logger,
    max_epochs=1000,  # Set the maximum number of epochs
    check_val_every_n_epoch=10,
    log_every_n_steps=100,
    callbacks=[pl.callbacks.ModelCheckpoint(dirpath='/beegfs/desy/user/kaechben/thesis/eval_jetnet30/checkpoints', filename='model-{epoch:02d}-{val_loss:.2f}',monitor='val_loss', mode='min'),EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")],
    default_root_dir="/beegfs/desy/user/kaechben/thesis/eval_jetnet30/",
    devices=1,
    accelerator="gpu",
    fast_dev_run=False,
    enable_progress_bar=False
)

num_classes=len(files)
# Create an instance of the model
model = MultiClassifier(datadir, num_classes,names,files)

# Create an instance of the data module
data_module = MyDataModule(datadir,names=names,files=files, test_split=0.2,val_split=0.1)
data_module.setup("train")
model.load_datamodule(data_module)
# Train the model
trainer.fit(model, data_module)
print("starting testing")
trainer.test(model, data_module.test_dataloader(),ckpt_path='best')
