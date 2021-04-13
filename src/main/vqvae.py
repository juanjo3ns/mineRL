import os
import csv
import time
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from plot import *
from os.path import join
from pathlib import Path
from pprint import pprint
from config import setSeed, getConfig
from collections import Counter, defaultdict
from main.utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from customLoader import *
from torchvision.transforms import transforms

from models.CustomVQVAE import VQVAE_PL

from pytorch_lightning.loggers import WandbLogger

from mod.q_functions import parse_arch

from IPython import embed

class VQVAE(VQVAE_PL):
    def __init__(self, conf):
        super(VQVAE, self).__init__(**conf['vqvae'])

        self.experiment = conf['experiment']
        self.batch_size = conf['batch_size']
        self.lr = conf['lr']
        self.split = conf['split']
        self.num_clusters = conf['vqvae']['num_embeddings']
        self.coord_cost = conf['coord_cost']

        self.delay = conf['delay']
        self.trajectories = conf['trajectories']
        self.trajectories_train, self.trajectories_val = get_train_val_split(self.trajectories, self.split)

        img_size = conf['img_size']
        self.conf = {'k_std': conf['k_std'], 'k_mean': conf['k_mean']}

        self.transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                ])

        self.test = conf['test']
        self.type = self.test['type']
        self.shuffle = self.test['shuffle']
        self.limit = self.test['limit']

    def training_step(self, batch, batch_idx):
        img, coords = batch
        
        i1, i2 = img[:, 0], img[:, 1]
        c1, c2 = coords[:, 0], coords[:, 1]

        vq_loss, img_recon, coord_recon, perplexity = self(i1, c1)
        
        img_recon_error = F.mse_loss(img_recon, i2)
        coord_recon_error = F.mse_loss(coord_recon, c2)

        coord_recon_error = self.coord_cost*coord_recon_error

        loss = img_recon_error + coord_recon_error + vq_loss

        self.logger.experiment.log({
            'loss/train': loss,
            'perplexity/train': perplexity,
            'loss_img_recon/train': img_recon_error,
            'loss_coord_recon/train': coord_recon_error,
            'loss_vq_loss/train': vq_loss
        })

        return loss

    def old_training_step(self, batch, batch_idx):

        x,y = batch[:,0], batch[:,1]

        vq_loss, data_recon, perplexity = self(x)
        recon_error = F.mse_loss(data_recon, y)
        loss = recon_error + vq_loss
        self.logger.experiment.log({
            'loss/train':loss,
            'perplexity/train': perplexity
        })

        return loss

    def validation_step(self, batch, batch_idx):

        img, coords = batch

        i1, i2 = img[:, 0], img[:, 1]
        c1, c2 = coords[:, 0], coords[:, 1]

        vq_loss, img_recon, coord_recon, perplexity = self(i1, c1)
        
        img_recon_error = F.mse_loss(img_recon, i2)
        coord_recon_error = F.mse_loss(coord_recon, c2)
        
        loss = img_recon_error + self.coord_cost*coord_recon_error + vq_loss
        
        self.logger.experiment.log({
            'loss/val': loss,
            'perplexity/val': perplexity,
            'loss_img_recon/val': img_recon_error,
            'loss_coord_recon/val': coord_recon_error,
            'loss_vq_loss/val': vq_loss
        })

        if batch_idx == 0:
            grid = make_grid(img_recon[:64].cpu().data)
            grid = grid.permute(1,2,0)
            self.logger.experiment.log({"Images": [wandb.Image(grid.numpy())]})

        return loss

    def old_validation_step(self, batch, batch_idx):

        x, y = batch[:, 0], batch[:, 1]

        vq_loss, data_recon, perplexity = self(x)
        recon_error = F.mse_loss(data_recon, y)
        loss = recon_error + vq_loss

        self.logger.experiment.log({
            'loss/val': loss,
            'perplexity/val': perplexity
        })

        if batch_idx == 0:
            grid = make_grid(data_recon[:64].cpu().data)
            grid = grid.permute(1, 2, 0)
            self.logger.experiment.log({"Images": [wandb.Image(grid.numpy())]})

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=1e-5)

    def train_dataloader(self):
        train_dataset = CustomMinecraftData(self.trajectories_train, transform=self.transform, delay=self.delay, **self.conf)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomMinecraftData(self.trajectories_val, transform=self.transform, delay=self.delay, **self.conf)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return val_dataloader

    def _construct_map(self):
        construct_map(self)
