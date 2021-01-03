import os
import csv
import sys
import cv2
import gym
import json
import time
import copy
import minerl
import wandb

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from os.path import join
from pathlib import Path
from pprint import pprint
from config import setSeed, getConfig
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from customLoader import CustomMinecraftData
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from models.VQVAE import VectorQuantizer, VectorQuantizerEMA, Encoder, Decoder

from pytorch_lightning.loggers import WandbLogger

from IPython import embed

class VQVAE(pl.LightningModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0,
                 batch_size=256, lr=0.001, split=0.95, img_size=64,
                 delay=False, trajectories='CustomTrajectories2'):
        super(VQVAE, self).__init__()


        self.batch_size = batch_size
        self.lr = lr
        self.split = split

        self.delay = delay
        self.trajectories = trajectories

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        # self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
        #                               out_channels=embedding_dim,
        #                               kernel_size=1,
        #                               stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(num_hiddens,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.example_input_array = torch.rand(batch_size, 3, img_size, img_size)

        self.transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                ])

    def forward(self, x):
        z = self._encoder(x)
        # z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

    def training_step(self, batch, batch_idx):
        x = batch
        y = batch
        if self.delay:
            x,y = batch

        vq_loss, data_recon, perplexity = self(x)
        recon_error = F.mse_loss(data_recon, y)
        loss = recon_error + vq_loss

        self.log('loss/train', loss, on_step=False, on_epoch=True)
        self.log('perplexity/train', perplexity, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = batch
        if self.delay:
            x,y = batch
        vq_loss, data_recon, perplexity = self(x)
        recon_error = F.mse_loss(data_recon, y)
        loss = recon_error + vq_loss

        self.log('loss/val', loss, on_step=False, on_epoch=True)
        self.log('perplexity/val', perplexity, on_step=False, on_epoch=True)

        if batch_idx == 0:
            grid = make_grid(data_recon[:64].cpu().data)
            grid = grid.permute(1,2,0)
            self.logger.experiment.log({"Images": [wandb.Image(grid.numpy())]})

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=1e-5)

    def train_dataloader(self):
        train_dataset = CustomMinecraftData(self.trajectories, 'train', self.split, transform=self.transform, delay=self.delay)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomMinecraftData(self.trajectories, 'val', self.split, transform=self.transform, delay=self.delay)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return val_dataloader


    def get_centroids(self, idx):
        z_idx = torch.tensor(idx).cuda()
        embeddings = torch.index_select(self._vq_vae._embedding.weight.detach(), dim=0, index=z_idx)
        embeddings = embeddings.view((1,2,2,64))
        embeddings = embeddings.permute(0, 3, 1, 2).contiguous()

        return self._decoder(embeddings)

    def compute_similarity(self):
        train_dataset = CustomMinecraftData(self.trajectories, 'train', 1, transform=self.transform, delay=False)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

        # compute embeddings of all trajectories
        csvfiles = []
        traj = -1

        if not os.path.exists(f"./results/VQVAE_CENTROIDS_0"):
            os.mkdir(f"./results/VQVAE_CENTROIDS_0")
        for i in range(10):
            if not os.path.exists(f"./results/VQVAE_CENTROIDS_0/vqvae_{i}"):
                os.mkdir(f"./results/VQVAE_CENTROIDS_0/vqvae_{i}")

        print("\nComputing embeddings from all data points...")
        for i, key in enumerate(train_dataloader):
            if i % train_dataset.trj_length == 0:
                if len(csvfiles) > 0:
                    for c in csvfiles:
                        c.close()
                traj += 1
                csvfiles = []
                for k in range(10):
                    csvfiles.append(open( f"./results/VQVAE_CENTROIDS_0/vqvae_{k}/rewards_{k}.{traj}.csv", 'a'))

            print(f"\tTrajectory {traj}", end = '\r')
            z = self._encoder(key.cuda())
            distances = self._vq_vae.compute_distances(z)
            distances = distances.squeeze().detach().cpu().numpy()
            for j, d in enumerate(distances):
                csvwriter = csv.writer(csvfiles[j], delimiter=',')
                csvwriter.writerow([-d])

    def save_encoding_indices(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        _, _, _, encoding_indices = self._vq_vae(z)
        return encoding_indices
