import os
import csv
import sys
import json
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from os.path import join
from pathlib import Path
from pprint import pprint
from config import setSeed, getConfig
from collections import OrderedDict
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from customLoader import CustomMinecraftData
from torchvision.transforms import transforms

from models.BYOL_Basics import ResNet18, MLPHead

from IPython import embed


class Contrastive(pl.LightningModule):
    def __init__(self, m=0.996, batch_size=256, lr=0.001, split=0.95,
                mlp_hidden_size=512, projection_size=128, img_size=64,
                delay=False, trajectories='CustomTrajectories2'):
        super(Contrastive, self).__init__()


        self.batch_size = batch_size
        self.split = split
        self.delay = delay
        self.trajectories = trajectories
        self.m = m
        self.lr = lr

        self.online_network = ResNet18(mlp_hidden_size, projection_size)
        self.target_network = ResNet18(mlp_hidden_size, projection_size)

        self.predictor = MLPHead(
            in_channels=self.online_network.projetion.net[-1].out_features,
            mlp_hidden_size=mlp_hidden_size,
            projection_size=projection_size)



        self.transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                ])

    def forward(self, batch_view_1, batch_view_2):
        p1 = self.predictor(self.online_network(batch_view_1))
        p2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            t2 = self.target_network(batch_view_1)
            t1 = self.target_network(batch_view_2)

        return p1,p2,t1,t2

    def on_train_start(self):
        self.initializes_target_network()

    def training_step(self, batch, batch_idx):
        batch_view_1 = batch[0]
        batch_view_2 = batch[1]

        loss = self.update(batch_view_1, batch_view_2)

        self.log('loss/train_epoch', loss, on_step=False, on_epoch=True)

        self._update_target_network_parameters()

        return loss

    def validation_step(self, batch, batch_idx):
        batch_view_1 = batch[0]
        batch_view_2 = batch[1]

        loss = self.update(batch_view_1, batch_view_2)

        self.log('loss/val_epoch', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(list(self.online_network.parameters()) + list(self.target_network.parameters()), lr=self.lr, amsgrad=False)

    def train_dataloader(self):
        train_dataset = CustomMinecraftData(self.trajectories, 'train', self.split, transform=self.transform, delay=self.delay)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomMinecraftData(self.trajectories, 'val', self.split, transform=self.transform, delay=self.delay)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return val_dataloader

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        p1,p2,t1,t2 = self(batch_view_1, batch_view_2)

        loss = self.regression_loss(p1, t1)
        loss += self.regression_loss(p2, t2)
        return loss.mean()

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def kmeans(self, embeddings):
        return KMeans(n_clusters=8, random_state=0, max_iter=500).fit(embeddings)

    def compute_rewards(self):
        train_dataset = CustomMinecraftData(self.trajectories, 'train', 1, transform=self.transform, delay=False)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
        self.eval()
        # compute embeddings of all trajectories
        print("\nComputing embeddings from all data points...")
        embeddings = []
        for key in train_dataloader:
            embeddings.append(self.predictor(self.online_network(key.cuda())).detach().cpu().numpy())
        embeddings = np.array(embeddings)

        # compute kmeans clusters
        print("Computing kmeans over embeddings...")
        kmeans = self.kmeans(embeddings.squeeze())
        csvfile = None

        folder = "byol_gs1_"
        # iterate over kmeans clusters
        for j, k in enumerate(kmeans.cluster_centers_):
            k = torch.from_numpy(k).cuda()
            # iterate over trajectories and steps
            if not os.path.exists(f"./results/{folder}{j}"):
                os.mkdir(f"./results/{folder}{j}")
            print(f"\nComparing to cluster {j}")
            traj = -1
            for i, e in enumerate(embeddings):
                if i % train_dataset.trj_length == 0:
                    if not csvfile == None:
                        csvfile.close()
                    traj += 1
                    csvfile = open( f"./results/{folder}{j}/rewards_{j}.{traj}.csv", 'a')
                print(f"\tTrajectory {traj}", end = '\r')
                # compute reward between cluster and step
                e = torch.from_numpy(e.squeeze()).cuda()
                r = self.regression_loss(e.unsqueeze(dim=0), k.unsqueeze(dim=0))
                r = round(r.detach().cpu().item(),3)
                # store reward csv
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([r])

        csvfile.close()
