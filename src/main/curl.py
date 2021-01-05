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

from models.CURL import CURL_PL

from IPython import embed


class CURL(CURL_PL):
    def __init__(self, conf):
        img_size = conf['img_size']
        obs_shape = (3, img_size, img_size)
        conf['curl']['obs_shape'] = obs_shape

        super(CURL, self).__init__(**conf['curl'])

        self.batch_size = conf['batch_size']
        self.lr = conf['lr']
        self.split = conf['split']
        self.delay = conf['delay']
        self.trajectories = conf['trajectories']

        self.tau = conf['tau']
        self.soft_update = conf['soft_update']

        self.transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                ])
        self.criterion = torch.nn.CrossEntropyLoss()


    def forward(self, data):
        key, query = data

        # Forward tensors through encoder
        z_a = self.encode(key)
        z_pos = self.encode(query, ema=True)

        # Compute distance
        logits = self.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)

        return logits, labels

    def training_step(self, batch, batch_idx):
        logits, labels = self(batch)
        loss = self.criterion(logits, labels)

        self.log('loss/train_epoch', loss, on_step=False, on_epoch=True)

        if batch_idx%2==0:
            self.soft_update_params()

        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self(batch)
        loss = self.criterion(logits, labels)

        self.log('loss/val_epoch', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, amsgrad=False)

    def train_dataloader(self):
        train_dataset = CustomMinecraftData(self.trajectories, 'train', self.split, transform=self.transform, delay=self.delay)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomMinecraftData(self.trajectories, 'val', self.split, transform=self.transform, delay=self.delay)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return val_dataloader

    def soft_update_params(self):
        net = self.encoder
        target_net = self.encoder_target
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def kmeans(self, embeddings, num_clusters):
        return KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

    def compute_rewards(self):
        train_dataset = CustomMinecraftData(self.trajectories, 'train', 1, transform=self.transform, delay=False)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

        # compute embeddings of all trajectories
        print("\nComputing embeddings from all data points...")
        embeddings = []
        for key in train_dataloader:
            embeddings.append(self.encode(key.cuda()).detach().cpu().numpy())
        embeddings = np.array(embeddings)

        num_clusters=8
        # compute kmeans clusters
        print("Computing kmeans over embeddings...")
        kmeans = self.kmeans(embeddings.squeeze(), num_clusters)
        self.goal_states = torch.from_numpy(kmeans.cluster_centers_.squeeze()).cuda()

        folder = "CURL_1.0_"
        csvfiles = []
        traj = -1
        for i in range(num_clusters):
            if not os.path.exists(f"./results/{folder}{i}"):
                os.mkdir(f"./results/{folder}{i}")

        print("\nComputing embeddings from all data points...")
        for i, e in enumerate(embeddings):
            if i % train_dataset.trj_length == 0:
                if len(csvfiles) > 0:
                    for c in csvfiles:
                        c.close()
                traj += 1
                csvfiles = []
                for k in range(num_clusters):
                    csvfiles.append(open( f"./results/{folder}{k}/rewards_{k}.{traj}.csv", 'a'))

            print(f"\tTrajectory {traj}", end = '\r')
            e = torch.from_numpy(e.squeeze()).cuda()
            distances = self.goal_state_distance(e)
            distances = distances.squeeze().detach().cpu().numpy()
            for j, d in enumerate(distances):
                csvwriter = csv.writer(csvfiles[j], delimiter=',')
                csvwriter.writerow([d])
