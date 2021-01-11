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
from collections import OrderedDict, defaultdict, Counter
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
        logits = self.compute_train(z_a, z_pos)
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


    def compute_embeddings(self,):
        train_dataset = CustomMinecraftData(self.trajectories, 'train', 1, transform=self.transform, delay=False)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

        # compute embeddings of all trajectories
        print("\nComputing embeddings from all data points...")
        embeddings = []
        for key in train_dataloader:
            embeddings.append(self.encode(key.cuda()).detach().cpu().numpy())
        embeddings = np.array(embeddings).squeeze()
        return embeddings, train_dataset

    def compute_rewards(self):

        num_clusters=8
        embeddings, train_dataset = self.compute_embeddings()
        kmeans = self.kmeans(embeddings, num_clusters)
        self.goal_states = torch.from_numpy(kmeans.cluster_centers_.squeeze()).cuda()

        folder = "CURL_1.0_"
        csvfiles = []
        traj = -1
        for i in range(num_clusters):
            if not os.path.exists(f"./results/{folder}{i}"):
                os.mkdir(f"./results/{folder}{i}")

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

    def store_goal_states(self):
        num_clusters=8
        kmeans, _, _ = self.compute_kmeans(num_clusters)
        for i, k in enumerate(kmeans.cluster_centers_):
            with open(f'./goal_states/flat_biome_curl_kmeans_0/{i}.npy', 'wb') as f:
                np.save(f, k)


    def load_trajectories(self):
        print("Loading trajectories...")

        all_trajectories = []
        files = sorted([x for x in os.listdir(f"./results/{self.trajectories}/") if 'coords' in x], key=lambda x: int(x.split('.')[1]))
        for file in files:
            with open(f"./results/{self.trajectories}/{file}") as csv_file:
                trajectory = []
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for i, row in enumerate(csv_reader):
                    trajectory.append(row)
                all_trajectories.append(trajectory)
        return np.array(all_trajectories)

    def index_map(self):

        num_clusters=8
        trajectories = self.load_trajectories()
        embeddings, train_dataset = self.compute_embeddings()
        trajectories = trajectories.reshape(-1, 3)

        width = 50
        div = int(100/width)
        print("Get index from all data points...")
        goals_in_unit = defaultdict(list)
        for i, (e, p) in enumerate(zip(embeddings, trajectories)):
            x = int((float(p[0])+50)/div)
            y = int((float(p[2])+50)/div)
            idx = x*width +y
            e = torch.from_numpy(e.squeeze()).cuda()
            k = self.compute_argmax(e)
            goals_in_unit[idx].append(k)

        matrix = np.zeros((width, width))
        for k,v in sorted(goals_in_unit.items()):
            x = int(k/width)
            y = int(k%width)
            c = Counter(v)
            matrix[x,y] = c.most_common(1)[0][0]

        plt.imshow(matrix)
        plt.legend(list(range(8)))
        plt.show()
