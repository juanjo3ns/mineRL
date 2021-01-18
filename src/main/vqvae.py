import os
import csv
import wandb

import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from os.path import join
from pathlib import Path
from pprint import pprint
from collections import Counter, defaultdict
from config import setSeed, getConfig

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from customLoader import CustomMinecraftData
from torchvision.transforms import transforms

from models.VQVAE import VQVAE_PL

from pytorch_lightning.loggers import WandbLogger

from IPython import embed

class VQVAE(VQVAE_PL):
    def __init__(self, conf):
        super(VQVAE, self).__init__(**conf['vqvae'])

        self.batch_size = conf['batch_size']
        self.lr = conf['lr']
        self.split = conf['split']

        self.delay = conf['delay']
        self.trajectories = conf['trajectories']
        img_size = conf['img_size']


        self.example_input_array = torch.rand(self.batch_size, 3, img_size, img_size)
        if self.delay:
            self.example_input_array = (
                torch.rand(self.batch_size, 3, img_size, img_size),
                torch.rand(self.batch_size, 3, img_size, img_size)
                )

        self.transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                ])


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

    def compute_similarity(self):
        train_dataset = CustomMinecraftData(self.trajectories, 'train', 1, transform=self.transform, delay=False)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

        # compute embeddings of all trajectories
        csvfiles = []
        traj = -1

        folder = 'VQVAE_CENTROIDS_2'
        gs = 8
        if not os.path.exists(f"./results/{folder}"):
            os.mkdir(f"./results/{folder}")
        for i in range(gs):
            if not os.path.exists(f"./results/{folder}/vqvae_{i}"):
                os.mkdir(f"./results/{folder}/vqvae_{i}")

        print("\nComputing embeddings from all data points...")
        for i, key in enumerate(train_dataloader):
            if i % train_dataset.trj_length == 0:
                if len(csvfiles) > 0:
                    for c in csvfiles:
                        c.close()
                traj += 1
                csvfiles = []
                for k in range(gs):
                    csvfiles.append(open( f"./results/{folder}/vqvae_{k}/rewards_{k}.{traj}.csv", 'a'))

            print(f"\tTrajectory {traj}", end = '\r')
            z = self._encoder(key.cuda())
            distances = self._vq_vae.compute_distances(z)
            distances = distances.squeeze().detach().cpu().numpy()
            for j, d in enumerate(distances):
                csvwriter = csv.writer(csvfiles[j], delimiter=',')
                csvwriter.writerow([-d])

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
        train_dataset = CustomMinecraftData(self.trajectories, 'train', 1, transform=self.transform, delay=False)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

        num_clusters=10
        trajectories = self.load_trajectories()
        trajectories = trajectories.reshape(-1, 3)

        print("Get index from all data points...")
        values = pd.DataFrame(columns=['x', 'y', 'index'])
        for i, (key, p) in enumerate(zip(train_dataloader, trajectories)):

            x = float(p[0])
            y = float(p[2])

            e = self._encoder(key.cuda())
            k = self.compute_argmax(e)
            values = values.append({'x': x, 'y': y, 'index': int(k)}, ignore_index=True)


        palette = sns.color_palette(n_colors=num_clusters)
        sns.scatterplot(x="x", y="y", hue="index", palette=palette, data=values)
        labels = ["Code #" + str(i) for i in range(num_clusters)]
        embed()
        plt.legend(labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
