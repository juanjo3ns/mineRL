import os
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

from main.encoder import PixelEncoder
from main.model import CURL

from pytorch_lightning.loggers import WandbLogger

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])


if os.getenv('USER') == 'juanjo':
    path_weights = Path('../weights/')
elif os.getenv('USER') == 'juan.jose.nieto':
    path_weights = Path('/mnt/gpid07/users/juan.jose.nieto/weights/')
else:
    raise Exception("Sorry user not identified!")



class Contrastive(pl.LightningModule):
    def __init__(self, feature_dim, tau, batch_size=256, lr=0.001, split=0.95,
                img_size=64, soft_update=2):
        super(Contrastive, self).__init__()


        self.batch_size = batch_size
        self.lr = lr
        self.split = split

        self.feature_dim = feature_dim
        self.tau = tau
        self.soft_update = soft_update
        obs_shape = (3, img_size, img_size)

        pixel_encoder = PixelEncoder(obs_shape, feature_dim)
        pixel_encoder_target = PixelEncoder(obs_shape, feature_dim)
        self.curl = CURL(obs_shape, feature_dim, pixel_encoder, pixel_encoder_target)

        # self.example_input_array = torch.rand(batch_size, 3, img_size, img_size)

        self.transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                ])
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        key, query = data

        # Forward tensors through encoder
        z_a = self.curl.encode(key)
        z_pos = self.curl.encode(query, ema=True)

        # Compute distance
        logits = self.curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)

        return logits, labels

    def training_step(self, batch, batch_idx):
        logits, labels = self(batch)
        loss = self.criterion(logits, labels)

        self.log('loss/train', loss, on_step=True, on_epoch=True)

        if batch_idx%2==0:
            self.soft_update_params()

        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self(batch)
        loss = self.criterion(logits, labels)

        self.log('loss/val', loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, amsgrad=False)

    def train_dataloader(self):
        train_dataset = CustomMinecraftData('CustomTrajectories2', 'train', self.split, transform=self.transform, delay=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomMinecraftData('CustomTrajectories2', 'val', self.split, transform=self.transform, delay=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return val_dataloader

    def soft_update_params(self):
        net = self.curl.encoder
        target_net = self.curl.encoder_target
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


wandb_logger = WandbLogger(
    project='mineRL',
    name=conf['experiment'],
    tags=['curl']
)

wandb_logger.log_hyperparams(conf['curl'])

contr = Contrastive(**conf['curl'])

trainer = pl.Trainer(
    gpus=1,
    max_epochs=conf['epochs'],
    progress_bar_refresh_rate=20,
    weights_summary='full',
    logger=wandb_logger,
    default_root_dir=f"./results/{conf['experiment']}"
)

trainer.fit(contr)
