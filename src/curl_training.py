import os
import sys
import cv2
import gym
import json
import time
import minerl
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from pathlib import Path
from pprint import pprint
from config import setSeed, getConfig

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from encoder import PixelEncoder
from CURL import CURL

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLNavigate-v0')
# MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)

feature_dim = conf['curl']['embedding_dim']
img_size = conf['curl']['img_size']
obs_shape = (3, img_size, img_size)


pixel_encoder = PixelEncoder(obs_shape, feature_dim)
pixel_encoder_target = PixelEncoder(obs_shape, feature_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = conf['batch_size']
tau = conf['curl']['encoder_tau']

curl = CURL(obs_shape, feature_dim, batch_size, pixel_encoder, pixel_encoder_target).to(device)
optimizer = optim.Adam(curl.encoder.parameters(), lr=conf['learning_rate'], amsgrad=False)

writer = SummaryWriter(log_dir=f"../tensorboard/{conf['experiment']}/")

curl.train()

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def saveModel(model, optim, iter):
	path = Path(f"../weights/{conf['experiment']}/{iter}.pt")
	torch.save({
        'state_dict': model.state_dict(),
		'optimizer': optim},
		path)

def save_image(j, i):
    img = np.concatenate((j[0], j[-1]), axis=1)
    fig, ax = plt.subplots()
    plt.imsave(f'./images/curl_sampled/{i}.png',img)
    plt.close()

for i, (current_state, action, reward, next_state, done) in enumerate(data.batch_iter(batch_size=batch_size, num_epochs=conf['epochs'], seq_len=conf['seq_len'])):
    batch = current_state['pov']
    obs_anchor = batch[:,0,:,:,:]
    obs_pos = batch[:,-1,:,:,:]

    obs_anchor = torch.from_numpy(obs_anchor).float().squeeze().to(device)
    obs_pos = torch.from_numpy(obs_pos).float().squeeze().to(device)

    obs_anchor = obs_anchor.permute(0,3,1,2)
    obs_pos = obs_pos.permute(0,3,1,2)

    z_a = curl.encode(obs_anchor)
    z_pos = curl.encode(obs_pos, ema=True)


    logits = curl.compute_logits(z_a, z_pos)
    labels = torch.arange(logits.shape[0]).long().to(device)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    writer.add_scalar('CURL/Loss', loss.item(), i)

    if i%2==0:
        soft_update_params(curl.encoder, curl.encoder_target, tau)

    if i%1000==0:
        saveModel(curl, optimizer, i)
