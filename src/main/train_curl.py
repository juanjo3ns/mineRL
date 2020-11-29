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
from main.random_shift import random_shift

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from main.encoder import PixelEncoder
from main.model import CURL

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])


# Minecraft initializations
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLNavigate-v0')
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/home/usuaris/imatge/juan.jose.nieto/mineRL/data/')
data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT, num_workers=1)


# Params
feature_dim = conf['curl']['embedding_dim']
img_size = conf['curl']['img_size']
obs_shape = (3, img_size, img_size)
batch_size = conf['batch_size']
tau = conf['curl']['encoder_tau']


# Dataloaders and so on
# transform = transforms.Compose([transforms.ToTensor()])
# env_list = ['MineRLNavigate-v0']
# mrl_train = MultiMinecraftData2(env_list, 'train', 1, False, transform=transform, **conf['gauss_step'])
# training_loader = DataLoader(mrl_train, batch_size=batch_size, shuffle=True)


# Weights path depending on execution server
if os.getenv('USER') == 'juanjo':
    path_weights = Path('../weights/')
elif os.getenv('USER') == 'juan.jose.nieto':
    path_weights = Path('/mnt/gpid07/users/juan.jose.nieto/weights/')
else:
    raise Exception("Sorry user not identified!")

if not os.path.exists(path_weights / conf['experiment']):
	os.mkdir(path_weights / conf['experiment'])


# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize models
pixel_encoder = PixelEncoder(obs_shape, feature_dim)
pixel_encoder_target = PixelEncoder(obs_shape, feature_dim)
curl = CURL(obs_shape, feature_dim, batch_size, pixel_encoder, pixel_encoder_target).to(device)
curl.train()


# Optimizers
optimizer = optim.Adam(curl.encoder.parameters(), lr=conf['learning_rate'], amsgrad=False)
optimizer_full = optim.Adam(curl.parameters(), lr=conf['learning_rate'], amsgrad=False)


# Tensorboard writer
writer = SummaryWriter(log_dir=f"../tensorboard/{conf['experiment']}/")


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def saveModel(path, exp, model, optim, iter):
	file_name = str(iter) + '.pt'
	path = path / exp / file_name
	torch.save({
        'state_dict': model.state_dict(),
		'optimizer': optim},
		path)

def save_image(j, i):
    img = np.concatenate((j[0], j[-1]), axis=1)
    fig, ax = plt.subplots()
    plt.imsave(f'./images/curl_sampled/{i}.png',img)
    plt.close()


# Training loop
for i, (current_state, action, reward, next_state, done) in enumerate(data.batch_iter(batch_size=batch_size, num_epochs=conf['epochs'], seq_len=conf['seq_len'])):

    if conf['sampling'] == 'uniform':
        idx = np.random.randint(5,15)
    elif conf['sampling'] == 'gaussian':
        idx = min(int(np.random.randn()*conf['gauss_step']['k_std']+conf['gauss_step']['k_mean']), conf['seq_len']-1)
    else:
        idx = conf['seq_len'] - 1


    batch = current_state['pov']
    # Split queries and keys
    query = batch[:,0,:,:,:]
    key = np.take(batch, idx, axis=1)

    # Apply random shift
    obs_anchor = random_shift(query, pad=4)
    obs_pos = random_shift(key, pad=4)

    # Put tensors in gpu if possible
    obs_anchor = torch.from_numpy(obs_anchor).float().squeeze().to(device)
    obs_pos = torch.from_numpy(obs_pos).float().squeeze().to(device)

    obs_anchor = obs_anchor.permute(0,3,1,2)
    obs_pos = obs_pos.permute(0,3,1,2)

    # Forward tensors through encoder
    z_a = curl.encode(obs_anchor)
    z_pos = curl.encode(obs_pos, ema=True)

    # Compute distance
    logits = curl.compute_logits(z_a, z_pos)
    labels = torch.arange(logits.shape[0]).long().to(device)

    # Compute loss
    loss = torch.nn.CrossEntropyLoss()(logits, labels)

    optimizer.zero_grad()
    optimizer_full.zero_grad()
    loss.backward()

    optimizer.step()
    optimizer_full.step()

    if i%conf['soft_update']==0:
        soft_update_params(curl.encoder, curl.encoder_target, tau)

    if i%5000==0:
        saveModel(path_weights, conf['experiment'], curl, optimizer, i)

    loss = loss.detach().cpu().numpy()
    print(f"Epoch {i}/{conf['epochs']} loss: {loss}", end='\r')
    writer.add_scalar('CURL/Loss', loss, i)
print()
