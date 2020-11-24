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

from curl.encoder import PixelEncoder
from curl.model import CURL

from IPython import embed

setSeed(2)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLNavigate-v0')
# MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/home/usuaris/imatge/juan.jose.nieto/mineRL/data/')
data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT, num_workers=1)

feature_dim = conf['curl']['embedding_dim']
img_size = conf['curl']['img_size']
obs_shape = (3, img_size, img_size)
batch_size = conf['batch_size']

if os.getenv('USER') == 'juanjo':
    path_weights = Path('../weights/')
elif os.getenv('USER') == 'juan.jose.nieto':
    path_weights = Path('/mnt/gpid07/users/juan.jose.nieto/weights/')
else:
    raise Exception("Sorry user not identified!")


pixel_encoder = PixelEncoder(obs_shape, feature_dim)
pixel_encoder_target = PixelEncoder(obs_shape, feature_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

curl = CURL(obs_shape, feature_dim, batch_size, pixel_encoder, pixel_encoder_target).to(device)


curl.eval()

if conf['curl']['load']:
    weights = torch.load(path_weights / conf['experiment'] / conf['curl']['epoch'])['state_dict']
    curl.load_state_dict(weights)

def save_image(img, name):
    fig, ax = plt.subplots()
    plt.imsave(f'../images/inference_attention_2/{name}.png',img)
    plt.close()

def save_fig(img, name):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'../images/inference_attention_2/{name}.svg')
    plt.close()

embed()
exit()

for i, (current_state, action, reward, next_state, done) in enumerate(data.batch_iter(batch_size=32, num_epochs=1, seq_len=200)):

    batch = current_state['pov']
    #
    # obs_anchor = batch[:,0,:,:,:]
    # obs_pos = batch[:,-1,:,:,:]
    embed()

    for s in skills:
        plt.imshow(s)
        plt.show()
        plt.close()

    obs_anchor = torch.from_numpy(state).float().unsqueeze(dim=0).to(device)
    obs_pos = torch.from_numpy(skills).float().squeeze().to(device)

    obs_anchor = obs_anchor.permute(0,3,1,2)
    obs_pos = obs_pos.permute(0,3,1,2)

    z_a = curl.encode(obs_anchor)
    z_pos = curl.encode(obs_pos, ema=True)

    logits = curl.compute_logits(z_a, z_pos)
    print(logits)
    logits = curl.compute_logits_(z_a, z_pos)
    print(logits)

    # obs_anchor = obs_anchor[5]

    # part 0 (store image trajectories every x samples)
    # for j, b in enumerate(batch[0]):
    #     if j%20==0:
    #         save_image(b, j)
    # ---------------

    # save_image(obs_anchor, 'original_image')
    # obs_anchor = torch.from_numpy(obs_anchor).unsqueeze(dim=0).float().to(device)
    #
    # obs_anchor = obs_anchor.permute(0,3,1,2)

    # part 1 store convolutional outputs
    # h, conv = curl.encoder.forward_conv(obs_anchor)
    # for j,c in enumerate(conv[0]):
    #     img = c.detach().cpu().numpy()
    #     save_image(img, j)
    # ---------

    # # part 2 store embeddings reshaped
    # z_a = curl.encode(obs_anchor)
    # z = z_a[0][:49].detach().cpu().numpy()
    # z = z.reshape(7,7)
    # save_fig(z, 'z_reshaped')
    # zw = z_a*curl.W
    # zw = zw.detach().cpu().numpy()
    # zw = np.diag(zw)[:49]
    # zw = zw.reshape(7,7)
    # save_fig(zw, 'zxW')
    # -----------

    # break
print('\n')
