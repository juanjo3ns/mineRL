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

from main.encoder import PixelEncoder
from main.model import CURL

from IPython import embed

setSeed(2)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLNavigate-v0')
# MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/home/usuaris/imatge/juan.jose.nieto/mineRL/data/')
# data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT, num_workers=1)

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

curl = CURL(obs_shape, feature_dim, pixel_encoder, pixel_encoder_target, load_goal_states=True, path_goal_states='./goal_states_flat_biome').to(device)


curl.eval()

print("Loading weights...")

if conf['curl']['load']:
    weights = torch.load(path_weights / conf['experiment'] / conf['curl']['epoch'])['state_dict']
    curl.load_state_dict(weights)

def save_image(img, name):
    fig, ax = plt.subplots()
    plt.imsave(f'../images/{name}.png',img)
    plt.close()

def save_fig(img, name):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'../images/{name}.svg')
    plt.close()

threshold = 18
print("Starting comparison...")
with torch.no_grad():
    matrix_logits = np.zeros((10,10))
    stack = []
    for i in range(8):
        stack.append(curl.get_goal_state(i))
    embed()


    # for i in range(10):
    #     gs = curl.get_goal_state(i)
    #     gs = torch.from_numpy(gs).to(device)
    #     for j in range(10):
    #         gs_comp = curl.get_goal_state(j)
    #         gs_comp = torch.from_numpy(gs_comp).to(device)
    #         embed()
    #         matrix_logits[i,j] = curl.compute_logits_(gs, gs_comp) > threshold
# fig, ax = plt.subplots(figsize=(24,15))
plt.imshow(matrix_logits, interpolation='nearest', aspect='auto')
plt.xticks(np.arange(0,10, step=1))
plt.yticks(np.arange(0,10, step=1))
plt.colorbar()
plt.tight_layout()
plt.show()

embed()
