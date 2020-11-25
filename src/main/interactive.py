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

setSeed(0)
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


env = gym.make('MineRLNavigateVectorObf-v0')

env.make_interactive(port=6666, realtime=True)

env.seed(2)
env.reset()

ini = time.time()
while True:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(obs['pov'].shape)
    time.sleep(0.1)
    if time.time()-ini > 200:
        break

env.close()
