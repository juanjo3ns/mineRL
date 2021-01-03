import os
import csv
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
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from customLoader import CustomMinecraftData
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from models.PixelEncoder import PixelEncoder
from models.CURL import CURL

from main.curl import Contrastive

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


contr = Contrastive(**conf['curl']).cuda()
path = './results/curl_0.1/mineRL/1ddjitvv/checkpoints/epoch=499-step=302999.ckpt'
checkpoint = torch.load(path)
contr.load_state_dict(checkpoint['state_dict'])
contr.compute_rewards()
