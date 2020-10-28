import os
import sys
import cv2
import torch
import random

import numpy as np
import matplotlib.pylab as plt
import torch.optim as optim
import torch.nn.functional as F

from models.VQVAE import VQVAE
from models.GatedPixelCNN import GatedPixelCNN
from config import setSeed, getConfig
from customLoader import MinecraftData

from pprint import pprint
from os.path import join
from pathlib import Path

from sklearn.manifold import TSNE
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.load('../weights/test_2/24999.pt')['state_dict']

model = VQVAE(**conf['vqvae']).to(device)

model.load_state_dict(a)


codebook = model._vq_vae._embedding.weight.cpu().detach()
codebook = TSNE(2).fit_transform(codebook)

plt.scatter(codebook[:, 0], codebook[:, 1])
plt.title(f"Codebook (k={conf['vqvae']['num_embeddings']}, d={conf['vqvae']['embedding_dim']}) (TSNE projected)")
plt.show()
