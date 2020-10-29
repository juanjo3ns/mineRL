import os
import sys
import cv2
import torch

import numpy as np
import matplotlib.pylab as plt
import torch.optim as optim
import torch.nn.functional as F

from models.VQVAE2 import VQVAE2
from config import setSeed, getConfig
from customLoader import MinecraftData

from pprint import pprint
from os.path import join
from pathlib import Path

from scipy.signal import savgol_filter
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])


transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                        ])


mrl_val = MinecraftData(conf['environment'], 'val', conf['split'], False, transform=transform)

validation_loader = DataLoader(mrl_val, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VQVAE2().to(device)

model.eval()

pprint(conf)

valid_originals = next(iter(validation_loader))
valid_originals = valid_originals.to(device)


for i in sorted(os.listdir('../weights/vqvae2_0'), key=lambda x: int(x.split('.')[0])):
    print(f"Loading model {i}...")
    weights = torch.load(f"../weights/vqvae2_0/{i}")['state_dict']
    model.load_state_dict(weights)
    _, valid_reconstructions, _, _ = model(valid_originals)
    grid = make_grid(valid_reconstructions.cpu().data, normalize=True)
    plt.imsave(f"../images/{i.split('.')[0]}.png", grid.permute(1,2,0).numpy())
