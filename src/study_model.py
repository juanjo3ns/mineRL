import os
import sys
import cv2
import torch
import random

import numpy as np
import matplotlib.pylab as plt
import torch.optim as optim
import torch.nn.functional as F

from VQVAE import VQVAE
from GatedPixelCNN import GatedPixelCNN
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

validation_loader = DataLoader(mrl_val, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.load('../weights/test_2/24999.pt')['state_dict']

model = VQVAE(**conf['vqvae']).to(device)

model.load_state_dict(a)

for i in range(5):
    q = next(iter(validation_loader))
    _, img, _ = model(q.cuda())
    img = img[0].permute(1,2,0).cpu().detach().numpy()
    plt.imshow(img+0.5)
    plt.show()
exit()

model.eval()

# for i in range(conf['vqvae']['num_embeddings']):
for i in range(20):
    # rand_index = np.random.randint(0,512, 256)
    rand_index = random.randint(0,512)
    img = model.get_centroids([rand_index]*256)[0]
    img = img.permute(1,2,0).cpu().detach().numpy()
    plt.imshow(img+0.5)
    plt.show()
