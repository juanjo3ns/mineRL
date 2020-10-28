import os
import sys
import cv2
import time
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt
import torch.nn.functional as F

from models.GatedPixelCNN import GatedPixelCNN
from config import setSeed, getConfig
from customLoader import LatentBlockDataset

from tqdm import tqdm
from os.path import join
from pathlib import Path
from pprint import pprint

from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter


from pixelsnail import models.PixelSNAIL
from scheduler import CycleScheduler

from IPython import embed

def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()
    train_loss = []

    for i, (top, label) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)
        top = top.squeeze()
        target = top

        out, _ = model(top)

        loss = criterion(out, target)
        loss.backward()

        train_loss.append(loss.item())


        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )
    return np.asarray(train_loss).mean(0)


def saveModel(model, optim, iter):
	path = Path(f"../weights/{conf['experiment']}/pixel_{str(iter + 1).zfill(3)}.pt")
	torch.save({
        'state_dict': model.state_dict(),
		'optimizer': optim},
		path)


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':

    setSeed(0)
    assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
    conf = getConfig(sys.argv[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    mrl_train = LatentBlockDataset('../data/latent_blocks/train.npy', True, conf['pixelsnail']['img_dim'], transform=transforms.ToTensor())
    mrl_test = LatentBlockDataset('../data/latent_blocks/val.npy', False, conf['pixelsnail']['img_dim'], transform=transforms.ToTensor())

    train_loader = DataLoader(mrl_train, batch_size=conf['pixelsnail']['batch_size'], shuffle=True)
    test_loader = DataLoader(mrl_test, batch_size=conf['pixelsnail']['batch_size'], shuffle=True)

    if not os.path.exists(join('../weights', conf['experiment'])):
        os.mkdir(join('../weights', conf['experiment']))

    model = PixelSNAIL(
        [conf['pixelsnail']['img_dim'], conf['pixelsnail']['img_dim']],
        conf['pixelsnail']['n_class'],
        conf['pixelsnail']['channel'],
        conf['pixelsnail']['kernel_size'],
        conf['pixelsnail']['n_block'],
        conf['pixelsnail']['n_res_block'],
        conf['pixelsnail']['n_res_channel'],
        dropout=conf['pixelsnail']['dropout'],
        n_out_res_block=conf['pixelsnail']['n_out_res_block']
    )

    if conf['pixelsnail']['load']:
        weights = torch.load(f'../weights/{conf['experiment']}/{conf['pixelsnail']['name']}')['state_dict']
        model.load_state_dict(weights)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf['pixelsnail']['lr'])
    model = model.to(device)

    scheduler = None
    if conf['pixelsnail']['sched'] == 'cycle':
        scheduler = CycleScheduler(
            optimizer, conf['pixelsnail']['lr'], n_iter=len(train_loader) * conf['pixelsnail']['epochs'], momentum=None
        )

    writer = SummaryWriter(log_dir=f"../tensorboard/{conf['experiment']}/")

    bias = 13
    for i in range(conf['pixelsnail']['epochs']):
        train_loss = train(i, train_loader, model, optimizer, scheduler, device)
        writer.add_scalar('PixelSNAIL/Train Loss', train_loss, i)

        saveModel(model, optimizer, i + bias)
