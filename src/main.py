import os
import sys
import cv2
import torch

import numpy as np
import matplotlib.pylab as plt
import torch.optim as optim
import torch.nn.functional as F

from model import Model
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

mrl = MinecraftData(conf['environment'], 0.7, False, transform=transform)

training_loader = DataLoader(mrl, batch_size=conf['batch_size'], shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(**conf['vqvae']).to(device)

pprint(conf)

optimizer = optim.Adam(model.parameters(), lr=conf['learning_rate'], amsgrad=False)

model.train()

train_res_recon_error = []
train_res_perplexity = []

writer = SummaryWriter(log_dir='../tensorboard/test_1/')

def saveModel(model, optim):
	path = Path('../weights/last_iteration.pt')
	torch.save({
        'state_dict': model.state_dict(),
		'optimizer': optim},
		path)


valid_originals = next(iter(training_loader))
valid_originals = valid_originals.to(device)




for i in range(conf['num_training_updates']):
    batch = next(iter(training_loader))
    data = batch.to(device)

    optimizer.zero_grad()
    vq_loss, data_recon, perplexity = model(data)
    # recon_error = F.mse_loss(data_recon, data) / mrl.data_variance
    recon_error = F.mse_loss(data_recon, data)
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    writer.add_scalar('Reconstruction Error', recon_error.item(), i)
    writer.add_scalar('Perplexity', perplexity.item(), i)

    if (i+1) % 100 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()
        model.eval()
        vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
        _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
        valid_reconstructions = model._decoder(valid_quantize)
        grid = make_grid(valid_reconstructions.cpu().data)+0.5
        writer.add_image('images', grid, i)
        model.train()

train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

for i, (r,p) in enumerate(zip(train_res_recon_error_smooth, train_res_perplexity_smooth)):
    writer.add_scalar('savgol/Reconstruction Error', r, i)
    writer.add_scalar('savgol/Perplexity', p, i)

saveModel(model, optimizer)
# f = plt.figure(figsize=(16,8))
# ax = f.add_subplot(1,2,1)
# ax.plot(train_res_recon_error_smooth)
# ax.set_yscale('log')
# ax.set_title('Smoothed NMSE.')
# ax.set_xlabel('iteration')
#
# ax = f.add_subplot(1,2,2)
# ax.plot(train_res_perplexity_smooth)
# ax.set_title('Smoothed Average codebook usage (perplexity).')
# ax.set_xlabel('iteration')
