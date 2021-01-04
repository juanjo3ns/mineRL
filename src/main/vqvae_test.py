import os
import sys
import wandb
import torch

from os.path import join
from pathlib import Path
from config import setSeed, getConfig

from main.vqvae import VQVAE

import pytorch_lightning as pl
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

vqvae = VQVAE(**conf['vqvae']).cuda()
path = './results/vqvae_0.4/mineRL/3kgpkqsu/checkpoints/epoch=499-step=37999.ckpt'
checkpoint = torch.load(path)
vqvae.load_state_dict(checkpoint['state_dict'])
vqvae.compute_similarity()
