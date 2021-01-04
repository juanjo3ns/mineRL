import os
import sys
import wandb
import torch

from pathlib import Path
from config import setSeed, getConfig
from main.byol import Contrastive

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

contr = Contrastive(**conf['byol']).cuda()
path = './results/byol_0.0/mineRL/3dfttou8/checkpoints/epoch=335-step=203615.ckpt'
checkpoint = torch.load(path)
contr.load_state_dict(checkpoint['state_dict'])
contr.compute_rewards()
