import os
import sys
import wandb

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

alg = 'byol'
wandb_logger = WandbLogger(
    project='mineRL',
    name=conf['experiment'],
    tags=[alg]
)

wandb_logger.log_hyperparams(conf[alg])

contr = Contrastive(**conf[alg])

trainer = pl.Trainer(
    fast_dev_run=True,
    gpus=1,
    max_epochs=conf['epochs'],
    progress_bar_refresh_rate=20,
    weights_summary='full',
    logger=wandb_logger,
    default_root_dir=f"./results/{conf['experiment']}"
)

trainer.fit(contr)
