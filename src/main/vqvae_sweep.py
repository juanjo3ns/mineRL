
import wandb
alg = 'vqvae'

def update_custom(conf, d):
    for k,v in d.items():
        if '.' in k:
            key = k.split('.')
            conf[key[0]][key[1]] = v
        else:
            conf[k] = v
    return conf

def main():
    import os
    import sys
    import wandb

    from pathlib import Path
    from config import setSeed, getConfig
    from main.vqvae import VQVAE
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger

    from IPython import embed

    run = wandb.init()
    conf = getConfig(sys.argv[1])

    conf = update_custom(conf, run.config)
    wandb_logger = WandbLogger(
        project='mineRL',
        name=run.name,
        tags=[alg, 'sweep']
    )

    wandb_logger.log_hyperparams(conf)

    vqvae = VQVAE(conf)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=conf['epochs'],
        progress_bar_refresh_rate=20,
        weights_summary='full',
        logger=wandb_logger,
        default_root_dir=f"./results/{conf['experiment']}"
    )

    trainer.fit(vqvae)

sweep_config = {
    "name": f"{alg}_2.sweep",
    "method": 'bayes',
    "metric": {
        "name": "loss/train",
        "goal": "minimize"
    },
    "parameters": {
        "vqvae.commitment_cost": {
              "distribution": "uniform",
              "max": 0.3,
              "min": 0.05
        },
        "lr": {
          "distribution": "uniform",
          "max": 0.005,
          "min": 0.00005
        }
  }
}

import os
del os.environ["SLURM_NTASKS"]
del os.environ["SLURM_JOB_NAME"]

sweep_id = wandb.sweep(sweep_config, project="mineRL")
wandb.agent(sweep_id, function=main, count=15)
