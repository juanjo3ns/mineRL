
import wandb

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

    conf = {
      "experiment": f"vqvae_{run.step}.sweep",
      "environment": "MineRLNavigate-v0",
      "trajectories": "CustomTrajectories2",
      "epochs": 50,
      "delay": False,
      "img_size": 64,
      "split": 0.90,
      "vqvae": {
          "num_hiddens": 64,
          "num_residual_hiddens": 32,
          "num_residual_layers": 2,
          "embedding_dim": 256,
          "num_embeddings": 10,
          "commitment_cost": 0.25,
          "decay": 0.99
      }
    }

    conf = {**conf , **run.config}

    wandb_logger = WandbLogger(
        project='mineRL',
        name=run.name,
        tags=['vqvae', 'sweep']
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
    "name": "vqvae_sweep",
    "method": 'bayes',
    "metric": {
        "name": "loss/train",
        "goal": "minimize"
    },
    "parameters": {
        "batch_size": {
          "distribution": "int_uniform",
          "max": 256,
          "min": 32
        },
        "lr": {
          "distribution": "uniform",
          "max": 0.01,
          "min": 0.0001
        },
  }
}

import os
del os.environ["SLURM_NTASKS"]
del os.environ["SLURM_JOB_NAME"]

sweep_id = wandb.sweep(sweep_config, project="mineRL")
wandb.agent(sweep_id, function=main, count=10)
