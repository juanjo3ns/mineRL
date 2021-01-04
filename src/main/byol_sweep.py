
import wandb

def main():
    import os
    import sys
    import wandb

    from pathlib import Path
    from config import setSeed, getConfig
    from main.byol import Contrastive

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger

    from IPython import embed

    conf = {
      "experiment": "byol_0.sweep",
      "environment": "MineRLNavigate-v0",
      "epochs": 40,
      "byol": {
        "lr": 0.001,
        "batch_size": 32,
        "mlp_hidden_size": 512,
        "projection_size": 128,
        "img_size": 64,
        "m": 0.996,
        "delay": True,
        "trajectories": "CustomTrajectories2"
      }
    }

    alg = 'byol'
    wandb_logger = WandbLogger(
        project='mineRL',
        name=conf['experiment'],
        tags=[alg]
    )

    wandb_logger.log_hyperparams(conf[alg])

    contr = Contrastive(**conf[alg])

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=conf['epochs'],
        progress_bar_refresh_rate=20,
        weights_summary='full',
        logger=wandb_logger,
        default_root_dir=f"./results/{conf['experiment']}"
    )

    trainer.fit(contr)

sweep_config = {
    "method": 'bayes',
    "metric": {
        "name": "loss/train_epoch",
        "goal": "minimize"
    },
    "parameters": {
    "batch_size": {
      "distribution": "int_uniform",
      "max": 64,
      "min": 32
    },
    "lr": {
      "distribution": "uniform",
      "max": 0.01,
      "min": 0.0001
    },
    "mlp_hidden_size": {
      "distribution": "int_uniform",
      "max": 512,
      "min": 128
    }
  }
}
sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=main)
