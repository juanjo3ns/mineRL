
import wandb
alg = 'curl'

def main():
    import os
    import sys
    import wandb

    from pathlib import Path
    from config import setSeed, getConfig
    from main.curl import CURL
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger

    from IPython import embed

    run = wandb.init()

    conf = {
      "experiment": f"{alg}_{run.step}.sweep",
      "environment": "MineRLNavigate-v0",
      "trajectories": "CustomTrajectories2",
      "epochs": 50,
      "delay": True,
      "tau": 0.005,
      "soft_update": 2,
      "img_size": 64,
      "split": 0.90,
      "curl": {
          "z_dim": 64
      },
      "test":{
          "path_goal_states": './goal_states/flat_biome_curl_kmeans_2',
          "path_weights": './curl_2.1/mineRL/2szj8vij/checkpoints/epoch=499-step=428999.ckpt'
      }
    }

    conf = {**conf , **run.config}

    wandb_logger = WandbLogger(
        project='mineRL',
        name=run.name,
        tags=[alg, 'sweep']
    )

    wandb_logger.log_hyperparams(conf)

    curl = CURL(conf)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=conf['epochs'],
        progress_bar_refresh_rate=20,
        weights_summary='full',
        logger=wandb_logger,
        default_root_dir=f"./results/{conf['experiment']}"
    )

    trainer.fit(curl)

sweep_config = {
    "name": f"{alg}_sweep",
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
sweep_id = wandb.sweep(sweep_config, project="mineRL")
wandb.agent(sweep_id, function=main, count=10)
