import os
import numpy as np
import torch
import torch.nn as nn
from random import randint
import pytorch_lightning as pl
from models.PixelEncoder import PixelEncoder

from IPython import embed

class CURL_PL(pl.LightningModule):
    """
    CURL
    """

    def __init__(self,
            obs_shape=(3,64,64),
            z_dim=50,
            output_type="continuous",
            load_goal_states=False,
            device=None,
            threshold=18,
            path_goal_states=None
            ):
        super(CURL_PL, self).__init__()

        self.encoder = PixelEncoder(obs_shape, z_dim)

        self.encoder_target = PixelEncoder(obs_shape, z_dim)

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

        if load_goal_states:
            self.threshold = threshold
            self.path_gs = path_goal_states
            self.dev = device
            self.goal_states = self.load_goal_states()
            self.num_goal_states = self.goal_states.shape[0]


    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def compute_logits_(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, self.goal_states.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        return logits.squeeze()[z_pos].detach().cpu().item()

    def goal_state_distance(self, z_a):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, self.goal_states.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        return logits

    def load_goal_states(self):
        goal_states = []
        for gs in sorted(os.listdir(self.path_gs)):
            if 'npy' in gs:
                goal_states.append(np.load(os.path.join(self.path_gs, gs)))
        goal_states = np.array(goal_states)
        goal_states = torch.from_numpy(goal_states).squeeze().float().to(self.dev)
        return goal_states

    def get_goal_state(self, idx):
        return self.goal_states[idx].detach().cpu().numpy()
