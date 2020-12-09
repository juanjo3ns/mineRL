import os
import numpy as np
import torch
import torch.nn as nn
from random import randint

from IPython import embed

class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, z_dim, encoder, encoder_target, output_type="continuous", load_goal_states=False, device=None):
        super(CURL, self).__init__()

        self.encoder = encoder

        self.encoder_target = encoder_target

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

        if load_goal_states:
            self.threshold = 20
            self.path_gs = './goal_states_single'
            self.device = device
            self.goal_states = self.load_goal_states()


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
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        return logits

    def load_goal_states(self):
        goal_states = []
        for gs in sorted(os.listdir(self.path_gs)):
            if 'npy' in gs:
                goal_states.append(np.load(os.path.join(self.path_gs, gs)))
        return goal_states

    def compute_baselines(self):
        baselines = []
        for gs in self.goal_states:
            gs = torch.from_numpy(gs).float().to(self.device)
            baseline = self.compute_logits_(gs, gs)
            baselines.append(baseline.detach().cpu().numpy())
        self.baselines = baselines

    def sample_goal_state(self):
        return randint(0, len(self.goal_states)-1)

    def get_goal_state(self, idx):
        return self.goal_states[idx]

    def get_baseline(self, idx):
        return self.baselines[idx]
