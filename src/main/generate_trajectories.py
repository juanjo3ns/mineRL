import os
import sys
import cv2
import gym
import json
import time
import copy
import minerl
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from pathlib import Path
from pprint import pprint
from config import setSeed, getConfig

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mod.env_wrappers import ClusteredActionWrapper, FrameSkip, ObtainCoordWrapper

from collections import OrderedDict
from IPython import embed

setSeed(0)

episodes = 250
steps = 700
frame_skip = 10

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLNavigate-v0')
env = gym.make('MineRLNavigate-v0')

folder = 'CustomTrajectories12'
outdir = f"./results/{folder}"

if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists(f"../data/{folder}"):
    os.mkdir(f"../data/{folder}")

env.goal_state = 0
env = FrameSkip(env, skip=frame_skip)
env = ObtainCoordWrapper(env, outdir)
env = ClusteredActionWrapper(env, frame_skip)

# env.make_interactive(port=6666, realtime=True)

env.seed(0)

for episode in range(episodes):
    obs = env.reset()
    trajectory = [obs['pov']]
    for step in range(steps // frame_skip):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        trajectory.append(obs['pov'])


    trajectory = np.array(trajectory)
    with open(f'../data/{folder}/trajectory_{episode}.npy', 'wb') as f:
        np.save(f, trajectory)

env.close()
