import json
import time
import os
import gym
import minerl
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLNavigate-v0')
# MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)



for current_state, action, reward, next_state, done in data.batch_iter(batch_size=1, num_epochs=1, seq_len=1):
    print(current_state.keys())
    print(current_state['pov'][0])
    print(current_state['inventory'])
