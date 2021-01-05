import os
import sys
import torch

import numpy as np
from pathlib import Path
from config import setSeed, getConfig
from main.curl import CURL

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


curl = CURL(conf).cuda()
path = './results/curl_1.0/mineRL/1042bq9w/checkpoints/epoch=499-step=302999.ckpt'
checkpoint = torch.load(path)
curl.load_state_dict(checkpoint['state_dict'])
curl.compute_rewards()
