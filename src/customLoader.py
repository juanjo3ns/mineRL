import os
import cv2
import torch

import numpy as np
import matplotlib.pylab as plt


from os.path import join
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset

from IPython import embed


class MinecraftData(Dataset):
    def __init__(self, env, split, extra, transform=None) -> None:
        self.environment = env
        self.train_split = split
        self.val_split = 1 - split
        self.extra = extra
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.data = self.loadData()
        # self.data_variance = np.var(self.data) / 255
        self.transform = transform

    def loadData(self) -> list:
        print('Loading data...')

        path = Path('../data')
        data = []

        for vid in os.listdir(path / self.environment):
            if self.extra:
                other_info = np.load(path / self.environment / vid / 'rendered.npz')

            vid_path = path / self.environment / vid / 'recording.mp4'
            frames = cv2.VideoCapture(str(vid_path))
            ret = True
            fc = 0
            while(frames.isOpened() and ret):
                ret, frame = frames.read()
                if ret and fc % 3 == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    data.append(frame)
                fc += 1
        return data


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        # img = torch.from_numpy(img).type(self.dtype)
        img = self.transform(img)
        return img



if __name__ == '__main__':
    transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                            ])
    mrl = MinecraftData('MineRLNavigate-v0', 0.7, False, transform=transform)
    embed()
    # img = mrl[10]
    # plt.imshow(img)
    # plt.show()
