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
    def __init__(self, env, mode, split, extra, transform=None, path='../data') -> None:
        self.path = path
        self.environment = env
        self.mode = mode
        self.split = split
        self.extra = extra
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.data = self.loadData()
        # self.data_variance = np.var(self.data) / 255
        self.transform = transform

    def loadData(self) -> list:
        data = []

        print('Loading data...')
        path = Path(self.path)

        num_vids = len(os.listdir(path / self.environment))
        video_list = os.listdir(path / self.environment)

        if self.mode == 'train':
            video_list = video_list[:int(self.split*num_vids)]
        else:
            video_list = video_list[int(self.split*num_vids):]

        for vid in video_list:
            if self.extra:
                other_info = np.load(path / self.environment / vid / 'rendered.npz')

            vid_path = path / self.environment / vid / 'recording.mp4'
            frames = cv2.VideoCapture(str(vid_path))
            ret = True
            fc = 0
            while(frames.isOpened() and ret):
                ret, frame = frames.read()
                if ret and fc % 2 == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    data.append(frame)
                fc += 1
        return data


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        # img = torch.from_numpy(img).type(self.dtype)
        if self.transform is not None:
            img = self.transform(img)
        return img

class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset
    """

    def __init__(self, file_path, train=True, shape=16, transform=None):
        print('Loading latent block data')
        self.data = np.load(file_path, allow_pickle=True)
        self.transform = transform
        self.shape = shape

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape((self.shape, self.shape))
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                            ])
    mrl = MinecraftData('MineRLNavigate-v0', 'train', 0.7, False, transform=transform)
    embed()
    # img = mrl[10]
    # plt.imshow(img)
    # plt.show()
