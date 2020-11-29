import os
import cv2
import torch

import numpy as np
import matplotlib.pylab as plt

from random import choice, randint
from os.path import join
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset

from IPython import embed


class MultiMinecraftData(Dataset):
    def __init__(self, env_list, mode, split, extra, transform=None, path='../data', **kwargs) -> None:
        self.path = path
        self.env_list = env_list
        self.mode = mode
        self.split = split
        self.extra = extra
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.data = self.loadData()
        # self.data_variance = np.var(self.data) / 255
        self.transform = transform
        self.k_std = kwargs['k_std']
        self.k_mean = kwargs['k_mean']

    def loadData(self) -> list:
        data = []

        print('Loading data...')
        path = Path(self.path)

        self.num_vids = 0

        for env in self.env_list:
            print(f"\n\tLoading environment {env}")
            self.num_vids += len(os.listdir(path / env))
            video_list = os.listdir(path / env)

            # if self.mode == 'train':
            #     video_list = video_list[:int(self.split*num_vids)]
            # else:
            #     video_list = video_list[int(self.split*num_vids):]

            for i, vid in enumerate(video_list):
                print(f"\tVid: {i}", end ='\r')
                video = []
                if self.extra:
                    other_info = np.load(path / env / vid / 'rendered.npz')

                vid_path = path / env / vid / 'recording.mp4'
                frames = cv2.VideoCapture(str(vid_path))
                ret = True
                fc = 0
                while(frames.isOpened() and ret):
                    ret, frame = frames.read()
                    if ret and fc % 1 == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video.append(frame)
                    fc += 1

                data.append(video)
        print()
        return data


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Pick trajectory
        trajectory = self.data[index]
        # Get query index uniformly
        num_frames = len(trajectory)-1
        query_idx = randint(0, num_frames)
        # Pick frame from trajectory
        query = trajectory[query_idx]
        # Compute key index by adding a normal distribution
        key_idx = query_idx + int(np.random.rand()*self.k_std + self.k_mean)
        # Get key frame
        key = trajectory[min(key_idx, num_frames)]
        if self.transform is not None:
            query = self.transform(query)
            key = self.transform(key)

        # Stack query and key to return [2,3,64,64]
        return torch.stack((query, key))


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
                if ret and fc % 1 == 0:
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

class LatentDataset(Dataset):
    """
    Loads latent block dataset
    """

    def __init__(self, file_name, transform=None):
        print('Loading latent block data')
        p = Path('../data')
        self.fname= file_name
        file_name = self.fname + '_t.npy'
        self.data_t = np.load(p / 'latent_blocks' / file_name, allow_pickle=True)
        file_name = self.fname + '_b.npy'
        self.data_b = np.load(p / 'latent_blocks' / file_name, allow_pickle=True)
        self.transform = transform

    def __getitem__(self, index):
        top = self.data_t[index]
        bottom = self.data_b[index]
        if self.transform is not None:
            top = self.transform(top)
            bottom = self.transform(bottom)
        label = 0
        return top, bottom, label

    def __len__(self):
        return len(self.data_t)

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
                            ])
    env_list = ['MineRLNavigate-v0', 'MineRLNavigateVectorObf-v0']
    mrl = MultiMinecraftData(env_list, 'train', 1, False, transform=transform)
    embed()
    # img = mrl[10]
    # plt.imshow(img)
    # plt.show()
