import os
import csv
import torch

import numpy as np
import pandas as pd
import seaborn as sns

from plot import *
from os.path import join
from pathlib import Path
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader, Subset
from customLoader import *
from torchvision.transforms import transforms

from IPython import embed


def get_loader(trajectories, transform, conf, shuffle=False, limit=None):
    train, _ = get_train_val_split(trajectories, 1)
    train_dataset = CustomMinecraftData(train, transform=transform, delay=False, **conf)

    if not limit == None:
        train_dataset = Subset(train_dataset, list(range(limit)))
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle, num_workers=0)
    return train_dataloader

def kmeans(embeddings, num_clusters):
    return KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

def compute_embeddings(loader, encode):
    return np.array([encode(data[:,0].cuda()).detach().cpu().numpy() for data in loader]).squeeze()

def get_images(loader):
    return torch.cat([data[:,0] for data in loader])


def load_trajectories(trajectories):
    print("Loading trajectories...")

    all_trajectories = []
    files = sorted([x for x in os.listdir(f"./results/{trajectories}/") if 'coords' in x], key=lambda x: int(x.split('.')[1]))
    for file in files:
        with open(f"./results/{trajectories}/{file}") as csv_file:
            trajectory = []
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for i, row in enumerate(csv_reader):
                trajectory.append(row)
            all_trajectories.append(trajectory)
    return np.array(all_trajectories).reshape(-1, 3)


def construct_map(enc):
    loader = get_loader(
        enc.trajectories,
        enc.transform,
        enc.conf,
        shuffle=enc.shuffle,
        limit=enc.limit)
    if 'Custom' in enc.trajectories[0]:
        trajectories = load_trajectories(enc.trajectories[0])
    embeddings = compute_embeddings(loader, enc.encoder)

    if enc.type == "index":
        index_map(trajectories, embeddings, enc)
    elif enc.type == "reward":
        reward_map(trajectories, embeddings, enc)
    elif enc.type == "embed":
        images = get_images(loader) + 0.5
        embed_map(embeddings, images, enc.experiment)
    else:
        raise NotImplementedError()

def index_map(trajectories, embeddings, enc):

    palette = sns.color_palette("Paired", n_colors=enc.num_clusters)

    print("Get index from all data points...")
    values = pd.DataFrame(columns=['x', 'y', 'Code:'])
    for i, (e, p) in enumerate(zip(embeddings, trajectories)):
        x = float(p[2])
        y = float(p[0])
        e = torch.from_numpy(e.squeeze()).cuda()
        k = enc.compute_argmax(e)
        values = values.append({'x': x, 'y': y, 'Code:': int(k)}, ignore_index=True)

    values['Code:'] = values['Code:'].astype('int32')
    plot_idx_maps(values, palette, "brief")

def reward_map(trajectories, embeddings, enc):
    print("Get index from all data points...")
    data_list = []
    for g in range(enc.num_clusters):
        print(f"Comparing data points with goal state {g}", end="\r")
        values = pd.DataFrame(columns=['x', 'y', 'reward'])
        for i, (e, p) in enumerate(zip(embeddings, trajectories)):
            x = float(p[2])
            y = float(p[0])
            e = torch.from_numpy(e.squeeze()).cuda()
            logits = enc.compute_logits(e)
            k = torch.argmax(logits).cpu().item()
            r = 0
            if k == g:
                max = logits[k].detach().cpu().item()
                logits[k] = 0
                k2 = torch.argmax(logits).cpu().item()
                sec_max = logits[k2].detach().cpu().item()
                r = (max-sec_max)/max
            values = values.append({'x': x, 'y': y, 'reward': r}, ignore_index=True)
        data_list.append(values)

    plot_reward_maps(data_list)

def embed_map(embeddings, images, exp):
    import tensorflow
    from torch.utils.tensorboard import SummaryWriter
    import tensorboard

    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
    writer = SummaryWriter(log_dir=os.path.join("./results", exp))
    writer.add_embedding(embeddings, label_img=images)
    writer.close()

def trainValSplit(traj_list, split):
    num_traj = len(traj_list)
    if split == 1:
        return traj_list, []
    else:
        # Since we can mix trajectories from different tasks, we want to shuffle them
        # e.g: otherwise we could have all treechop trajectories as validation
        shuffle(traj_list)
        return traj_list[:int(split*num_traj)], traj_list[int(split*num_traj):]

def get_train_val_split(trajectories, split):
    path = Path('../data')
    total_t = []
    if 'Custom' in trajectories[0]:
        for t in trajectories:
            items = sorted(os.listdir(path / t), key=lambda x: int(x.split('.')[0].split('_')[1]))
            items = [path / t / x for x in items]
            total_t.extend(items)
    else:
        for t in trajectories:
            items = sorted(os.listdir(path / t))
            items = [path / t / x for x in items]
            total_t.extend(items)
    return trainValSplit(total_t, split)
