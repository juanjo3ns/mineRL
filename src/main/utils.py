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
from collections import Counter
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

def compute_kmeans(embeddings, num_clusters):
    return KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

def compute_embeddings(loader, encode):
    # return np.array([encode(data[:, 0].cuda()).detach().cpu().numpy() for data, coord in loader]).squeeze()
    return np.array([encode(data[:, 0].cuda(), coord[:, 0].cuda()).detach().cpu().numpy() for data, coord in loader]).squeeze()
    # return np.array([encode(coord[:, 0].cuda()).detach().cpu().numpy() for data, coord in loader]).squeeze()

def get_images(loader):
    return torch.cat([data[:,0] for data, coord in loader])


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

    embeddings = compute_embeddings(loader, enc.encode)

    if enc.type == "index":
        indexes = get_indexes(trajectories, embeddings, enc)
        index_map(enc, indexes)
    elif enc.type == "reward":
        reward_map(trajectories, embeddings, enc)
    elif enc.type == "embed":
        images = get_images(loader) + 0.5
        embed_map(embeddings, images, enc.experiment)
    elif enc.type == "centroides":
        indexes = get_indexes(trajectories, embeddings, enc)
        centroides_map(enc, loader, indexes)
    else:
        raise NotImplementedError()


def get_indexes(trajectories, embeddings, enc):
    print("Get index from all data points...")
    values = pd.DataFrame(columns=['x', 'y', 'Code:'])
    for i, (e, p) in enumerate(zip(embeddings, trajectories)):
        x = float(p[2])
        y = float(p[0])
        e = torch.from_numpy(e).cuda()
        k = enc.compute_argmax(e.unsqueeze(dim=0))
        values = values.append(
            {'x': x, 'y': y, 'Code:': int(k)}, ignore_index=True)

    values['Code:'] = values['Code:'].astype('int32')
    return values

def centroides_map(encoder, loader, indexes):
    experiment = encoder.experiment
    _, coords = encoder.decode(img=False)

    # unnormalize coordinates
    mu = loader.dataset.coord_mean
    std = loader.dataset.coord_std
    coords = np.array([ x*std + mu for x in coords])
    
    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])

    world = getWorld(encoder.trajectories[0])
    palette = sns.color_palette("Paired", n_colors=encoder.num_clusters)

    experiment = encoder.test['path_weights'].split('/')[0]

    show_centroides_inmap(df, indexes, palette, experiment, world)

def index_map(enc, indexes):
    code_list = indexes['Code:'].tolist()
    codes_count = Counter(code_list)
    palette = sns.color_palette("Paired", n_colors=len(list(set(code_list))))

    experiment = enc.test['path_weights'].split('/')[0]
    world = getWorld(enc.trajectories[0])

    plot_idx_maps(indexes, palette, experiment, world)
    skill_appearance(codes_count, palette, experiment, world)


def reward_map(trajectories, embeddings, enc):
    print("Get index from all data points...")
    data_list = []
    data_list_ = []
    reward_list = []
    traj_list = []
    for g in range(enc.num_clusters):
        print(f"Comparing data points with goal state {g}", end="\r")
        values = pd.DataFrame(columns=['x', 'y', 'reward'])
        values_ = pd.DataFrame(columns=['x', 'y', 'reward'])
        for i, (e, p) in enumerate(zip(embeddings, trajectories)):
            # if i > 71:
            #     break
            x = float(p[2])
            y = float(p[0])
            e = torch.from_numpy(e).cuda()

            # normal approach
            # k = enc.compute_argmax(e.unsqueeze(dim=0))
            # r = 0
            # if k == g:
            #     r = enc.compute_reward(e.unsqueeze(dim=0))

            # second approach
            # r = enc.compute_reward_mod(e.unsqueeze(dim=0), g)
            # if not (i+1) % 71 == 0:
            #     e2 = torch.from_numpy(embeddings[i+1]).cuda()
            #     r_next_state = enc.compute_reward_mod(e2.unsqueeze(dim=0), g)
            #     r -= r_next_state
            #     values = values.append({'x': x, 'y': y, 'reward': r}, ignore_index=True)

            # G (return) approach
            # if i>71:
            r = enc.compute_reward(e.unsqueeze(dim=0), g)

                # values = values.append({'x': x, 'y': y, 'reward': r}, ignore_index=True)

            reward_list.append(r)
            traj_list.append((x,y))

            if (i+1) % 71 == 0:
                G = 0
                gamma = 0.9
                for j, (r, c) in enumerate(zip(reward_list[::-1], traj_list[::-1])):
                    if not j==(len(reward_list)-1):
                        r -= reward_list[::-1][j+1]
                    G = r + gamma*G
                    values = values.append({'x': c[0], 'y': c[1], 'reward': r}, ignore_index=True)
                    values_ = values_.append({'x': c[0], 'y': c[1], 'reward': G}, ignore_index=True)
                reward_list = []
                traj_list = []


        data_list.append(values)
        data_list_.append(values_)

    plot_reward_maps(data_list, getWorld(enc.trajectories[0]), enc.experiment.split('_')[0])
    plot_reward_maps(data_list_, getWorld(enc.trajectories[0]), enc.experiment.split('_')[0], is_return=True)
    # plot_return_values(data_list, getWorld(enc.trajectories[0]), enc.experiment.split('_')[0])
    # plot_return_values(data_list_, getWorld(enc.trajectories[0]), enc.experiment.split('_')[0], is_return=True)

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


'''
Mapping from Trajectories to Worlds.
We can have multiple datasets of trajectories that belong to a unique world.
'''
def getWorld(t):
    if '8' in t or '9' in t:
        return 0
    elif '10' in t:
        return 1
    elif '11' in t:
        return 2
    elif '12' in t:
        return 4
    else:
        raise NotImplementedError()
