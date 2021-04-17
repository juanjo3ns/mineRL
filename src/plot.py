import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from IPython import embed

COLOR = 'gray'
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
# plt.rcParams['axes.facecolor'] = '#303030'

def compute_entropy(c):
    counts = np.array(list(c.values()), dtype=float)
    prob = counts/counts.sum()
    return (-prob*np.log2(prob)).sum()

def skill_appearance(codes, palette, id="0", alg="curl"):

    entropy = compute_entropy(codes)

    df = pd.DataFrame([dict(codes)]).T
    df = df.sort_values(0, ascending=False)
    df['skill'] = df.index
    palette = [palette[x] for x in df['skill'].tolist()]
    df.columns = ['count', 'skill']
    fig, ax = plt.subplots()
    sns.barplot(x='skill', y='count', palette=palette, data=df, order=df['skill'])
    t1 = '$p(z=z_{i})$'
    ax.set_title(f"{t1}   Entropy={round(entropy,2)}")
    ax.set_xlabel('skill')
    plt.savefig(f'/home/juanjo/Pictures/Minecraft/CW/CW_{id}_{alg}_skillhistogram_cw0_pixels_coords.3.png', transparent=True)

'''
Given a dataframe of x,y,index columns and a palette of colors,
plot points in their coordinates x,y and distinguish them by index.
'''
def plot_idx_maps(data, palette, id="0", alg="curl"):
    fig, ax = plt.subplots(figsize=(9,9))
    sns.scatterplot(x="x", y="y", hue="Code:", palette=palette, data=data)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.get_legend().remove()
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'/home/juanjo/Pictures/Minecraft/CW/CW_{id}_{alg}_indexmap_cw0_pixels_coords.3.png', transparent=True)

'''
Given a list of dataframes, plot index map for each goal state where the instead
of index we have a reward for each point.
'''
def plot_reward_maps(data_list, id="0", alg="curl", is_return=False):

    num_plots = len(data_list)
    if num_plots == 8:
        x,y = 2,4
    elif num_plots == 9:
        x,y = 3,3
    elif num_plots == 10:
        x,y = 2,5
    else:
        x,y = 3,5

    fig, axn = plt.subplots(x,y, sharex=True, sharey=True, constrained_layout=True, figsize=(15,6))

    is_return = 'return' if is_return else ''

    for i, ax in enumerate(axn.flat):
        if i < len(data_list):
            if is_return=='return':
                ax.set_title('$G(s, z=z_{' + str(i) + '})$') # do not use f-string here
            else:
                ax.set_title('$r(s, z=z_{' + str(i) + '})$') # do not use f-string here
            g = ax.scatter(data_list[i]['x'],data_list[i]['y'], c=data_list[i]['reward'], marker='.')
        # ax.axis('off')
    fig.colorbar(g, ax=axn[:,-1])
    plt.savefig(f'/home/juanjo/Pictures/Minecraft/CW/CW_{id}_{alg}_rewardmaps_alternative_{is_return}.png', transparent=True)

'''
Given a list of dataframes, plot return values for each timestep
'''
def plot_return_values(data_list, id="0", alg="curl", is_return=False):
    num_plots = len(data_list)
    if num_plots == 8:
        x,y = 2,4
    elif num_plots == 9:
        x,y = 3,3
    elif num_plots == 10:
        x,y = 2,5
    else:
        x,y = 3,5

    is_return = 'return' if is_return else 'reward'

    fig, axn = plt.subplots(x,y, figsize=(15,6), sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(axn.flat):
        if i < len(data_list):
            data = data_list[i]
            data.columns = ['x','y', is_return]
            sns.lineplot(x=data.index, y=is_return, data=data, ax=ax)
            if is_return=='return':
                ax.set_title('$G_{t}(z=z_{' + str(i) + '})$')
            else:
                ax.set_title('$r_{t}(z=z_{' + str(i) + '})$')

    plt.savefig(f'/home/juanjo/Pictures/Minecraft/CW/CW_{id}_{alg}_returnvalues_single_{is_return}.png', transparent=True)


def plot_q_maps(data_list):


    num_plots = len(data_list)
    if num_plots == 8:
        x,y = 2,4
    elif num_plots == 9:
        x,y = 3,3
    elif num_plots == 10:
        x,y = 2,5
    else:
        x,y = 3,5

    fig, axn = plt.subplots(x,y, sharex=True, sharey=True, constrained_layout=True)

    for i, ax in enumerate(axn.flat):
        if i < len(data_list):
            ax.set_title('$q(s, z=z_{' + str(i) + '})$') # do not use f-string here
            g = ax.scatter(data_list[i]['x'],data_list[i]['y'], c=data_list[i]['q_value'], marker='.')
        # ax.axis('off')
    fig.colorbar(g, ax=axn[:,-1])
    plt.show()


def compare_func():
    x = np.arange(0,1,0.01)
    plt.plot(x,np.minimum(np.ones(100),-np.log(x)))
    plt.plot(x, np.power(1000,-x))
    plt.plot(x, np.power(100,-x))
    plt.plot(x, np.power(10,-x))
    plt.legend(['min(1, -log_e(x))', '1e3^-x', '1e2^-x', '10^-x'])
    plt.show()

# compare_func()

# def skill_appearance():
#     fig, ax = plt.subplots()
#     x = [0.12972549, 0.00556863, 0.0185098 , 0.01223529, 0.09945098, 0.10764706, 0.136, 0.09541176, 0.11784314, 0.08419608, 0.00894118, 0.06478431, 0.11968627]
#     palette = sns.color_palette("Paired", n_colors=12)
#     palette.append((48/255,48/255,48/255))
#     sns.barplot(np.arange(len(x)),x, palette=palette)
#     ax.set_title('$p(z=z_{i})$')
#     ax.set_xlabel('skill')
#     plt.savefig('skills_histogram.png', transparent=True)

# skill_appearance()
