import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from IPython import embed

'''
Given a dataframe of x,y,index columns and a palette of colors,
plot points in their coordinates x,y and distinguish them by index.
'''
def plot_idx_maps(data, palette, legend):
    sns.scatterplot(x="x", y="y", hue="Code:", palette=palette, data=data, legend=legend)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

'''
Given a list of dataframes, plot index map for each goal state where the instead
of index we have a reward for each point.
'''
def plot_reward_maps(data_list):
    num_plots = len(data_list)
    if num_plots == 8:
        x,y = 2,4
    elif num_plots == 9:
        x,y = 3,3
    else:
        x,y = 3,5

    fig, axn = plt.subplots(x,y, sharex=True, sharey=True, constrained_layout=True)

    for i, ax in enumerate(axn.flat):
        if i < len(data_list):
            ax.set_title('$r(s, z=z_{' + i + '})$') # do not use f-string here
            g = ax.scatter(data_list[i]['x'],data_list[i]['y'], c=data_list[i]['reward'], marker='.')

    fig.colorbar(g, ax=axn[:,-1])
    plt.show()
