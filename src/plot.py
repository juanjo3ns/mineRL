import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed

'''
Given a dataframe of x,y,index columns and a palette of colors,
plot points in their coordinates x,y and distinguish them by index.
'''
def plot_idx_maps(data, palette, legend):
    palette.insert(0,(0,0,0))
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
    elif num_plots == 10:
        x,y = 2,5
    else:
        x,y = 3,5

    fig, axn = plt.subplots(x,y, sharex=True, sharey=True, constrained_layout=True)

    for i, ax in enumerate(axn.flat):
        if i < len(data_list):
            ax.set_title('$r(s, z=z_{' + str(i) + '})$') # do not use f-string here
            g = ax.scatter(data_list[i]['x'],data_list[i]['y'], c=data_list[i]['reward'], marker='.')

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
