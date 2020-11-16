""" Plots.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_true_predicted_variance(
        y_true, y_mu, y_std, 
        x=None, xlabel=None, ylabel=None, 
        figsize=(8,6), plot_title=None):
    """ use only when both and mean predictions are known
    """

    y_true = y_true.flatten()
    y_mu = y_mu.flatten()
    y_std = y_std.flatten()
    
    l = y_true.shape[0]
    if x is None:
        x = range(l)

    plt.figure(figsize=figsize)
    plt.title(plot_title)
    gs = gridspec.GridSpec(3,1)
    
    # mean variance
    plt.subplot(gs[:-1,:])
    plt.plot(x, y_mu, '#990000', ls='-', lw=1.5, zorder=9, 
             label='predicted')
    plt.fill_between(x, (y_mu+2*y_std), (y_mu-2*y_std),
                     alpha=0.2, color='m', label='+-2sigma')
    plt.plot(x, y_true, '#e68a00', ls='--', lw=1, zorder=9, 
             label='true')
    plt.legend(loc='upper right')
    plt.title('true vs predicted')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=0)

    # errors        
    plt.subplot(gs[2,:])
    plt.plot(x, np.abs(np.array(y_true).flatten()-y_mu), '#990000', 
             ls='-', lw=0.5, zorder=9)
    plt.fill_between(x, np.zeros([l,1]).flatten(), 2*y_std,
                     alpha=0.2, color='m')
    plt.title("model error and predicted variance")
    plt.xlabel(xlabel)
    plt.ylabel('error ' + ylabel)
    plt.tight_layout()