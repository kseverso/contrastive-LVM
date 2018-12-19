import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
import matplotlib

def score_plot(X, w, pairs, fp, label=None, figname='ScorePlot', cmax=None, cmin=None, cmap='inferno'):
    sns.set_context("paper", rc={"lines.linewidth": 1.5, "lines.markersize": 3, 'axes.labelsize': 6,
                                 'text.fontsize': 6, 'legend.fontsize': 5,
                                 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'ylabel.fontsize': 6,
                                 'xlabel.fontsize': 6, 'text.usetex': False, 'axes.titlesize': 6,
                                 'axes.labelsize': 6, 'xtick.major.size': 0.5, 'xtick.major.pad': 2,
                                 'xtick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.major.pad': 2,
                                 'ytick.direction ': 'inout'})
    dim = len(pairs)
    fig, axs = plt.subplots(1, dim, dpi=300)
    fig.set_size_inches(2 * dim, 2)

    order_w = np.argsort(np.linalg.norm(w, axis=1))
    order_w = np.flipud(order_w)

    X = X[:, order_w]

    if label is not None:

        if cmax is None:
            cmax = np.max(label[~np.isnan(label)])
        if cmin is None:
            cmin = np.min(label[~np.isnan(label)])

        pidx = ~np.isnan(label)
        if dim == 1:
            norm = matplotlib.colors.Normalize(vmax=cmax, vmin=cmin)
            axs.scatter(X[pidx, pairs[0][0]], X[pidx, pairs[0][1]], c=label[pidx],
                       cmap=cmap, norm=norm, alpha=0.7)
            #plt.title("Target Latent Space")
            axs.set_xlabel("Latent Dimension " + str(pairs[0][0]))
            axs.set_ylabel("Latent Dimension " + str(pairs[0][1]))
            axs.tick_params(direction='in', length=1.5, width=0.75)

        else:
            for j, ax in enumerate(axs.flatten()):
                norm = matplotlib.colors.Normalize(vmax=cmax, vmin=cmin)
                ax.scatter(X[pidx, pairs[j][0]], X[pidx, pairs[j][1]], c=label[pidx],
                           cmap=cmap, norm=norm, alpha=0.7)
                #plt.title("Target Latent Space")
                ax.set_xlabel("Latent Dimension " + str(pairs[j][0]))
                ax.set_ylabel("Latent Dimension " + str(pairs[j][1]))
                ax.tick_params(direction='in', length=1.5, width=0.75)
    else:
        if dim==1:
            axs.scatter(X[:, pairs[0][0]], X[:, pairs[0][1]])
            # plt.title("Target Latent Space")
            axs.set_xlabel("Latent Dimension " + str(pairs[0][0]))
            axs.set_ylabel("Latent Dimension " + str(pairs[0][1]))
            axs.tick_params(direction='in', length=1.5, width=0.75)
        else:
            for j, ax in enumerate(axs.flatten()):
                ax.scatter(X[:, pairs[j][0]], X[:, pairs[j][1]])
                #plt.title("Target Latent Space")
                ax.set_xlabel("Latent Dimension " + str(pairs[j][0]))
                ax.set_ylabel("Latent Dimension " + str(pairs[j][1]))
                ax.tick_params(direction='in', length=1.5, width=0.75)

    fig.tight_layout()
    fig.savefig(fp + figname)
