import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
import matplotlib
from scipy import linalg
import bayespy.plot as bpplt


def factor_plot(w, a, fp, fn, tick_label=None):
    '''
    :param w: factor loading matrix to be plotted
    :param fn: filename
    :param fp: filepath
    :return:
    '''
    order_w = np.argsort(np.linalg.norm(w, axis=1))
    order_w = np.flipud(order_w)


    bpplt.pyplot.figure(figsize=(10, 10))
    bpplt.hinton(w[order_w, :].T)
    plt.savefig(fp + fn)

    fig, ax = plt.subplots(1,1,dpi=300)
    ax.plot(1/(np.exp(a[order_w])), 'o')
    fig.tight_layout()
    fig.savefig(fp + 'alpha' + fn)