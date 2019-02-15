import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
import matplotlib
from scipy import linalg
import bayespy.plot as bpplt

def factor_plot(w, s, fp, tick_label=None):
    '''

    :param w:
    :param s:
    :param fp:
    :param iter:
    :return:
    '''

    order_w = np.argsort(np.linalg.norm(w, axis=1))
    order_w = np.flipud(order_w)

    order_s = np.argsort(np.linalg.norm(s, axis=1))
    order_s = np.flipud(order_s)

    bpplt.pyplot.figure(figsize=(10, 10))
    bpplt.hinton(s[order_s, :].T)
    if tick_label.any() != None:
        nd = w.shape[1]
        plt.yticks(np.arange(1, nd + 1), tick_label.reshape(-1, 1))
    plt.savefig(fp + 'SharedLoading.png')

    bpplt.pyplot.figure(figsize=(10, 10))
    bpplt.hinton(w[order_w, :].T)
    if tick_label.any() != None:
        nd = s.shape[1]
        plt.yticks(np.arange(1, nd + 1), tick_label.reshape(-1, 1))
    plt.savefig(fp + 'TargetLoading.png')
