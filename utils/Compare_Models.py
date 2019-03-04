import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
import matplotlib
from scipy import linalg

def compare_models(fn, num_seed, fp, iter=1000):
    '''
    function to compare models based on different seeds
    :param fn: filenames - do not include .pkl
    :param num_seed: number of models to include, assumes 0 to num_seed models were tested
    :param fp: filepath to save figure
    :param iter: number of iterations to plot
    :return:
    '''

    fig, ax = plt.subplots(1,1,dpi=300)
    fig.set_size_inches(4,4)

    opt_lb = 1e+8
    opt_seed = np.NAN

    full_list = np.zeros(num_seed)

    for seed in np.arange(num_seed):
        try:
            res = joblib.load(fn + str(seed) + '.pkl')
            ax.plot(res['lb'][-iter:])
            full_list[seed] = res['lb'][-1]
            if res['lb'][-1] < opt_lb:
                opt_lb = res['lb'][-1]
                opt_seed = seed
        except:
            print(fn + str(seed) + ' was not run.')

    ax.set_xlabel('Final ' + str(iter) + ' iterations')
    ax.set_ylabel('Likelihood')

    fig.savefig(fp + 'CompObj.png')

    return opt_seed, opt_lb
