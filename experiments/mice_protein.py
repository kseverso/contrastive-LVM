import numpy as np
import sys
sys.path.append('/Users/kristen.severson.ibm/PycharmProjects/cLVM/contrastive-LVM/')
from clvm_tfp import clvm
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from contrastive import CPCA
import seaborn as sns


# Import data; data originally described in C. Higuera, K.J. Gardiner, and K.J. Cios, "Self-organizing feature maps
# identify proteins critical to learning in a mouse model of down syndrome," PLOS ONE, vol 10, p e0129126, 2015.
data = np.genfromtxt('/Users/kristen.severson.ibm/PycharmProjects/cPCA/contrastive/experiments/datasets/Data_Cortex_Nuclear.csv',
                     delimiter=',',skip_header=1,usecols=range(1,78),filling_values=0)
classes = np.genfromtxt('/Users/kristen.severson.ibm/PycharmProjects/cPCA/contrastive/experiments/datasets/Data_Cortex_Nuclear.csv',
                        delimiter=',',skip_header=1,usecols=range(78,81),dtype=None)

target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]

labels = len(target_idx_A)*[0] + len(target_idx_B)*[1]
target_idx = np.concatenate((target_idx_A,target_idx_B))

A = data[target_idx]

background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
B = data[background_idx]

#Specify model
Ks = 2 #latent dimensionality of shared space
Ki = 2 #latent dimensionality of target space

pm = np.linspace(0, 0.75, 4)
nA, mA = A.shape
col = ['Reds', 'Blues']
ms = ['o', 's']

fig, axs = plt.subplots(3, 4, figsize=(11, 8))

# loop over four levels of missing data
for j in range(4):

    # create missing data
    # set seed for reproducibility of missing pattern
    np.random.seed(0)
    A_inputed = np.copy(A)
    a_level = np.sum(np.isnan(A_inputed)) / (nA * mA)
    while a_level < pm[j]:
        # choose a row
        r = np.random.randint(nA)
        c = np.random.randint(mA)
        l = np.random.randint(3, 6)
        A_inputed[r:r + l, c] = np.NaN
        a_level = np.sum(np.isnan(A_inputed)) / (nA * mA)

    #z-score
    A_inputed = (A_inputed-np.nanmean(A_inputed, axis=0)) / np.nanstd(np.concatenate((A_inputed, B)), axis=0)
    B_scaled = (B - np.mean(B, axis=0)) / np.nanstd(np.concatenate((A_inputed, B)), axis=0)

    model = clvm(A_inputed, B_scaled, Ks, Ki, target_missing=True, robust_flag=True)
    t_clvm = model.variational_inference(seed=2, num_epochs=6000, plot=False)

    for i, l in enumerate(np.sort(np.unique(labels))):
        idx = np.where(labels == l)
        sns.kdeplot(np.squeeze(t_clvm[idx, 0]), np.squeeze(t_clvm[idx, 1]), cmap=col[i],
                    shade=True, shade_lowest=False, ax=axs[2, j])

    #use mean imputation in combination with PCA and cPCA
    A_inputed[np.isnan(A_inputed)] = 0

    mdl = CPCA()
    pca_res = mdl.fit_transform(A_inputed, B)

    for i, l in enumerate(np.sort(np.unique(labels))):
        idx = np.where(labels == l)
        sns.kdeplot(np.squeeze(pca_res[0][idx, 0]), np.squeeze(pca_res[0][idx, 1]), cmap=col[i],
                    shade=True, shade_lowest=False, ax=axs[0, j])

        sns.kdeplot(np.squeeze(pca_res[1][idx, 0]), np.squeeze(pca_res[1][idx, 1]), cmap=col[i],
                    shade=True, shade_lowest=False, ax=axs[1, j])

    if j == 0:
        axs[0, j].set_ylabel('PCA \\ Latent Dimension 2')
        axs[0, j].set_title('1% Missing')
        axs[1, j].set_ylabel('cPCA \\ Latent Dimension 2')
        axs[2, j].set_ylabel('cLVM \\ Latent Dimension 2')
        axs[2, j].set_xlabel('Latent Dimension 1')
    if j==1:
        axs[0, j].set_title('25% Missing')
        axs[2, j].set_xlabel('Latent Dimension 1')
    if j==2:
        axs[0, j].set_title('50% Missing')
        axs[2, j].set_xlabel('Latent Dimension 1')
    if j==3:
        axs[0, j].set_title('75% Missing')
        axs[2, j].set_xlabel('Latent Dimension 1')

fig.tight_layout()
plt.show()


