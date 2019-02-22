import numpy as np
from clvm_tfp import clvm
import matplotlib.pyplot as plt
from sklearn.externals import joblib

data = np.genfromtxt('/Users/kristen.severson.ibm/PycharmProjects/cPCA/contrastive/experiments/datasets/Data_Cortex_Nuclear.csv',
                     delimiter=',',skip_header=1,usecols=range(1,78),filling_values=0)
classes = np.genfromtxt('/Users/kristen.severson.ibm/PycharmProjects/cPCA/contrastive/experiments/datasets/Data_Cortex_Nuclear.csv',
                        delimiter=',',skip_header=1,usecols=range(78,81),dtype=None)

target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]

labels = len(target_idx_A)*[0] + len(target_idx_B)*[1]
target_idx = np.concatenate((target_idx_A,target_idx_B))

target = data[target_idx]

background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
background = data[background_idx]

# Perform mean-centering
mB = np.mean(background, axis=0)
B = background - mB

mA = np.mean(target, axis=0)
A = target - mA

#Z-score the data
A = (A-np.mean(A, axis=0)) / np.std(np.concatenate((A, B)), axis=0)
B = (B-np.mean(B, axis=0)) / np.std(np.concatenate((A, B)), axis=0)

D = np.shape(A)[1]
Nx = np.shape(A)[0]
Ny = np.shape(B)[0]

#Specify model
Ks = 2 #latent dimensionality of shared space
Ki = 2 #latent dimensionality of indepedent space


#model = clvm(A, B, D-1, Ki, sharedARD=True, robust_flag=True)
model = clvm(A, B, Ks, Ki)
model.variational_inference(num_epochs=20000, plot=True, labels=labels)

