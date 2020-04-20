import numpy as np
import sys
sys.path.append('../')

from clvm_tfp import clvm
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from utils.factor_plot import factor_plot

#Load Data
data = joblib.load(filename='./datasets/RNASeq_Processed.pkl')
A = np.array(data['A'])
B = np.array(data['B'])
labels = np.array(data['labels'])

n, d = A.shape

model = clvm(A, B, k_shared=d-1, sharedARD=True)
t_clvm = model.variational_inference(num_epochs=10000, plot=False)

fig, ax = plt.subplots(1,1)
ax.scatter(t_clvm[:,0], t_clvm[:,1], c=labels, alpha=0.7)
ax.set_xlabel('Latent Dimension 1')
ax.set_ylabel('Latent Dimension 2')

print(model.s_hat)
#factor_plot(model.s_hat, model.a_hat, fp='.', fn='test_fig')

plt.show()