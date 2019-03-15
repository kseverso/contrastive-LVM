import numpy as np
import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))
from clvm_tfp import clvm
import matplotlib.pyplot as plt
from sklearn.externals import joblib


# def build_toy_dataset():
#     np.random.seed(1)
#
#     N = 200
#     D = 30
#     gap = 3
#     # In B, all the data pts are from the same distribution, which has different variances in three subspaces.
#     B = np.zeros((N, D))
#     B[:, 0:10] = np.random.normal(0, 10, (N, 10))
#     B[:, 10:20] = np.random.normal(0, 3, (N, 10))
#     B[:, 20:30] = np.random.normal(0, 1, (N, 10))
#
#     # In A there are four clusters.
#     A = np.zeros((N, D))
#     A[:, 0:10] = np.random.normal(0, 10, (N, 10))
#     # group 1
#     A[0:50, 10:20] = np.random.normal(0, 1, (50, 10))
#     A[0:50, 20:30] = np.random.normal(0, 1, (50, 10))
#     # group 2
#     A[50:100, 10:20] = np.random.normal(0, 1, (50, 10))
#     A[50:100, 20:30] = np.random.normal(gap, 1, (50, 10))
#     # group 3
#     A[100:150, 10:20] = np.random.normal(2 * gap, 1, (50, 10))
#     A[100:150, 20:30] = np.random.normal(0, 1, (50, 10))
#     # group 4
#     A[150:200, 10:20] = np.random.normal(2 * gap, 1, (50, 10))
#     A[150:200, 20:30] = np.random.normal(gap, 1, (50, 10))
#     labels = [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50
#
#     # Perform mean-centering
#     mB = np.mean(B, axis=0)
#     B = B - mB
#
#     mA = np.mean(A, axis=0)
#     A = A - mA
#
#     return A, B, labels


def build_toy_dataset():
    np.random.seed(0)

    N = 400
    D = 30
    gap = 3
    # In B, all the data pts are from the same distribution, which has different variances in three subspaces.
    B = np.zeros((N, D))
    B[:, 0:10] = np.random.normal(0, 10, (N, 10))
    B[:, 10:20] = np.random.normal(0, 3, (N, 10))
    B[:, 20:30] = np.random.normal(0, 1, (N, 10))

    # In A there are four clusters.
    A = np.zeros((N, D))
    A[:, 0:10] = np.random.normal(0, 10, (N, 10))
    # group 1
    A[0:100, 10:20] = np.random.normal(0, 1, (100, 10))
    A[0:100, 20:30] = np.random.normal(0, 1, (100, 10))
    # group 2
    A[100:200, 10:20] = np.random.normal(0, 1, (100, 10))
    A[100:200, 20:30] = np.random.normal(gap, 1, (100, 10))
    # group 3
    A[200:300, 10:20] = np.random.normal(2 * gap, 1, (100, 10))
    A[200:300, 20:30] = np.random.normal(0, 1, (100, 10))
    # group 4
    A[300:400, 10:20] = np.random.normal(2 * gap, 1, (100, 10))
    A[300:400, 20:30] = np.random.normal(gap, 1, (100, 10))
    labels = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100

    # Perform mean-centering
    mB = np.mean(B, axis=0)
    B = B - mB

    mA = np.mean(A, axis=0)
    A = A - mA

    return A, B, labels

x_test, y_test, labels = build_toy_dataset()

model = clvm(x_test, y_test, 10, 2)#, sharedARD=True)
model.restore_graph(fl='./checkpoint/model1234.ckpt',plot=True, num_epochs=1000, labels=labels, paramsOnly=False)
plt.show()