import numpy as np
from clvm_tfp import clvm
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def build_toy_missing_dataset(ml):
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

    ix = np.random.rand(N,D) < ml

    A[ix] = np.NaN
    plt.pcolor(A)
    plt.show()

    print('Percent missing data:', np.sum(np.isnan(A))/(N*D))


    # Perform mean-centering
    mB = np.nanmean(B, axis=0)
    B = B - mB

    mA = np.nanmean(A, axis=0)
    A = A - mA

    return A, B, labels

x_train, y_train, labels = build_toy_missing_dataset(0.6)

model = clvm(x_train, y_train, 10,2, target_missing=True)
#model.map(plot=args.plot, num_epochs=int(args.num_epochs), labels=labels)
model.variational_inference(plot=True, num_epochs=10000, labels=labels)