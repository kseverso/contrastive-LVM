import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import warnings


#plt.style.use("ggplot")
warnings.filterwarnings('ignore')


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

"""
Variable names consistent with "Unsupervised Learning with Contrastive Latent Variable Models" except for k_target
x, y = oberved data with dimensions x: d x n and y: d x m
zi, zj = shared latent variables with dimensions: k_shared
ti = lower dimensional representation unique to target data with dimensions: k_target
qzi, qzj, qti = variational gaussian rep for zi, zj, ti respectively
s = shared factor loading with dimensions: d x k_shared
w = target factor loading with dimensions: d x k_target
noise = noise
"""

class clvm:
    def __init__(self, target, background, k_shared, k_target):

        self.n, self.d = target.shape
        self.m = background.shape[0]
        self.target = target
        self.background = background
        self.k_shared = k_shared
        self.k_target =k_target

        self.log_joint = ed.make_log_joint_fn(self.probabilistic_pca)
        self.log_q = ed.make_log_joint_fn(self.variational_model)

    def probabilistic_pca(self):  # (unmodeled) data
        # Parameters
        # shared factor loading
        s = tf.get_variable("w", shape = [self.k_shared, self.d])# initializer=tf.ones([self.k_shared, self.d], dtype=tf.float32))
        # target factor loading
        w = tf.get_variable("bx", shape=[self.k_target, self.d])#initializer=tf.ones([self.k_target, self.d], dtype=tf.float32))
        # noise
        noise = tf.get_variable("s", initializer=tf.ones([1], dtype=tf.float32))

        # Latent vectors
        zi = ed.Normal(loc=tf.zeros([self.n, self.k_shared]), scale=tf.ones([self.n, self.k_shared]), name='zi')
        zj = ed.Normal(loc=tf.zeros([self.m, self.k_shared]), scale=tf.ones([self.m, self.k_shared]), name='zj')
        ti = ed.Normal(loc=tf.zeros([self.n, self.k_target]), scale=tf.ones([self.n, self.k_target]), name='ti')

        # Observed vectors
        x = ed.Normal(loc=tf.matmul(zi, s) + tf.matmul(ti, w),
                           scale=noise * tf.ones([self.n, self.d]), name="x")
        y = ed.Normal(loc=tf.matmul(zj, s),
                           scale=noise * tf.ones([self.m, self.d]), name="y")

        return (x, y), (zi, zj, ti)

    def variational_model(self, qzi_mean, qzi_stddv, qzj_mean, qzj_stddv, qti_mean, qti_stddv):
        qzi = ed.Normal(loc=qzi_mean, scale=qzi_stddv, name="qzi")
        qzj = ed.Normal(loc=qzj_mean, scale=qzj_stddv, name="qzj")
        qti = ed.Normal(loc=qti_mean, scale=qti_stddv, name="qti")

        return qzi, qzj, qti

    def fn(self, zi, zj, ti):
        return self.log_joint(zi=zi, zj=zj, ti=ti, x=self.target, y=self.background)

    def target_q(self, qzi, qzj, qti, qzi_mean, qzi_stddv, qzj_mean, qzj_stddv, qti_mean, qti_stddv):
        return self.log_q(qzi=qzi, qzj=qzj, qti=qti, qzi_mean=qzi_mean, qzi_stddv=qzi_stddv,
                          qzj_mean=qzj_mean, qzj_stddv=qzj_stddv, qti_mean=qti_mean, qti_stddv=qti_stddv)

    def map(self):

        zi = tf.Variable(np.ones([self.n, self.k_shared]), dtype=tf.float32)
        zj = tf.Variable(np.ones([self.m, self.k_shared]), dtype=tf.float32)
        ti = tf.Variable(np.ones([self.n, self.k_target]), dtype=tf.float32)

        energy = -self.fn(zi, zj, ti)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        train = optimizer.minimize(energy)

        init = tf.global_variables_initializer()

        learning_curve = []

        num_epochs = 500

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    cE, _cx, _cy, _ci = sess.run([energy, zi, zj, ti])
                    learning_curve.append(cE)

            z_inferred_map = sess.run(ti)

        return z_inferred_map

    def variational_inference(self, num_epochs = 1500):

        qzi_mean = tf.Variable(np.ones([self.n, self.k_shared]), dtype=tf.float32)
        qzj_mean = tf.Variable(np.ones([self.m, self.k_shared]), dtype=tf.float32)
        qti_mean = tf.Variable(np.ones([self.n, self.k_target]), dtype=tf.float32)
        qzi_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.n, self.k_shared]), dtype=tf.float32))
        qzj_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.m, self.k_shared]), dtype=tf.float32))
        qti_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.n, self.k_target]), dtype=tf.float32))

        qzi, qzj, qti = self.variational_model(qzi_mean=qzi_mean, qzi_stddv=qzi_stddv, qzj_mean=qzj_mean,
                                               qzj_stddv=qzj_stddv, qti_mean=qti_mean, qti_stddv=qti_stddv)

        energy = self.fn(qzi, qzj, qti)
        entropy = -self.target_q(qzi, qzj, qti, qzi_mean, qzi_stddv, qzj_mean, qzj_stddv, qti_mean, qti_stddv)

        elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(-elbo)

        init = tf.global_variables_initializer()

        learning_curve = []

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    learning_curve.append(sess.run([elbo]))

            zi_inferred = sess.run(qti_mean)

        plt.figure()
        plt.plot(range(1, num_epochs, 5), learning_curve)
        plt.show()


        return zi_inferred



if __name__ == '__main__':
    x_train, y_train, labels = build_toy_dataset()
    print('shape of target data:', x_train.shape)
    print('shape of background data:', y_train.shape)

    model = clvm(x_train, y_train, 10, 2)

    z_post = model.map()
    print('MAP ti shape:', z_post.shape)

    c = ['k', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
         'tab:gray', 'tab:olive', 'tab:cyan']
    ms = ['o', 's', '*', '^', 'v', ',', '<', '>', '8', 'p']
    plt.figure()
    for i, l in enumerate(np.sort(np.unique(labels))):
        idx = np.where(labels == l)
        plt.scatter(z_post[idx, 0], z_post[idx, 1], marker=ms[i], color=c[i])
    plt.title("Target Latent Space MAP")

    tf.reset_default_graph() #need to do this so that you don't get error that variable already exists!!

    z_post = model.variational_inference()
    print('VI ti shape:', z_post.shape)

    plt.figure()
    for i, l in enumerate(np.sort(np.unique(labels))):
        idx = np.where(labels == l)
        plt.scatter(z_post[idx, 0], z_post[idx, 1], marker=ms[i], color=c[i])
    plt.title("Target Latent Space VI")

    plt.show()






