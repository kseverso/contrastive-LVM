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


class ppca_model:
    def __init__(self, target, background, ks, ki):

        self.nx, self.d = target.shape
        self.ny = background.shape[0]
        self.target = target
        self.background = background
        self.Ks = ks
        self.Ki = ki

        #self.model_template = tf.make_template("model", self.probabilistic_pca)
        self.log_joint = ed.make_log_joint_fn(self.probabilistic_pca)
        print('log joint:', self.log_joint)
        self.log_q = ed.make_log_joint_fn(self.variational_model)
        print('log_q:', self.log_q)


    def probabilistic_pca(self):  # (unmodeled) data
        # Parameters
        # w = tf.get_variable("w", [])
        # bx = tf.get_variable("bx", [])
        # s = tf.get_variable("s", [])
        #with tf.variable_scope('', reuse=tf.AUTO_REUSE):

        w = tf.get_variable("w", shape = [self.Ks, self.d])# initializer=tf.ones([self.Ks, self.d], dtype=tf.float32))
        bx = tf.get_variable("bx", shape=[self.Ki, self.d])#initializer=tf.ones([self.Ki, self.d], dtype=tf.float32))
        s = tf.get_variable("s", initializer=tf.ones([1], dtype=tf.float32))

        # w = tf.ones([self.Ks, self.d], dtype=tf.float32)
        # bx = tf.ones([self.Ki, self.d], dtype=tf.float32)
        # s = tf.ones([1], dtype=tf.float32)

        # Latent vectors
        zx = ed.Normal(loc=tf.zeros([self.nx, self.Ks]), scale=tf.ones([self.nx, self.Ks]), name='zx')
        zy = ed.Normal(loc=tf.zeros([self.ny, self.Ks]), scale=tf.ones([self.ny, self.Ks]), name='zy')
        zi = ed.Normal(loc=tf.zeros([self.nx, self.Ki]), scale=tf.ones([self.nx, self.Ki]), name='zi')

        # Observed vectors
        x = ed.Normal(loc=tf.matmul(zx, w) + tf.matmul(zi, bx),
                           scale=s * tf.ones([self.nx, self.d]), name="x")
        y = ed.Normal(loc=tf.matmul(zy, w),
                           scale=s * tf.ones([self.ny, self.d]), name="y")
        return (x, y), (zx, zy, zi)

    def variational_model(self, qzx_mean, qzx_stddv, qzy_mean, qzy_stddv, qzi_mean, qzi_stddv):
        qzx = ed.Normal(loc=qzx_mean, scale=qzx_stddv, name="qzx")
        qzy = ed.Normal(loc=qzy_mean, scale=qzy_stddv, name="qzy")
        qzi = ed.Normal(loc=qzi_mean, scale=qzi_stddv, name="qzi")

        return qzx, qzy, qzi

    def fn(self, zx, zy, zi):
        return self.log_joint(zx=zx, zy=zy, zi=zi, x=self.target, y=self.background)

    def target_q(self, qzx, qzy, qzi, qzx_mean, qzx_stddv, qzy_mean, qzy_stddv, qzi_mean, qzi_stddv):
        return self.log_q(qzx=qzx, qzy=qzy, qzi=qzi, qzx_mean=qzx_mean, qzx_stddv=qzx_stddv,
                          qzy_mean=qzy_mean, qzy_stddv=qzy_stddv, qzi_mean=qzi_mean, qzi_stddv=qzi_stddv)

    def map(self):

        zx = tf.Variable(np.ones([self.nx, self.Ks]), dtype=tf.float32)
        zy = tf.Variable(np.ones([self.ny, self.Ks]), dtype=tf.float32)
        zi = tf.Variable(np.ones([self.nx, self.Ki]), dtype=tf.float32)

        energy = -self.fn(zx, zy, zi)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        train = optimizer.minimize(energy)

        init = tf.global_variables_initializer()

        t = []

        num_epochs = 500

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    cE, cx, cy, ci = sess.run([energy, zx, zy, zi])
                    t.append(cE)

            z_inferred_map = sess.run(zi)

        return z_inferred_map

    def vi(self):

        qzx_mean = tf.Variable(np.ones([self.nx, self.Ks]), dtype=tf.float32)
        qzy_mean = tf.Variable(np.ones([self.ny, self.Ks]), dtype=tf.float32)
        qzi_mean = tf.Variable(np.ones([self.nx, self.Ki]), dtype=tf.float32)
        qzx_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.nx, self.Ks]), dtype=tf.float32))
        qzy_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.ny, self.Ks]), dtype=tf.float32))
        qzi_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.nx, self.Ki]), dtype=tf.float32))

        qzx, qzy, qzi = self.variational_model(qzx_mean=qzx_mean, qzx_stddv=qzx_stddv, qzy_mean=qzy_mean,
                                               qzy_stddv=qzy_stddv, qzi_mean=qzi_mean, qzi_stddv=qzi_stddv)

        energy = self.fn(qzx, qzy, qzi)
        entropy = -self.target_q(qzx, qzy, qzi, qzx_mean, qzx_stddv, qzy_mean, qzy_stddv, qzi_mean, qzi_stddv)

        elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(-elbo)

        init = tf.global_variables_initializer()

        t = []

        num_epochs = 1500

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    t.append(sess.run([elbo]))

            zi_inferred = sess.run(qzi_mean)

        plt.figure()
        plt.plot(range(1, num_epochs, 5), t)
        plt.show()


        return zi_inferred



if __name__ == '__main__':
    num_datapoints = 5000
    data_dim = 2
    latent_dim = 1
    stddv_datapoints = 0.5

    x_train, y_train, labels = build_toy_dataset()
    print('shape of target data:', x_train.shape)
    print('shape of background data:', y_train.shape)
    model = ppca_model(x_train, y_train, 10, 2)

    z_post = model.map()
    # plt.figure()
    # plt.plot(w_hat[:, 0], w_hat[:, 1], 'o')
    # #plt.show()

    #z_post = model.vi()

    print('zi shape:', z_post.shape)

    c = ['k', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
         'tab:gray', 'tab:olive', 'tab:cyan']
    ms = ['o', 's', '*', '^', 'v', ',', '<', '>', '8', 'p']
    plt.figure()
    for i, l in enumerate(np.sort(np.unique(labels))):
        idx = np.where(labels == l)
        plt.scatter(z_post[idx, 0], z_post[idx, 1], marker=ms[i], color=c[i])
    plt.title("Target Latent Space MAP")

    tf.reset_default_graph()

    z_post = model.vi()
    plt.figure()
    for i, l in enumerate(np.sort(np.unique(labels))):
        idx = np.where(labels == l)
        plt.scatter(z_post[idx, 0], z_post[idx, 1], marker=ms[i], color=c[i])
    plt.title("Target Latent Space VI")

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # for i, l in enumerate(np.sort(np.unique(labels))):
    #     idx = np.where(labels == l)
    #     plt.scatter(zx_post[idx, 0], zx_post[idx, 1], marker=ms[i], color=c[i])
    # plt.title("Target Class Shared Space")
    #
    # plt.subplot(1, 2, 2)
    # plt.scatter(zy_post[:, 0], zy_post[:, 1])
    # plt.title("Background Class Shared Space")
    # plt.savefig('./experiments/Edward/z' + str(seed) + '.png')


    # plt.figure()
    # plt.plot(w_hat2[:,0], w_hat2[:,1],'o')
    plt.show()






