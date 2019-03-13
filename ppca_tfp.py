import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import warnings

plt.style.use("ggplot")
warnings.filterwarnings('ignore')


def build_toy_dataset(N, D, K, sigma=1):

    x_train = np.zeros((D, N))
    w_train = np.random.normal(0.0, 2.0, size=(D, K))
    z_train = np.random.normal(0.0, 1.0, size=(K, N))
    mean = np.dot(w_train, z_train)
    for n in range(N):
        x_train[:, n] = np.random.multivariate_normal(mean[:, n], sigma*np.eye(D))

    print("True principal axes:")
    print(w_train)
    print("True ratio:")
    print(w_train[0]/w_train[1])

    return x_train.T, w_train


def build_test_dataset(N, D, K, w, sigma=1):
    x_test = np.zeros((D, N))
    z = np.random.normal(0.0, 1.0, size=(K, N))
    mean = np.dot(w, z)
    for n in range(N):
        x_test[:, n] = np.random.multivariate_normal(mean[:, n], sigma*np.eye(D))

    return x_test.T

class ppca_model:
    def __init__(self, data, k):

        self.n, self.d = data.shape
        self.data = data
        self.k = k

        self.log_joint = ed.make_log_joint_fn(self.probabilistic_pca)
        self.log_q = ed.make_log_joint_fn(self.variational_model)

    def probabilistic_pca(self):  # (unmodeled) data
        w = ed.Normal(loc=tf.zeros([self.k, self.d]),
                      scale=2.0 * tf.ones([self.k, self.d]),
                      name="w")  # parameter
        z = ed.Normal(loc=tf.zeros([self.n, self.k]),
                      scale=tf.ones([self.n, self.k]),
                      name="z")  # parameter
        x = ed.Normal(loc=tf.matmul(z, w),
                      scale=0.5 * tf.ones([self.n, self.d]),
                      name="x")  # (modeled) data
        return x, (w, z)

    def variational_model(self, qw_mean, qw_stddv, qz_mean, qz_stddv):
        qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
        qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
        return qw, qz

    def target(self, w, z):
        return self.log_joint(w=w, z=z, x=self.data)

    def target_q(self, qw, qz, qw_mean, qz_mean, qw_stddv, qz_stddv):
        return self.log_q(qw=qw, qz=qz, qw_mean=qw_mean, qw_stddv=qw_stddv, qz_mean=qz_mean, qz_stddv=qz_stddv)

    def map(self):

        w = tf.Variable(np.ones([self.k, self.d]), dtype=tf.float32)
        z = tf.Variable(np.ones([self.n, self.k]), dtype=tf.float32)
        print('test what is target:', self.target(w, z))
        energy = -self.target(w, z)

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
                    cE, cw, cz = sess.run([energy, w, z])
                    t.append(cE)

            w_inferred_map = sess.run(w)
            z_inferred_map = sess.run(z)

        return w_inferred_map, z_inferred_map

    def vi(self):
        tf.reset_default_graph()

        qw_mean = tf.Variable(np.ones([self.k, self.d]), dtype=tf.float32)
        qz_mean = tf.Variable(np.ones([self.n, self.k]), dtype=tf.float32)
        qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.k, self.d]), dtype=tf.float32))
        qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.n, self.k]), dtype=tf.float32))

        qw, qz = self.variational_model(qw_mean=qw_mean, qw_stddv=qw_stddv, qz_mean=qz_mean, qz_stddv=qz_stddv)

        energy = self.target(qw, qz)
        entropy = -self.target_q(qw=qw, qz=qz, qw_mean=qw_mean, qw_stddv=qw_stddv,
                                 qz_mean=qz_mean, qz_stddv=qz_stddv)

        elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        train = optimizer.minimize(-elbo)

        init = tf.global_variables_initializer()

        t = []

        num_epochs = 300

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    t.append(sess.run([elbo]))

            w_mean_inferred = sess.run(qw_mean)
            z_mean_inferred = sess.run(qz_mean)


        return w_mean_inferred, z_mean_inferred, t

    def replace_params(self, w, z):

        def interceptor(rv_constructor, *rv_args, **rv_kwargs):
            name = rv_kwargs.pop("name")
            if name == "w":
                rv_kwargs["value"] = w
            if name == "z":
                rv_kwargs["value"] = z
            return rv_constructor(*rv_args, **rv_kwargs)

        return interceptor

    def generate(self, w_inferred, z_inferred):

        with ed.interception(self.replace_params(w=w_inferred, z=z_inferred)):
            generate = self.probabilistic_pca()

        with tf.Session() as sess:
            x_generated, _ = sess.run(generate)

        return x_generated


class ppca_predict:
    def __init__(self, data, k, w_inferred):

        self.n, self.d = data.shape
        self.data = data
        self.k = k
        self.w_inferred = w_inferred

        self.log_joint = ed.make_log_joint_fn(self.probabilistic_pca)
        self.log_q = ed.make_log_joint_fn(self.variational_model)

    def probabilistic_pca(self):  # (unmodeled) data
        w = ed.Normal(loc=tf.zeros([self.k, self.d]),
                      scale=2.0 * tf.ones([self.k, self.d]),
                      name="w")  # parameter
        z = ed.Normal(loc=tf.zeros([self.n, self.k]),
                      scale=tf.ones([self.n, self.k]),
                      name="z")  # parameter
        x = ed.Normal(loc=tf.matmul(z, w),
                      scale=0.5 * tf.ones([self.n, self.d]),
                      name="x")  # (modeled) data
        return (x, w), z

    def variational_model(self, qz_mean, qz_stddv):
        qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
        return qz

    def target(self, z):
        return self.log_joint(z=z, w=self.w_inferred, x=self.data)

    def target_q(self, qz, qz_mean, qz_stddv):
        return self.log_q(qz=qz, qz_mean=qz_mean, qz_stddv=qz_stddv)

    def vi(self):
        tf.reset_default_graph()

        qz_mean = tf.Variable(np.ones([self.n, self.k]), dtype=tf.float32)
        qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.n, self.k]), dtype=tf.float32))

        qz = self.variational_model(qz_mean=qz_mean, qz_stddv=qz_stddv)

        energy = self.target(qz)
        entropy = -self.target_q(qz=qz, qz_mean=qz_mean, qz_stddv=qz_stddv)

        elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        train = optimizer.minimize(-elbo)

        init = tf.global_variables_initializer()

        t = []

        num_epochs = 300

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    t.append(sess.run([elbo]))

            z_mean_inferred = sess.run(qz_mean)

        return z_mean_inferred, t

    def replace_params(self, z):

        def interceptor(rv_constructor, *rv_args, **rv_kwargs):
            name = rv_kwargs.pop("name")
            if name == "w":
                rv_kwargs["value"] = self.w_inferred
            if name == "z":
                rv_kwargs["value"] = z
            return rv_constructor(*rv_args, **rv_kwargs)

        return interceptor

    def generate(self, z_inferred):

        with ed.interception(self.replace_params(z=z_inferred)):
            generate = self.probabilistic_pca()

        with tf.Session() as sess:
            (x_generated, _), _ = sess.run(generate)

        return x_generated



if __name__ == '__main__':
    num_datapoints = 5000
    data_dim = 2
    latent_dim = 1
    stddv_datapoints = 0.5

    x_train, w_true = build_toy_dataset(num_datapoints, data_dim, latent_dim, sigma=stddv_datapoints)
    x_test = build_test_dataset(1000, data_dim, latent_dim, w_true, sigma=stddv_datapoints)

    plt.figure()
    plt.scatter(x_test[:, 0], x_test[:, 1], alpha=0.7, label='Testing Data')
    plt.scatter(x_train[:, 0], x_train[:, 1], alpha=0.7, label='Training Data')
    plt.legend()

    print('shape of training data:', x_train.shape)
    model = ppca_model(x_train, 1)
    w_hat, z_hat = model.map()
    print('Learned principal axes:')
    print(w_hat)
    print('Learned ratio:')
    print(w_hat[0][0]/w_hat[0][1])

    w_hat2, z_hat2, t = model.vi()
    print('Learned principal axes:')
    print(w_hat2)
    print('Learned ratio:')
    print(w_hat2[0][0]/w_hat2[0][1])

    plt.figure()
    plt.plot(range(1, 300, 5), t)
    plt.xlabel('Iteration')

    x_gen = model.generate(w_hat2, z_hat2)

    plt.figure()
    plt.scatter(x_train[:,0], x_train[:,1], alpha = 0.7, label='Training Data')
    plt.scatter(x_gen[:,0], x_gen[:,1], alpha=0.7, label='Generated Data')
    plt.legend()

    test_model = ppca_predict(x_test, 1, w_hat2)

    z_test, t = test_model.vi()
    plt.figure()
    plt.plot(range(1, 300, 5), t)
    plt.xlabel('Iteration')

    x_gen2 = test_model.generate(z_test)

    plt.figure()
    plt.scatter(x_test[:, 0], x_test[:, 1], alpha=0.7, label='Testing Data')
    plt.scatter(x_gen2[:, 0], x_gen2[:, 1], alpha=0.7, label='Generated Data')
    plt.legend()

    plt.show()





