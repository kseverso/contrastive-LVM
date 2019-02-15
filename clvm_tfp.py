import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import argparse

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
Variable names consistent with those in
"Unsupervised Learning with Contrastive Latent Variable Models" 
except loading factor dimensionalities k and t --> k_shared and k_target

x, y = oberved data with dimensions x: d x n and y: d x m
zi, zj = shared latent variables with dimensions: k_shared
ti = target latent variables with dimensions: k_target
qzi, qzj, qti = variational gaussian rep for zi, zj, ti respectively
s = shared factor loading with dimensions: d x k_shared
w = target factor loading with dimensions: d x k_target
noise = noise
"""

class clvm:
    def __init__(self, target_dataset, background_dataset, k_shared=10, k_target=2, robust_flag=False):

        self.n, self.d = target_dataset.shape
        self.m = background_dataset.shape[0]
        self.target_dataset = target_dataset
        self.background_dataset = background_dataset
        self.k_shared = k_shared
        self.k_target = k_target

        #flags for model variants
        self.robust = robust_flag

        self.log_joint = ed.make_log_joint_fn(self.clvm_model)
        self.log_q = ed.make_log_joint_fn(self.variational_model)

    def get_parameter(self, shape, name, pos_flag=False):
        if pos_flag:
            v = tf.nn.softplus(tf.get_variable(name=name, shape=shape))
        else:
            v = tf.get_variable(name=name, shape=shape)
        return v

    def trainable_lognormal(self, shape, name=None):
        with tf.variable_scope(name, default_name="lognormal"):
            min_scale = 1e-5
            loc = tf.get_variable("loc", shape)
            scale = tf.get_variable(
                "scale", shape, initializer=tf.random_normal_initializer(stddev=0.1))
            rv = ed.TransformedDistribution(
                distribution=ed.Normal(loc, tf.maximum(tf.nn.softplus(scale), min_scale)),
                bijector=tfp.bijectors.Exp(), name=name)
            return rv

    def trainable_gamma(self, shape, name=None):
        """Learnable Gamma via shape and scale parameterization."""
        with tf.variable_scope(None, default_name="trainable_gamma"):
            return ed.Gamma(tf.nn.softplus(tf.get_variable("shape", shape)),
                            1.0 / tf.nn.softplus(tf.get_variable("scale", shape)),
                            name=name)

    def make_value_setter(self, **model_kwargs):
        """Creates a value-setting interceptor."""

        def set_values(f, *args, **kwargs):
            """Sets random variable values to its aligned value."""
            name = kwargs.get("name")
            if name in model_kwargs:
                kwargs["value"] = model_kwargs[name]
            return ed.interceptable(f)(*args, **kwargs)

        return set_values

    def clvm_model(self):  # (unmodeled) data
        # note: using initializer=tf.ones([self.k_shared, self.d], dtype=tf.float32)) caused issues

        # Latent vectors
        zi = ed.Normal(loc=tf.zeros([self.n, self.k_shared]), scale=tf.ones([self.n, self.k_shared]), name='zi')
        zj = ed.Normal(loc=tf.zeros([self.m, self.k_shared]), scale=tf.ones([self.m, self.k_shared]), name='zj')
        ti = ed.Normal(loc=tf.zeros([self.n, self.k_target]), scale=tf.ones([self.n, self.k_target]), name='ti')

        latent_vars = (zi, zj, ti)

        # Parameters
        # shared factor loading
        s = self.get_parameter([self.k_shared, self.d], "s")
        # target factor loading
        w = self.get_parameter([self.k_target, self.d], "w")
        # noise
        if self.robust:
            noise = ed.Gamma(concentration=1e-3*tf.ones([1]), rate=1e-3*tf.ones([1]), name='noise')
            latent_vars = latent_vars + (noise,)
        else:
            noise = self.get_parameter([1], "noise", True)

        # Observed vectors
        x = ed.Normal(loc=tf.matmul(zi, s) + tf.matmul(ti, w),
                      scale=noise * tf.ones([self.n, self.d]), name="x")
        y = ed.Normal(loc=tf.matmul(zj, s),
                      scale=noise * tf.ones([self.m, self.d]), name="y")

        return (x, y), latent_vars

    def variational_model(self, params):
        '''

        :param params: dictionary of the variational parameters (typically called lambda) that minimize the KL
        divergence between q and the posterior
        :return:
        '''
        qzi_mean = params['qzi_mean']
        qzi_stddv = params['qzi_stddv']
        qzj_mean = params['qzj_mean']
        qzj_stddv = params['qzj_stddv']
        qti_mean = params['qti_mean']
        qti_stddv = params['qti_stddv']

        qzi = ed.Normal(loc=qzi_mean, scale=qzi_stddv, name="qzi")
        qzj = ed.Normal(loc=qzj_mean, scale=qzj_stddv, name="qzj")
        qti = ed.Normal(loc=qti_mean, scale=qti_stddv, name="qti")

        q_latent_vars = (qzi, qzj, qti)

        if self.robust:
            #qnoise = self.trainable_lognormal(tf.ones([1]), "qnoise")
            qnoise_conc = params['qnoise_conc']
            qnoise_rate = params['qnoise_rate']
            #qnoise = ed.Gamma(concentration=qnoise_conc, rate=qnoise_rate, name="qnoise")
            qnoise = ed.RandomVariable(tfp.distributions.LogNormal(loc=qnoise_conc, scale=qnoise_rate))
            q_latent_vars = q_latent_vars + (qnoise, )

        return q_latent_vars

    def target(self, target_vars):
        zi = target_vars['zi']
        zj = target_vars['zj']
        ti = target_vars['ti']
        if self.robust:
            noise = target_vars['noise']
            lj = self.log_joint(zi=zi, zj=zj, ti=ti, noise=noise, x=self.target_dataset, y=self.background_dataset)
        else:
            lj = self.log_joint(zi=zi, zj=zj, ti=ti, x=self.target_dataset, y=self.background_dataset)
        return lj

    def target_q(self, target_vars, params):
        qzi = target_vars['zi']
        qzj = target_vars['zj']
        qti = target_vars['ti']
        if self.robust:
            qnoise = target_vars['noise']
            ljq = self.log_q(params, qzi=qzi, qzj=qzj, qti=qti, qnoise=qnoise)
        else:
            ljq = self.log_q(params, qzi=qzi, qzj=qzj, qti=qti)
        return ljq

    def map(self, num_epochs = 1500, plot=True):
        tf.reset_default_graph() #need to do this so that you don't get error that variable already exists!!

        zi = tf.Variable(np.ones([self.n, self.k_shared]), dtype=tf.float32)
        zj = tf.Variable(np.ones([self.m, self.k_shared]), dtype=tf.float32)
        ti = tf.Variable(np.ones([self.n, self.k_target]), dtype=tf.float32)

        target_vars = {'zi': zi, 'zj': zj, 'ti': ti}

        if self.robust:
            target_vars['noise'] = tf.Variable(tf.ones([1]), dtype=tf.float32)

        energy = -self.target(target_vars)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        train = optimizer.minimize(energy)

        init = tf.global_variables_initializer()

        learning_curve = []

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    cE, _cx, _cy, _ci = sess.run([energy, zi, zj, ti])
                    learning_curve.append(cE)

            t_inferred_map = sess.run(ti)
        if (plot):
            print('MAP ti shape:', t_inferred_map.shape)

            c = ['k', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                'tab:gray', 'tab:olive', 'tab:cyan']
            ms = ['o', 's', '*', '^', 'v', ',', '<', '>', '8', 'p']
            plt.figure()
            for i, l in enumerate(np.sort(np.unique(labels))):
                idx = np.where(labels == l)
                plt.scatter(t_inferred_map[idx, 0], t_inferred_map[idx, 1], marker=ms[i], color=c[i])
            plt.title("Target Latent Space MAP")
            plt.show()

        return t_inferred_map

    def variational_inference(self, num_epochs=2500, plot=True):

        tf.reset_default_graph() #need to do this so that you don't get error that variable already exists!!

        qzi_mean = tf.Variable(np.ones([self.n, self.k_shared]), dtype=tf.float32)
        qzj_mean = tf.Variable(np.ones([self.m, self.k_shared]), dtype=tf.float32)
        qti_mean = tf.Variable(np.ones([self.n, self.k_target]), dtype=tf.float32)
        qzi_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.n, self.k_shared]), dtype=tf.float32))
        qzj_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.m, self.k_shared]), dtype=tf.float32))
        qti_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.n, self.k_target]), dtype=tf.float32))

        params = {'qzi_mean': qzi_mean, 'qzi_stddv': qzi_stddv, 'qti_mean': qti_mean, 'qti_stddv': qti_stddv,
                  'qzj_mean': qzj_mean, 'qzj_stddv': qzj_stddv}

        if self.robust:
            params['qnoise_conc'] = tf.Variable(1e-2*tf.ones([1]), dtype=tf.float32)
            params['qnoise_rate'] = tf.nn.softplus(1e-2*tf.Variable(np.ones([1]), dtype=tf.float32))

        if self.robust:
            qzi, qzj, qti, qnoise = self.variational_model(params)
            qtarget_vars = {'zi': qzi, 'zj': qzj, 'ti': qti, 'noise': qnoise}
        else:
            qzi, qzj, qti = self.variational_model(params)
            qtarget_vars = {'zi': qzi, 'zj': qzj, 'ti': qti}

        energy = self.target(qtarget_vars)
        entropy = -self.target_q(qtarget_vars, params)

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

            ti_inferred = sess.run(qti_mean)

        if (plot):
            plt.figure()
            plt.plot(range(1, num_epochs, 5), learning_curve)
            plt.title("Learning Curve VI")

            print('VI ti shape:', ti_inferred.shape)

            plt.figure()
            c = ['k', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                'tab:gray', 'tab:olive', 'tab:cyan']
            ms = ['o', 's', '*', '^', 'v', ',', '<', '>', '8', 'p']
            for i, l in enumerate(np.sort(np.unique(labels))):
                idx = np.where(labels == l)
                plt.scatter(ti_inferred[idx, 0], ti_inferred[idx, 1], marker=ms[i], color=c[i])
            plt.title("Target Latent Space VI")

            plt.show()


        return ti_inferred



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shared", default=10)
    parser.add_argument("--k_target", default=2)
    parser.add_argument("--plot", default=True)
    args = parser.parse_args()

    x_train, y_train, labels = build_toy_dataset()
    print('shape of target data:', x_train.shape)
    print('shape of background data:', y_train.shape)

    model = clvm(x_train, y_train, int(args.k_shared), int(args.k_target), robust_flag=True)
    model.map(plot=args.plot)
    model.variational_inference(plot=args.plot)






