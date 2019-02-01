import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import argparse

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as bij
from tensorflow_probability import edward2 as ed
import warnings

import pandas as pd
from scipy import linalg
from bayespy import plot as bpplt

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

def build_ARD_dataset():
    np.random.seed(0)

    N = 400; D = 30

    #dimensionality
    k = 10
    t = 2

    # Shared factor loading
    S = 2*np.random.rand(D, k) - 1
    # Target factor loading
    W = 2*np.random.rand(D, t) - 1

    # target latent variables
    t = np.zeros((N,t))
    t[0:100, :] = np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], 100)
    t[100:200, :] = np.random.multivariate_normal([-1, -1], [[0.1, 0], [0, 0.1]], 100)
    t[200:300, :] = np.random.multivariate_normal([-1, 1], [[0.1, 0], [0, 0.1]], 100)
    t[300:400, :] = np.random.multivariate_normal([1, -1], [[0.1, 0], [0, 0.1]], 100)
    labels = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100

    # background latent variables
    zt = np.random.normal(0, 1, (N,k))
    zb = np.random.normal(0, 1, (N,k))

    A = np.matmul(t, W.T) + np.matmul(zt, S.T) + np.random.normal(0, 0.05, (N, D))
    B = np.matmul(zb, S.T) + np.random.normal(0, 0.05, (N, D))

    return A, B, labels

def factor_plot(w, s, fp, target_fn, shared_fn,tick_label=None):
    '''

    :param fn:
    :param fp:
    :param iter:
    :return:
    '''
    print("type-w, type-s:", type(w), type(s))
    order_w = np.argsort(np.linalg.norm(w, axis=1))
    order_w = np.flipud(order_w)

    order_s = np.argsort(np.linalg.norm(s, axis=1))
    order_s = np.flipud(order_s)

    bpplt.pyplot.figure(figsize=(10, 10))
    bpplt.hinton(s[order_s, :].T)
    # if tick_label.any() != None:
    #     nd = w.shape[1]
    #     plt.yticks(np.arange(1, nd + 1), tick_label.reshape(-1, 1))
    plt.savefig(fp + shared_fn)

    bpplt.pyplot.figure(figsize=(10, 10))
    bpplt.hinton(w[order_w, :].T)
    # if tick_label.any() != None:
    #     nd = s.shape[1]
    #     plt.yticks(np.arange(1, nd + 1), tick_label.reshape(-1, 1))
    plt.savefig(fp + target_fn)

class clvm:
    def __init__(self, target_dataset, background_dataset, k_shared=10, k_target=2, TargetARD=False, BackgroundARD=True, seed=0):
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
        #ideally remove defaults and make sure user always enters values
        self.m = target_dataset.shape[1]
        self.nx = target_dataset.shape[0]
        self.ny = background_dataset.shape[0]
        self.target_dataset = target_dataset
        self.background_dataset = background_dataset
        self.TargetARD = TargetARD
        self.BackgroundARD=BackgroundARD
        self.Ks = self.m-1
        self.Ki = 2
        # if self.BackgroundARD:
        #     self.Ks = self.m-3 #latent dimensionality of shared space
        # if self.TargetARD:
        #     self.Ki = self.m-1 #latent dimensionality of indepedent space
        self.k_shared = self.Ks
        self.k_target = self.Ki
        self.log_joint = ed.make_log_joint_fn(self.create_model_shell)
        self.log_q = ed.make_log_joint_fn(self.variational_model)

    def lognormal_q(self, shape, name=None):
        with tf.variable_scope(name, default_name="lognormal_q"):
            rv = ed.Gamma(tf.nn.softplus(self.foo(name=name+"shape", shape=shape)),1.0 / tf.nn.softplus(self.foo(name=name+"scale", shape=shape)),name=name)
            return rv
            min_scale = 1e-5
            loc = tf.get_variable("loc", shape)
            scale = tf.get_variable("scale", shape, initializer=tf.random_normal_initializer(stddev=0.1))
            distribution=ed.Normal(loc, tf.maximum(tf.nn.softplus(scale), min_scale), name="distr")
            rv = ed.TransformedDistribution(distribution,bijector=bij.Exp(), name="rv")
            return rv

    def foo(self, shape, name):
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            v = tf.get_variable(name, shape)
        return v

    def create_model_shell(self, s_concen, s_rate, alpha_concen, alpha_rate, beta_concen, beta_rate, seed=0):  # (unmodeled) data
        #Specify model

        #ARD, Shared space
        if self.BackgroundARD:
            alpha = ed.Gamma(concentration=alpha_concen, rate=alpha_rate, name="alpha")
            # alpha = ed.Gamma(tf.nn.softplus(tf.get_variable("alpha_shape", shape=[self.Ks])),
            #      1.0 / tf.nn.softplus(tf.get_variable("alpha_scale", shape=[self.Ks])),
            #      name="alpha")
            w = ed.Normal(loc=tf.zeros([self.Ks, self.m]), scale=tf.einsum('i,j->ij', tf.reciprocal(alpha), tf.ones([self.m])), name='w')
        else:
            alpha=None
            w = self.foo(name="w_initial", shape=[self.Ks, self.m])


        #ARD, Target space
        if self.TargetARD:
            beta = ed.Gamma(concentration=beta_concen, rate=beta_rate,name="beta")
            # beta = ed.Gamma(tf.nn.softplus(tf.get_variable("beta_shape", shape=[self.Ki])),
            #      1.0 / tf.nn.softplus(tf.get_variable("beta_scale", shape=[self.Ki])),
            #      name="beta")
            bx = ed.Normal(loc=tf.zeros([self.Ki, self.m]),
                         scale=tf.einsum('i,j->ij', tf.reciprocal(beta), tf.ones([self.m])), name="bx")
        else:
            beta=None
            bx = self.foo(shape=[self.Ki, self.m], name='bx_initial')

        #Robustness
        s = ed.Gamma(concentration=s_concen, rate=s_rate, name="s")

        #Latent vectors
        zx = ed.Normal(loc=tf.zeros([self.nx, self.Ks]), scale=tf.ones([self.nx, self.Ks]), name='zx')
        zy = ed.Normal(loc=tf.zeros([self.ny, self.Ks]), scale=tf.ones([self.ny, self.Ks]), name='zy')
        zi = ed.Normal(loc=tf.zeros([self.nx, self.Ki]), scale=tf.ones([self.nx, self.Ki]), name='zi')

        #Observed vectors
        x = ed.Normal(loc=tf.matmul(zx, w) + tf.matmul(zi, bx),
                        scale=s * tf.ones([self.nx, self.m]), name="x")
        y = ed.Normal(loc=tf.matmul(zy, w),
                        scale=tf.reciprocal(s) * tf.ones([self.ny, self.m]),name="y")             
        
        return (x, y), (zx, zy, zi), (s, w, alpha, bx, beta)

    def variational_model(self, qzx_mean, qzx_stddv, qzy_mean, qzy_stddv, qzi_mean, 
        qzi_stddv, qw_mean, qw_stddv, qbx_mean, qbx_stddv, 
        qs_shape, qalpha_shape, qbeta_shape, seed=0):

        variational_model_variables = dict()
        #Robustness
        qs = self.lognormal_q(qs_shape, name="qs")
        variational_model_variables["qs"] = qs

        #ARD, Shared space
        if self.BackgroundARD:
            qalpha = self.lognormal_q(qalpha_shape, name="qalpha")
            qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
            variational_model_variables["qalpha"] = qalpha
            variational_model_variables["qw"] = qw
        else:
            qalpha=None
            qw=None

        #ARD, Target space
        if self.TargetARD: 
            qbeta = self.lognormal_q(qbeta_shape, name="qbeta")
            qbx = ed.Normal(loc=qbx_mean, scale=qbx_stddv, name="qbx")
            variational_model_variables["qbeta"] = qbeta
            variational_model_variables["qbx"] = qbx
        else:
            qbeta=None
            qbx=None

        #Latent vectors
        qzx = ed.Normal(loc=qzx_mean, scale=qzx_stddv, name="qzx")
        qzy = ed.Normal(loc=qzy_mean,scale=qzy_stddv, name="qzy")
        qzi = ed.Normal(loc=qzi_mean,scale=qzi_stddv, name="qzi")
        
        variational_model_variables["qzx"] = qzx
        variational_model_variables["qzy"] = qzy
        variational_model_variables["qzi"] = qzi

        return (qzx, qzy, qzi), (qs, qalpha, qw, qbeta, qbx)

########################################################################

    def target(self, zx, zy, zi, s, beta, bx, w, alpha, s_concen, s_rate, alpha_concen, alpha_rate, beta_concen, beta_rate):
        return self.log_joint(zx=zx, zy=zy, zi=zi, w=w, s=s, x=self.target_dataset, y=self.background_dataset, beta=beta, bx=bx, alpha=alpha,
        s_concen=s_concen, s_rate=s_rate, alpha_concen=alpha_concen, alpha_rate=alpha_rate, beta_concen=beta_concen, beta_rate=beta_rate)

    def target_q(self, qzx, qzy, qzi, qs, qbeta, qbx, qw, qalpha,
                qzx_mean, qzy_mean, qzi_mean, qbx_mean, qw_mean,
                qzx_stddv, qzy_stddv, qzi_stddv, qbx_stddv, qw_stddv,
                qs_shape, qalpha_shape, qbeta_shape):
        return self.log_q(qzx=qzx, qzy=qzy, qzi=qzi, qs=qs, qbeta=qbeta, qbx=qbx, qw=qw, qalpha=qalpha,
                qzx_mean=qzx_mean, qzy_mean=qzy_mean, qzi_mean=qzi_mean, qbx_mean=qbx_mean, qw_mean=qw_mean,
                qzx_stddv=qzx_stddv, qzy_stddv=qzy_stddv, qzi_stddv=qzi_stddv, qbx_stddv=qbx_stddv, qw_stddv=qw_stddv,
                qs_shape=qs_shape, qalpha_shape=qalpha_shape, qbeta_shape=qbeta_shape,)


########################################################################

    def map(self, num_epochs = 13000, plot=True):
        tf.reset_default_graph() #need to do this so that you don't get error that variable already exists!!

        zx = self.foo(name="zx_initial", shape=[self.ny, self.k_shared])
        zy= self.foo(name="zy_initial", shape=[self.ny, self.k_shared])
        zi = self.foo(name="zi_initial", shape=[self.ny, self.k_target])

        alpha_concen=tf.nn.softplus(self.foo(name="alpha_shape", shape=[self.Ks]))
        alpha_rate=1.0 / tf.nn.softplus(self.foo(name="alpha_scale", shape=[self.Ks]))
        beta_concen=tf.nn.softplus(self.foo(name="beta_shape", shape=[self.Ki]))
        beta_rate=1.0 / tf.nn.softplus(self.foo(name="beta_scale", shape=[self.Ki]))
        s_concen=tf.nn.softplus(self.foo(name="s_shape", shape=[1]))
        s_rate=1.0 / tf.nn.softplus(self.foo(name="s_scale", shape=[1]))

        w = tf.get_variable("w_initial", shape=[self.Ks, self.m])
        bx = tf.get_variable("bx_initial", shape=[self.Ki, self.m])
        (_x, _y), (_zx, _zy, _zi), (s, _w, alpha, _bx, beta) = self.create_model_shell(s_concen, s_rate, alpha_concen, alpha_rate, beta_concen, beta_rate)

        energy = -self.target(zx, zy, zi, s, beta, bx, w, alpha, s_concen, s_rate, alpha_concen, alpha_rate, beta_concen, beta_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.04)
        train = optimizer.minimize(energy)

        init = tf.global_variables_initializer()

        learning_curve = []

        with tf.Session() as sess:
            sess.run(init)
            energy_eval = energy.eval()
            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    cE, _cx, _cy, _ci = sess.run([energy, zx, zy, zi])
                    learning_curve.append(cE)

            t_inferred_map = sess.run(zi)
            bx_post = sess.run(bx) 
            w_post = sess.run(w)
        factor_plot(bx_post,w_post,"/Users/prachi.sinha@ibm.com/Desktop/contrastive-LVM/", "map-target_dim_m-1.png", "map-shared_dim_m-3.png")
        if (plot):
            plt.figure()
            plt.plot(range(1, num_epochs, 5), learning_curve)
            plt.title("Learning Curve MAP")

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

    def variational_inference(self, num_epochs = 5000, plot=True):
        tf.reset_default_graph() #need to do this so that you don't get error that variable already exists!!
        alpha_concen=tf.nn.softplus(self.foo(name="alpha_shape", shape=[self.Ks]))
        alpha_rate=1.0 / tf.nn.softplus(self.foo(name="alpha_scale", shape=[self.Ks]))
        beta_concen=tf.nn.softplus(self.foo(name="beta_shape", shape=[self.Ki]))
        beta_rate=1.0 / tf.nn.softplus(self.foo(name="beta_scale", shape=[self.Ki]))
        s_concen=tf.nn.softplus(self.foo(name="s_shape", shape=[1]))
        s_rate=1.0 / tf.nn.softplus(self.foo(name="s_scale", shape=[1]))

        w = tf.get_variable("w_initial", shape=[self.Ks, self.m])
        bx = tf.get_variable("bx_initial", shape=[self.Ki, self.m])
        (_x, _y), (_zx, _zy, _zi), (s, _w, alpha, _bx, beta) = self.create_model_shell(s_concen, s_rate, alpha_concen, alpha_rate, beta_concen, beta_rate)

        qzx_mean = tf.Variable(np.ones([self.ny, self.k_shared]), dtype=tf.float32)
        qzy_mean = tf.Variable(np.ones([self.ny, self.k_shared]), dtype=tf.float32)
        qzi_mean = tf.Variable(np.ones([self.nx, self.k_target]), dtype=tf.float32)
        qzx_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.ny, self.k_shared]), dtype=tf.float32))
        qzy_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.nx, self.k_shared]), dtype=tf.float32))
        qzi_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.ny, self.k_target]), dtype=tf.float32))

        qs_shape = s.shape
        
        if self.BackgroundARD:
            qw_mean = tf.Variable(np.ones([self.Ks, self.m]), dtype=tf.float32)
            qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.Ks, self.m]), dtype=tf.float32))
            qalpha_shape = alpha.shape
        else:
            qw_mean = None
            qw_stddv = None
            qalpha_shape = None

        if self.TargetARD:
            qbx_mean = tf.Variable(np.ones([self.Ki, self.m]), dtype=tf.float32)
            qbx_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.Ki, self.m]), dtype=tf.float32))
            qbeta_shape = beta.shape
        else:
            qbx_mean = None
            qbx_stddv = None
            qbeta_shape = None

        (qzx, qzy, qzi), (qs, qalpha, qw, qbeta, qbx) = self.variational_model(qzx_mean=qzx_mean, qzx_stddv=qzx_stddv, qzy_mean=qzy_mean, qzy_stddv=qzy_stddv, 
        qzi_mean=qzi_mean, qzi_stddv=qzi_stddv, qw_mean=qw_mean, qw_stddv=qw_stddv, qbx_mean=qbx_mean, qbx_stddv=qbx_stddv, 
        qs_shape=qs_shape, qalpha_shape=qalpha_shape, qbeta_shape=qbeta_shape)

        
        
        energy = self.target(qzx, qzy, qzi, qs, qbeta, qbx, qw, qalpha, s_concen, s_rate, alpha_concen, alpha_rate, beta_concen, beta_rate)
        entropy = -self.target_q(qzx, qzy, qzi, qs, qbeta, qbx, qw, qalpha,
                qzx_mean, qzy_mean, qzi_mean, qbx_mean, qw_mean,
                qzx_stddv, qzy_stddv, qzi_stddv, qbx_stddv, qw_stddv,
                qs_shape, qalpha_shape, qbeta_shape)

        elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(-elbo)

        init = tf.global_variables_initializer()

        learning_curve = []

        with tf.Session() as sess:
            sess.run(init)
            elbo_eval = elbo.eval()
            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    learning_curve.append(sess.run([elbo]))

            ti_inferred = sess.run(qzi_mean)
            if self.TargetARD:
                bx_post = sess.run(qbx) 
            else:
                bx_post = bx.eval()
            if self.BackgroundARD:
                w_post = sess.run(qw) 
            else:
                w_post = w.eval()

        print("elbo", elbo_eval)

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

            factor_plot(bx_post,w_post,"/Users/prachi.sinha@ibm.com/Desktop/contrastive-LVM/", "target_dim_m-1.png", "shared_dim_m-3.png")


        return ti_inferred



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shared", default=10)
    parser.add_argument("--k_target", default=2)
    parser.add_argument("--plot", default=True)
    args = parser.parse_args()

    x_train, y_train, labels = build_ARD_dataset()
    print('shape of target data:', x_train.shape)
    print('shape of background data:', y_train.shape)

    model = clvm(x_train, y_train, int(args.k_shared), int(args.k_target))
    model.map(plot=args.plot)
    # model.variational_inference(plot=args.plot)