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
    def __init__(self, target_dataset, background_dataset, k_shared=10, k_target=2, TargetARD=True, BackgroundARD=True, seed=0):

        self.m = target_dataset.shape[1]
        self.nx = target_dataset.shape[0]
        self.ny = background_dataset.shape[0]
        self.target_dataset = target_dataset
        self.background_dataset = background_dataset
        self.k_shared = k_shared
        self.k_target = k_target
        
        # robustness parameter and its correponding inference variable
        self.s = None
        # self.qs = None

        #ARD for shared space and corresponding inference variables
        if BackgroundARD:
            self.alpha = None #vector of precision variables for w
            # self.qalpha = None
            # self.qw = None
        self.w = None #shared factor loading

        #ARD for target space and corresponding inference variables
        if TargetARD:
            self.beta = None
            # self.qbeta = None
            # self.qbx = None
        self.bx = None

        self.log_joint = ed.make_log_joint_fn(self.create_model_shell)
        self.log_q = ed.make_log_joint_fn(self.variational_model)

    def lognormal_q(self, shape, name=None):
        with tf.variable_scope(name, default_name="lognormal_q"):
            min_scale = 1e-5
            loc = tf.get_variable("loc", shape)
            scale = tf.get_variable(
                "scale", shape, initializer=tf.random_normal_initializer(stddev=0.1))
            rv = ed.TransformedDistribution(
                distribution=ed.Normal(loc, tf.maximum(tf.nn.softplus(scale), min_scale)),
                bijector=bij.Exp())
            return rv

    def create_model_shell(self,TargetARD, BackgroundARD, Ks, Ki, seed=0):  # (unmodeled) data
        #Specify model
        if BackgroundARD:
            Ks = self.m-3 #latent dimensionality of shared space
        if TargetARD:
            Ki = self.m-1 #latent dimensionality of indepedent space

        #Robustness
        self.s = ed.Gamma(concentration=1e-3*tf.ones([1]), rate=1e-3*tf.ones([1]))

        #ARD, Shared space
        if BackgroundARD:
            self.alpha = ed.Gamma(concentration=1e-3*tf.ones([Ks]), rate=1e-3*tf.ones([Ks]))
            self.w = ed.Normal(loc=tf.zeros([Ks, self.m]),
                            scale=tf.einsum('i,j->ij', tf.reciprocal(self.alpha), tf.ones([self.m])), name='w')
        else:
            self.w = tf.get_variable(shape=[Ks, self.m], name='W')

        #ARD, Target space
        if TargetARD:
            self.beta = ed.Gamma(concentration=1e-3*tf.ones([Ki]), rate=1e-3*tf.ones([Ki]))
            self.bx = ed.Normal(loc=tf.zeros([Ki, self.m]),
                         scale=tf.einsum('i,j->ij', tf.reciprocal(self.beta), tf.ones([self.m])), name='bx')
        else:
            self.bx = tf.get_variable(shape=[Ki, self.m], name='Bx')

        #Latent vectors
        zx = ed.Normal(loc=tf.zeros([self.nx, Ks]), scale=tf.ones([self.nx, Ks]), name='zx')
        zy = ed.Normal(loc=tf.zeros([self.ny, Ks]), scale=tf.ones([self.ny, Ks]), name='zy')
        zi = ed.Normal(loc=tf.zeros([self.nx, Ki]), scale=tf.ones([self.nx, Ki]), name='zi')

        #Observed vectors
        x = ed.Normal(loc=tf.matmul(zx, self.w) + tf.matmul(zi, self.bx),
                        scale=self.s * tf.ones([self.nx, self.m]))
        y = ed.Normal(loc=tf.matmul(zy, self.w),
                        scale=tf.reciprocal(self.s) * tf.ones([self.ny, self.m]))             
        
        return (x, y), (zx, zy, zi)

    def variational_model(self, TargetARD, BackgroundARD, Ks, Ki, 
        qzx_mean, qzx_stddv, qzy_mean, qzy_stddv, qzi_mean, 
        qzi_stddv, qw_mean, qw_stddv, qbx_mean, qbx_stddv,seed=0):

        variational_model_variables = dict()
        #Robustness
        qs = self.lognormal_q(self.s.shape)
        variational_model_variables["qs"] = qs

        #ARD, Shared space
        if BackgroundARD:
            qalpha = self.lognormal_q(self.alpha.shape)
            qw = ed.Normal(loc=qw_mean, scale=qw_stddv)
            variational_model_variables["qalpha"] = qalpha
            variational_model_variables["qw"] = qw
        else:
            qalpha=None
            qw=None

        #ARD, Target space
        if TargetARD: 
            qbeta = self.lognormal_q(self.beta.shape)
            qbx = ed.Normal(loc=qbx_mean, scale=qbx_stddv)
            variational_model_variables["qbeta"] = qbeta
            variational_model_variables["qbx"] = qbx
        else:
            qbeta=None
            qbx=None

        #Latent vectors
        qzx = ed.Normal(loc=qzx_mean, scale=qzx_stddv)
        qzy = ed.Normal(loc=qzy_mean,scale=qzy_stddv)
        qzi = ed.Normal(loc=qzi_mean,scale=qzi_stddv)
        
        variational_model_variables["qzx"] = qzx
        variational_model_variables["qzy"] = qzy
        variational_model_variables["qzi"] = qzi

        return variational_model_variables

########################################################################

    def target_old(self, zi, zj, ti):
        return self.log_joint(zi=zi, zj=zj, ti=ti, x=self.target_dataset, y=self.background_dataset)

    def target_q_old(self, qzi, qzj, qti, qzi_mean, qzi_stddv, qzj_mean, qzj_stddv, qti_mean, qti_stddv):
        return self.log_q(qzi=qzi, qzj=qzj, qti=qti, qzi_mean=qzi_mean, qzi_stddv=qzi_stddv,
                          qzj_mean=qzj_mean, qzj_stddv=qzj_stddv, 
                          qti_mean=qti_mean, qti_stddv=qti_stddv)
    
    def target(self, zx, zy, zi, s, beta, bx, w, alpha):
        return self.log_joint(zx=zx, zy=zy, zi=zi, s=s, x=self.target_dataset, y=self.background_dataset, beta=beta, bx=bx, w=w, alpha=alpha)

    def target_q(self, qzx, qzy, qzi, qs, qbeta, qbx, qw, qalpha,
                qzx_mean, qzy_mean, qzi_mean, qs_mean, qbeta_mean, qbx_mean, qw_mean, qalpha_mean,
                qzx_stddv, qzy_stddv, qzi_stddv, qs_stddv, qbeta_stddv, qbx_stddv, qw_stddv, qalpha_stddv):
        return self.log_q(qzx=qzx, qzy=qzy, qzi=qzi, qs=qs, qbeta=qbeta, qbx=qbx, qw=qw, qalpha=qalpha,
                qzx_mean=qzx_mean, qzy_mean=qzy_mean, qzi_mean=qzi_mean, qs_mean=qs_mean, qbeta_mean=qbeta_mean, qbx_mean=qbx_mean, qw_mean=qw_mean, qalpha_mean=qalpha_mean,
                qzx_stddv=qzx_stddv, qzy_stddv=qzy_stddv, qzi_stddv=qzi_stddv, qs_stddv=qs_stddv, qbeta_stddv=qbeta_stddv, qbx_stddv=qbx_stddv, qw_stddv=qw_stddv, qalpha_stddv=qalpha_stddv)

    def map(self, num_epochs = 1500, plot=True):
        tf.reset_default_graph() #need to do this so that you don't get error that variable already exists!!

        zi = tf.Variable(np.ones([self.ny, self.k_shared]), dtype=tf.float32)
        zj = tf.Variable(np.ones([self.nx, self.k_shared]), dtype=tf.float32)
        ti = tf.Variable(np.ones([self.nx, self.k_target]), dtype=tf.float32)

        energy = -self.target(zi, zj, ti)

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

    def variational_inference(self, num_epochs = 1500, plot=True):

        tf.reset_default_graph() #need to do this so that you don't get error that variable already exists!!

        qzi_mean = tf.Variable(np.ones([self.ny, self.k_shared]), dtype=tf.float32)
        qzj_mean = tf.Variable(np.ones([self.ny, self.k_shared]), dtype=tf.float32)
        qti_mean = tf.Variable(np.ones([self.nx, self.k_target]), dtype=tf.float32)
        qzi_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.ny, self.k_shared]), dtype=tf.float32))
        qzj_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.nx, self.k_shared]), dtype=tf.float32))
        qti_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.ny, self.k_target]), dtype=tf.float32))

        qzi, qzj, qti = self.variational_model(qzi_mean=qzi_mean, qzi_stddv=qzi_stddv, qzj_mean=qzj_mean,
                                               qzj_stddv=qzj_stddv, qti_mean=qti_mean, qti_stddv=qti_stddv)

        energy = self.target(qzi, qzj, qti)
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

    model = clvm(x_train, y_train, int(args.k_shared), int(args.k_target))
    model.map(plot=args.plot)
    model.variational_inference(plot=args.plot)