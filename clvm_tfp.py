import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import argparse

#from utils.factor_plot import factor_plot

from bayespy import plot as bpplt

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import warnings

from sklearn.externals import joblib

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
    def __init__(self, target_dataset, background_dataset, k_shared=10, k_target=2, robust_flag=False, sharedARD=False,
                 targetARD=False, target_missing=False, background_missing=False):
        """
        Initialization for the clvm class
        :param target_dataset: numpy array of size n (observations) x d (measurements)
        :param background_dataset: numpy array of size m (observations) x d (measurements)
        :param k_shared: integer specifying the dimensionality of the shared latent space
        :param k_target: integer specifying the dimensionality of the target latent space
        :param robust_flag: boolean indicating if inverse gamma prior for noise is used
        :param sharedARD:  boolean indicating if ARD prior is used for shared factor loading
        :param targetARD:  boolean indicating if ARD prior is used for target factor loading
        :param target_missing: boolean indicating if there is missing data in the target dataset;
               missing data should be indicated by elements equal to np.NaN
        :param background_missing:  boolean indicating if there is missing data in the background dataset;
               missing data should be indicated by elements equal to np.NaN
        """

        #perform a few input checks
        if type(target_dataset) is not np.ndarray:
            print('Please input an nd array for the target dataset')
            return
        if type(background_dataset) is not np.ndarray:
            print('Please input an nd array for the background dataset')
            return
        if target_dataset.shape[1] != background_dataset.shape[1]:
            print('Warning: Datasets do not have the same dimensionality')
            return

        self.n, self.d = target_dataset.shape
        self.m = background_dataset.shape[0]
        self.target_dataset = target_dataset
        self.background_dataset = background_dataset
        self.k_shared = k_shared
        self.k_target = k_target

        #flags for model variants
        self.robust = robust_flag
        self.targetARD = targetARD
        self.sharedARD = sharedARD
        self.target_missing = target_missing
        self.background_missing = background_missing

        if self.target_missing:
            tobs = np.ones(self.target_dataset.shape).astype(np.int64)
            tobs[np.isnan(self.target_dataset)] = 0
            tr, tc = np.where(tobs == 1)
            self.idx_obs = np.squeeze(np.dstack((tr, tc)))
            trm, tcm = np.where(tobs == 0)
            self.idx_mis = np.squeeze(np.dstack((trm, tcm)))
            self.target_dataset = self.target_dataset[~np.isnan(self.target_dataset)]

        if self.background_missing:
            bobs = np.ones(self.background_dataset.shape).astype(np.int64)
            bobs[np.isnan(self.background_dataset)] = 0
            br, bc = np.where(bobs == 1)
            self.idy_obs = np.squeeze(np.dstack((br, bc)))
            brm, bcm = np.where(bobs == 0)
            self.idy_mis = np.squeeze(np.dstack((brm, bcm)))
            self.background_dataset = self.background_dataset[~np.isnan(self.background_dataset)]

        # initialize variables for the posterior estimates
        self.ti_hat = None
        self.zi_hat = None
        self.zj_hat = None
        self.w_hat = None
        self.s_hat = None
        self.noise_hat = None
        self.a_hat = None
        self.b_hat = None

        self.log_joint = ed.make_log_joint_fn(self._clvm_model)
        self.log_q = ed.make_log_joint_fn(self._variational_model)

    def _get_parameter(self, shape, name, pos_flag=False):
        """
        Function to automatically create model parameters based on a specified shape and name
        :param shape: list of the shape of the variable
        :param name: string containing the variable name
        :param pos_flag: optional flag indicating whether the variable must be positive
        :return: tensorflow variable
        """
        if pos_flag:
            v = tf.nn.softplus(tf.get_variable(name=name, shape=shape))
        else:
            v = tf.get_variable(name=name, shape=shape)
        return v

    def generate(self, use_inferred=True):
        """
        Function to generate samples from the clvm model; automatically uses the posterior estimates
        :param use_inferred: boolean to indicate if the  inferred latent variables, z_i, z_j, t_i should be used for
        the generative sample; alternative is random draws from prior distributions of z_i, z_j, t_i
        :return: returns a generated sample the same size as the traing data
        """

        if use_inferred:
            var_list = {'zi': self.zi_hat, 'zj': self.zj_hat, 'ti': self.ti_hat}
        else:
            var_list = {}

        if self.targetARD:
            var_list['beta'] = self.b_hat
            var_list['w'] = self.w_hat
        else:
            assign_w = tf.assign(tf.get_default_graph().get_tensor_by_name('clvm_params/w:0'), self.w_hat)
        if self.sharedARD:
            var_list['alpha'] = self.a_hat
            var_list['s'] = self.s_hat
        else:
            assign_s = tf.assign(tf.get_default_graph().get_tensor_by_name('clvm_params/s:0'), self.s_hat)
        if self.robust:
            var_list['noise'] = self.noise_hat
        else:
            assign_noise = tf.assign(tf.get_default_graph().get_tensor_by_name('clvm_params/noise:0'), self.noise_hat)

        with ed.interception(self._make_value_setter(**var_list)):
                generate = self._clvm_model()

        with tf.Session() as sess:
            if not self.sharedARD:
                sess.run(assign_s)
            if not self.targetARD:
                sess.run(assign_w)
            if not self.robust:
                sess.run(assign_noise)
            (x_generated, y_generated), _ = sess.run(generate)

        return x_generated, y_generated

    def _make_value_setter(self, **model_kwargs):
        """Creates a value-setting interceptor."""

        def set_values(f, *args, **kwargs):
            """Sets random variable values to its aligned value."""
            name = kwargs.get("name")
            if name in model_kwargs:
                kwargs["value"] = model_kwargs[name]
            return ed.interceptable(f)(*args, **kwargs)

        return set_values

    def _clvm_model(self):
        """
        Set-up the model based on the flags specified in __init__
        :return: observed and latent variable tuples
        """

        # Latent vectors
        # All models have at least zi, zj, and ti
        zi = ed.Normal(loc=tf.zeros([self.n, self.k_shared]), scale=tf.ones([self.n, self.k_shared]), name='zi')
        zj = ed.Normal(loc=tf.zeros([self.m, self.k_shared]), scale=tf.ones([self.m, self.k_shared]), name='zj')
        ti = ed.Normal(loc=tf.zeros([self.n, self.k_target]), scale=tf.ones([self.n, self.k_target]), name='ti')

        latent_vars = (zi, zj, ti)

        with tf.variable_scope('clvm_params', reuse=tf.AUTO_REUSE):
            # Parameters
            # Depending on the modeling choices, random, unobserved variables will be appended to latent_vars
            # shared factor loading
            if self.sharedARD:
                alpha = ed.Gamma(concentration=1e-3*tf.ones([self.k_shared]),
                                 rate=1e-3*tf.ones([self.k_shared]), name="alpha")
                s = ed.Normal(loc=tf.zeros([self.k_shared, self.d]),
                              scale=tf.einsum('i,j->ij', tf.reciprocal(alpha), tf.ones([self.d])), name="s")
                latent_vars = latent_vars + (alpha, s,)
            else:
                s = self._get_parameter([self.k_shared, self.d], "s")

            # target factor loading
            if self.targetARD:
                beta = ed.Gamma(concentration=1e-3*tf.ones([self.k_target]),
                                rate=1e-3*tf.ones([self.k_target]), name="beta")
                w = ed.Normal(loc=tf.zeros([self.k_target, self.d]),
                              scale=tf.einsum('i,j->ij', tf.reciprocal(beta), tf.ones([self.d])), name="w")
                latent_vars = latent_vars + (beta, w,)
            else:
                w = self._get_parameter([self.k_target, self.d], "w")

            # noise
            if self.robust:
                noise = ed.Gamma(concentration=tf.ones([1]), rate=tf.ones([1]), name='noise')
                latent_vars = latent_vars + (noise,)
            else:
                noise = self._get_parameter([1], "noise", True)

        # Observed vectors, depends on the particular data, therefore not included in parameter scope
        if self.target_missing:
            ix_obs = self.idx_obs
            ix_mis = self.idx_mis
            x = ed.Normal(loc=tf.gather_nd(tf.matmul(zi, s) + tf.matmul(ti, w), ix_obs),
                          scale=tf.gather_nd(noise*tf.ones([self.n, self.d]), ix_obs), name="x")
            x_mis = ed.Normal(loc=tf.gather_nd(tf.matmul(zi, s) + tf.matmul(ti, w), ix_mis),
                              scale=tf.gather_nd(noise*tf.ones([self.n, self.d]), ix_mis), name="x_mis")
            latent_vars = latent_vars + (x_mis, )
        else:
            x = ed.Normal(loc=tf.matmul(zi, s) + tf.matmul(ti, w),
                        scale=noise * tf.ones([self.n, self.d]), name="x")
        if self.background_missing:
            iy_obs = self.idy_obs
            iy_mis = self.idy_mis
            y = ed.Normal(loc=tf.gather_nd(tf.matmul(zj, s), iy_obs),
                          scale=tf.gather_nd(noise*tf.ones([self.m, self.d]), iy_obs), name="y")
            y_mis = ed.Normal(loc=tf.gather_nd(tf.matmul(zj, s), iy_mis),
                              scale=tf.gather_nd(noise*tf.ones([self.m, self.d]), iy_mis), name="y_mis")
            latent_vars = latent_vars + (y_mis, )
        else:
            y = ed.Normal(loc=tf.matmul(zj, s),
                          scale=noise * tf.ones([self.m, self.d]), name="y")

        return (x, y), latent_vars

    def _variational_model(self, params):
        '''
        Set-up the variational approximation based on the flag specified in __init__
        :param params: dictionary of the variational parameters (typically called lambda) that minimize the KL
        divergence between q and the posterior
        :return: tuple of variational random variables and list of associated dictionary keys
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
        q_latent_vars_names = ['zi', 'zj', 'ti']

        with tf.variable_scope('clvm_params'):

            if self.robust:
                qnoise_loc = params['qnoise_loc']
                qnoise_scale = params['qnoise_scale']
                qnoise = ed.RandomVariable(tfp.distributions.LogNormal(loc=qnoise_loc, scale=qnoise_scale))
                q_latent_vars = q_latent_vars + (qnoise, )
                q_latent_vars_names.append('noise')

            if self.sharedARD:
                qalpha_loc = params['qalpha_loc']
                qalpha_scale = params['qalpha_scale']
                qalpha = ed.RandomVariable(tfp.distributions.LogNormal(loc=qalpha_loc, scale=qalpha_scale))

                qs_loc = params['qs_loc']
                qs_scale = params['qs_scale']
                qs = ed.Normal(loc=qs_loc, scale=qs_scale, name="qs")
                q_latent_vars = q_latent_vars + (qs, qalpha, )
                q_latent_vars_names.append('s')
                q_latent_vars_names.append('alpha')

            if self.targetARD:
                qbeta_loc = params['qbeta_loc']
                qbeta_scale = params['qbeta_scale']
                qbeta = ed.RandomVariable(tfp.distributions.LogNormal(loc=qbeta_loc, scale=qbeta_scale))

                qw_loc = params['qw_loc']
                qw_scale = params['qw_scale']
                qw = ed.Normal(loc=qw_loc, scale=qw_scale, name="qw")
                q_latent_vars = q_latent_vars + (qw, qbeta, )
                q_latent_vars_names.append('w')
                q_latent_vars_names.append('beta')

        if self.target_missing:
            qtarget_loc = params['qx_loc']
            qtarget_scale = params['qx_scale']
            qx = ed.Normal(loc=qtarget_loc, scale=qtarget_scale, name="qx")
            q_latent_vars = q_latent_vars + (qx, )
            q_latent_vars_names.append('x_mis')

        if self.background_missing:
            qbackground_loc = params['qy_loc']
            qbackground_scale = params['qy_scale']
            qy = ed.Normal(loc=qbackground_loc, scale=qbackground_scale, name="qy")
            q_latent_vars = q_latent_vars + (qy, )
            q_latent_vars_names.append('y_mis')

        return q_latent_vars, q_latent_vars_names

    def _target(self, target_vars): #might be able to combine target and target_q
        """

        :param target_vars: dictionary of the current target variable values
        :return: log_joint of the clvm model evaluated at target_vars
        """
        zi = target_vars['zi']
        zj = target_vars['zj']
        ti = target_vars['ti']

        noise = []
        s = []
        alpha = []
        w = []
        beta = []
        x_mis = []
        y_mis = []

        if self.robust:
            noise = target_vars['noise']

        if self.sharedARD:
            s = target_vars['s']
            alpha = target_vars['alpha']

        if self.targetARD:
            w = target_vars['w']
            beta = target_vars['beta']

        if self.target_missing:
            x_mis = target_vars['x_mis']

        if self.background_missing:
            y_mis = target_vars['y_mis']

        return self.log_joint(zi=zi, zj=zj, ti=ti, noise=noise, s=s, alpha=alpha, w=w, beta=beta,
                              x_mis=x_mis, y_mis=y_mis, x=self.target_dataset, y=self.background_dataset)

    def _target_q(self, target_vars, params):
        """

        :param target_vars: dictionary of the current variational
        :param params:
        :return:
        """
        qzi = target_vars['zi']
        qzj = target_vars['zj']
        qti = target_vars['ti']

        qnoise = []
        qbeta = []
        qs = []
        qalpha = []
        qw = []
        qx = []
        qy = []

        if self.robust:
            qnoise = target_vars['noise']

        if self.sharedARD:
            qalpha = target_vars['alpha']
            qs = target_vars['s']

        if self.targetARD:
            qbeta = target_vars['beta']
            qw = target_vars['w']

        if self.target_missing:
            qx = target_vars['x_mis']

        if self.background_missing:
            qy = target_vars['y_mis']

        return self.log_q(params, qzi=qzi, qzj=qzj, qti=qti, qnoise=qnoise, qalpha=qalpha, qs=qs, qw=qw, qbeta=qbeta,
                          qx=qx, qy=qy)

    def _initialize_variational_vars(self):
        """

        :return: dictionary of variational model parameters, sets up graph
        """
        qzi_mean = tf.Variable(tf.random.normal([self.n, self.k_shared]), dtype=tf.float32, name='qzi_loc')
        qzj_mean = tf.Variable(tf.random.normal([self.m, self.k_shared]), dtype=tf.float32, name='qzj_loc')
        qti_mean = tf.Variable(tf.random.normal([self.n, self.k_target]), dtype=tf.float32, name='qti_loc')
        qzi_stddv = tf.nn.softplus(
            tf.Variable(tf.random.normal([self.n, self.k_shared], stddev=0.01), dtype=tf.float32), name='qzi_scale')
        qzj_stddv = tf.nn.softplus(
            tf.Variable(tf.random.normal([self.m, self.k_shared], stddev=0.01), dtype=tf.float32), name='qzj_scale')
        qti_stddv = tf.nn.softplus(
            tf.Variable(tf.random.normal([self.n, self.k_target], stddev=0.01), dtype=tf.float32), name='qti_scale')

        params = {'qzi_mean': qzi_mean, 'qzi_stddv': qzi_stddv, 'qti_mean': qti_mean, 'qti_stddv': qti_stddv,
                  'qzj_mean': qzj_mean, 'qzj_stddv': qzj_stddv}

        with tf.variable_scope('clvm_params'):

            if self.robust:
                params['qnoise_loc'] = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='qnoise_loc')
                params['qnoise_scale'] = tf.nn.softplus(tf.Variable(tf.random.normal([1], stddev=0.01), dtype=tf.float32),
                                                        name='qnoise_scale')

            if self.sharedARD:
                params['qs_loc'] = tf.Variable(tf.random.normal([self.k_shared, self.d]), dtype=tf.float32, name='qs_loc')
                params['qs_scale'] = tf.nn.softplus(
                    tf.Variable(tf.random.normal([self.k_shared, self.d], stddev=0.1), dtype=tf.float32), name='qs_scale')

                params['qalpha_loc'] = tf.Variable(tf.random.normal([self.k_shared]), dtype=tf.float32, name='qalpha_loc')
                params['qalpha_scale'] = tf.nn.softplus(
                    tf.Variable(tf.random_normal([self.k_shared], stddev=0.01), dtype=tf.float32), name='alpha_scale')

            if self.targetARD:
                params['qw_loc'] = tf.Variable(tf.random.normal([self.k_target, self.d]), dtype=tf.float32, name='qw_loc')
                params['qw_scale'] = tf.nn.softplus(
                    tf.Variable(tf.random.normal([self.k_target, self.d], stddev=0.1), dtype=tf.float32), name='qw_scale')

                params['qbeta_loc'] = tf.Variable(tf.random_normal([self.k_target]), dtype=tf.float32, name='qbeta_loc')
                params['qbeta_scale'] = tf.nn.softplus(
                    tf.Variable(tf.random.normal([self.k_target], stddev=0.01), dtype=tf.float32), name='qbeta_scale')

        if self.target_missing:
            params['qx_loc'] = tf.Variable(tf.random_normal([self.idx_mis.shape[0]]), name='qx_loc')
            params['qx_scale'] = tf.Variable(tf.nn.softplus(tf.random_normal([self.idx_mis.shape[0]])), name='qx_scale')

        if self.background_missing:
            params['qy_loc'] = tf.Variable(tf.random_normal([self.idy_mis.shape[0]]), name='qy_loc')
            params['qy_scale'] = tf.Variable(tf.nn.softplus(tf.random_normal([self.idy_mis.shape[0]])), name='qy_scale')

        return params


    def variational_inference(self, num_epochs=10000, plot=False, labels=None, seed=1234,
                              fn='model_VI', fp='./results/', saveGraph=False, paramsOnly=True):
        """
        method to perform variational inference on a clvm model
        :param num_epochs: optional, number of interations
        :param plot: optional, boolean if plots should be created
        :param labels: optional, integers of size N
        :param seed: optional, set the random seed
        :param fn: optional, filename to store results
        :param fp: optional, filepath to store the results
        :param saveGraph: optional, boolean to save full tensorflow graph
        :return: MAP estimate of the target latent space
        """

        tf.reset_default_graph() #need to do this so that you don't get error that variable already exists!!

        tf.random.set_random_seed(seed) #set random seed for reproducibility

        params = self._initialize_variational_vars() #initialize the parameters of the variational model
        vars, names = self._variational_model(params)

        qti = vars[names.index('ti')]
        qzi = vars[names.index('zi')]
        qzj = vars[names.index('zj')]

        qtarget_vars = {'zi': qzi, 'zj': qzj, 'ti': qti}

        if self.robust:
            qnoise = vars[names.index('noise')]
            qtarget_vars['noise'] = qnoise

        if self.sharedARD:
            qs = vars[names.index('s')]
            qalpha = vars[names.index('alpha')]
            qtarget_vars['s'] = qs
            qtarget_vars['alpha'] = qalpha

        if self.targetARD:
            qw = vars[names.index('w')]
            qbeta = vars[names.index('beta')]
            qtarget_vars['w'] = qw
            qtarget_vars['beta'] = qbeta

        if self.target_missing:
            qx = vars[names.index('x_mis')]
            qtarget_vars['x_mis'] = qx

        if self.background_missing:
            qy = vars[names.index('y_mis')]
            qtarget_vars['y_mis'] = qy

        energy = self._target(qtarget_vars)
        entropy = -self._target_q(qtarget_vars, params)

        elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(-elbo)

        init = tf.global_variables_initializer()

        if saveGraph:
            if paramsOnly:
                saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'clvm_params'))
            else:
                saver = tf.train.Saver()

        learning_curve = []

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    learning_curve.append(sess.run([elbo]))
                if i % 2000 == 0:
                    #save model every 2000 iterations
                    self._save_MAP(sess, fp, fn, i, learning_curve, seed, plot)

                    if saveGraph:
                        save_path = saver.save(sess, './checkpoint/model' + str(seed) + '.ckpt')
                        print("Model saved in path: %s" % save_path)

            self._save_MAP(sess, fp, fn, i, learning_curve, seed, plot)

            if saveGraph:
                save_path = saver.save(sess, './checkpoint/model' + str(seed) + '.ckpt')
                print("Model saved in path: %s" % save_path)

        if plot:
            plt.figure()
            plt.plot(range(1, num_epochs, 5), learning_curve)
            plt.title("Learning Curve VI")

            plt.figure()
            c = ['k', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                'tab:gray', 'tab:olive', 'tab:cyan']
            ms = ['o', 's', '*', '^', 'v', ',', '<', '>', '8', 'p']
            for i, l in enumerate(np.sort(np.unique(labels))):
                idx = np.where(labels == l)
                plt.scatter(self.ti_hat[idx, 0], self.ti_hat[idx, 1], marker=ms[i], color=c[i])
            plt.title("Target Latent Space VI")

        return self.ti_hat

    def _save_MAP(self, sess, fp, fn, iter, learning_curve, seed, plot):
        """
        internal function to save MAP estimates during VI; saves result to pkl file specified by filename and filepath
        :param sess: tensorflow session
        :param fp: filepath
        :param fn: filename
        :param iter: iteration number
        :param learning_curve: list of objective function values
        :param seed: seed
        :return: None
        """
        self.ti_hat = sess.run(tf.get_default_graph().get_tensor_by_name('qti_loc:0'))
        self.zi_hat = sess.run(tf.get_default_graph().get_tensor_by_name('qzi_loc:0'))
        self.zj_hat = sess.run(tf.get_default_graph().get_tensor_by_name('qzj_loc:0'))

        if self.sharedARD:
            self.s_hat = sess.run(tf.get_default_graph().get_tensor_by_name('clvm_params/qs_loc:0'))
            self.a_hat = sess.run(tf.get_default_graph().get_tensor_by_name('clvm_params/qalpha_loc:0'))
            if plot:
                factor_plot(self.s_hat, self.a_hat, fp, fn + 'VI_sharedFL' + str(seed) + '.png')
        else:
            self.s_hat = sess.run(tf.get_default_graph().get_tensor_by_name('clvm_params/s:0'))
            self.a_hat = None

        if self.targetARD:
            self.w_hat = sess.run(tf.get_default_graph().get_tensor_by_name('clvm_params/qw_loc:0'))
            self.b_hat = sess.run(tf.get_default_graph().get_tensor_by_name('clvm_params/qbeta_loc:0'))
            if plot:
                factor_plot(self.w_hat, self.b_hat, fp, fn + 'VI_targetFL' + str(seed) + '.png')
        else:
            self.w_hat = sess.run(tf.get_default_graph().get_tensor_by_name('clvm_params/w:0'))
            self.b_hat = None

        if self.robust:
            self.noise_hat = sess.run(tf.get_default_graph().get_tensor_by_name('clvm_params/qnoise_loc:0'))
        else:
            self.noise_hat = sess.run(tf.get_default_graph().get_tensor_by_name('clvm_params/noise:0'))

        if self.target_missing:
            x_hat = sess.run(tf.get_default_graph().get_tensor_by_name('qx_loc:0'))
        else:
            x_hat = None

        if self.background_missing:
            y_hat = sess.run(tf.get_default_graph().get_tensor_by_name('qy_loc:0'))
        else:
            y_hat = None

        model = {'lb': learning_curve,
                 'S': self.s_hat,
                 'W': self.w_hat,
                 'noise': self.noise_hat,
                 'alpha': self.a_hat,
                 'beta': self.b_hat,
                 'ti': self.ti_hat,
                 'zi': self.zi_hat,
                 'zj': self.zj_hat,
                 'x': x_hat,
                 'y': y_hat,
                 'k_shared': self.k_shared,
                 'k_target': self.k_target,
                 'sharedARD': self.sharedARD,
                 'targetARD': self.targetARD,
                 'robust': self.robust}

        save_name = fp + fn + str(seed) + 'iter' + str(iter) + '.pkl'
        joblib.dump(model, save_name)

    def restore_graph(self, fl="./checkpoint/Finalmodel.ckpt", num_epochs=10000, plot=False, labels=None, seed=1234,
                      fn='model_VI', fp='./results/', saveGraph=False, paramsOnly=True):
        """
        function to restore the values of a previously trained clvm
        :param fn: filename of the checkpoint file
        :param predict: optional, boolean to indicate if predictions are being made
        :param num_epochs: optional, number of iterations if predictions are being made
        :param plot: optional, boolean if plots should be generated
        :param labels: optional, integers of size N
        :return: None
        """
        params = self._initialize_variational_vars()
        vars, names = self._variational_model(params)
        qti = vars[names.index('ti')]
        qzi = vars[names.index('zi')]
        qzj = vars[names.index('zj')]

        qtarget_vars = {'zi': qzi, 'zj': qzj, 'ti': qti}

        if self.robust:
            qnoise = vars[names.index('noise')]
            qtarget_vars['noise'] = qnoise

        if self.sharedARD:
            qs = vars[names.index('s')]
            qalpha = vars[names.index('alpha')]
            qtarget_vars['s'] = qs
            qtarget_vars['alpha'] = qalpha

        if self.targetARD:
            qw = vars[names.index('w')]
            qbeta = vars[names.index('beta')]
            qtarget_vars['w'] = qw
            qtarget_vars['beta'] = qbeta

        if self.target_missing:
            qx = vars[names.index('x_mis')]
            qtarget_vars['x_mis'] = qx

        if self.background_missing:
            qy = vars[names.index('y_mis')]
            qtarget_vars['y_mis'] = qy

        energy = self._target(qtarget_vars)
        entropy = -self._target_q(qtarget_vars, params)

        elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(-elbo)

        if paramsOnly:
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'clvm_params'))
        else:
            saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        learning_curve = []

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, fl)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    learning_curve.append(sess.run([elbo]))
                if i % 2000 == 0:
                    # save model every 2000 iterations
                    self._save_MAP(sess, fp, fn, i, learning_curve, seed)

                    if saveGraph:
                        save_path = saver.save(sess, './checkpoint/model' + str(seed) + '.ckpt')
                        print("Model saved in path: %s" % save_path)

            self._save_MAP(sess, fp, fn, i, learning_curve, seed)

            if saveGraph:
                save_path = saver.save(sess, './checkpoint/model' + str(seed) + '.ckpt')
                print("Model saved in path: %s" % save_path)

        if plot:
            plt.figure()
            plt.plot(range(1, num_epochs, 5), learning_curve)
            plt.title("Learning Curve VI")

            plt.figure()
            c = ['k', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                 'tab:gray', 'tab:olive', 'tab:cyan']
            ms = ['o', 's', '*', '^', 'v', ',', '<', '>', '8', 'p']
            for i, l in enumerate(np.sort(np.unique(labels))):
                idx = np.where(labels == l)
                plt.scatter(self.ti_hat[idx, 0], self.ti_hat[idx, 1], marker=ms[i], color=c[i])
            plt.title("Target Latent Space VI")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shared", default=10)
    parser.add_argument("--k_target", default=2)
    parser.add_argument("--plot", default=True)
    parser.add_argument("--robust", default=False)
    parser.add_argument("--sharedARD", default=False)
    parser.add_argument("--targetARD", default=False)
    parser.add_argument("--num_epochs", default=5000)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    x_train, y_train, labels = build_toy_dataset()
    print('shape of target data:', x_train.shape)
    print('shape of background data:', y_train.shape)

    model = clvm(x_train, y_train, int(args.k_shared), int(args.k_target), robust_flag=args.robust,
                 sharedARD=args.sharedARD, targetARD=args.targetARD)
    #model.map(plot=args.plot, num_epochs=int(args.num_epochs), labels=labels)
    model.variational_inference(plot=args.plot, num_epochs=int(args.num_epochs), labels=labels, seed=int(args.seed))
    x_gen, y_gen = model.generate()

    plt.figure()
    plt.scatter(x_gen[:,0], x_gen[:,1], alpha=0.7, label='Generated Data')
    plt.scatter(x_train[:,0], x_train[:,1], alpha=0.7, label='Training Data')
    plt.legend()

    plt.show()





