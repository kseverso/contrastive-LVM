# Issue with using matplotlib --> Terminating app due to uncaught exception
# 'NSInvalidArgumentException', reason: '-[NSApplication _setup:]: unrecognized
# selector sent to instance 0x7ff11e54fd60' --> seems to be an issue with macosx

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import numpy.linalg as LA

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import warnings
from sklearn.externals import joblib

# plt.style.use("ggplot")
warnings.filterwarnings('ignore')

# Consider a data set X = {x_n} of N data points, where each data point is
# D-dimensional. We aim to represent each x_n under a latent variable z_n lower
# dimension, K < D$. The set of principal axes W relates the latent variables to
# the data. Specifically, we assume that each latent variable is normally
# distributed, z_n ~ N(0, I). The corresponding data point is generated via a
# projection, x_n|z_n ~ N(Wz_n,sigma^2*I), where the matrix W (dim: D x K) are
# known as the principal axes. In probabilistic PCA, we are typically interested
# in estimating the principal axes W and the noise term sigma^2. Probabilistic
# PCA generalizes classical PCA. Marginalizing out the the latent variable, the
# distribution of each data point is x_n ~ N(0, WW^T + sigma^2*I). Classical PCA
# is the specific case of probabilistic PCA when the covariance of the noise
# becomes infinitesimally small, sigma^2 -> 0.
#
# We set up our model below. In our analysis, we assume sigma is known, and
# instead of point estimating W as a model parameter, we place a prior over it
# in order to infer a distribution over principal axes.

def build_outlier_dataset():
    np.random.seed(0)

    N = 400; D = 30; gap=3
    # In B, all the data pts are from the same distribution, which has different variances in three subspaces.
    B = np.zeros((N, D))
    B[:,0:10] = np.random.normal(0,10,(N,10))
    B[:,10:20] = np.random.normal(0,3,(N,10))
    B[:,20:30] = np.random.normal(0,1,(N,10))

    # In A there are four clusters.
    A = np.zeros((N, D))
    A[:,0:10] = np.random.normal(0,10,(N,10))
    # group 1
    A[0:100, 10:20] = np.random.normal(0,1,(100,10))
    A[0:100, 20:30] = np.random.normal(0,1,(100,10))
    # group 2
    A[100:200, 10:20] = np.random.normal(0,1,(100,10))
    A[100:200, 20:30] = np.random.normal(gap,1,(100,10))
    # group 3
    A[200:300, 10:20] = np.random.normal(2*gap,1,(100,10))
    A[200:300, 20:30] = np.random.normal(0,1,(100,10))
    # group 4
    A[300:400, 10:20] = np.random.normal(2*gap,1,(100,10))
    A[300:400, 20:30] = np.random.normal(gap,1,(100,10))
    labels = [0]*100+[1]*100+[2]*100+[3]*100

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(A)
    # plt.colorbar()

    #randomly select some points and randomly add values
    np.random.seed(0)
    randi = np.random.randint(0, 399, 20)
    randj = np.random.randint(0, 399, 20)

    for i in np.arange(20):
        if np.random.rand() > 0.5:
            scale = 1.
        else:
            scale = -1.
        #A[randi[i], :] = A[randi[i], :] + 2 * np.random.randn() + scale * np.random.randint(5, 10, size=30)
        A[randi[i], :] = 40 * np.random.rand(30) - 20

    for j in np.arange(20):
        B[randj[j], :] = 40*np.random.rand(30) - 20

    # plt.subplot(1,2,2)
    # plt.imshow(A)
    # plt.colorbar()

    # plt.show()

    # Perform mean-centering
    mB = np.mean(B, axis=0)
    B = B - mB

    mA = np.mean(A, axis=0)
    A = A - mA

    return A, B, labels

def build_dataset():
    np.random.seed(0)

    N = 400; D = 30; gap=3
    # In B, all the data pts are from the same distribution, which has different variances in three subspaces.
    B = np.zeros((N, D))
    B[:,0:10] = np.random.normal(0,10,(N,10))
    B[:,10:20] = np.random.normal(0,3,(N,10))
    B[:,20:30] = np.random.normal(0,1,(N,10))

    # In A there are four clusters.
    A = np.zeros((N, D))
    A[:,0:10] = np.random.normal(0,10,(N,10))
    # group 1
    A[0:100, 10:20] = np.random.normal(0,1,(100,10))
    A[0:100, 20:30] = np.random.normal(0,1,(100,10))
    # group 2
    A[100:200, 10:20] = np.random.normal(0,1,(100,10))
    A[100:200, 20:30] = np.random.normal(gap,1,(100,10))
    # group 3
    A[200:300, 10:20] = np.random.normal(2*gap,1,(100,10))
    A[200:300, 20:30] = np.random.normal(0,1,(100,10))
    # group 4
    A[300:400, 10:20] = np.random.normal(2*gap,1,(100,10))
    A[300:400, 20:30] = np.random.normal(gap,1,(100,10))
    labels = [0]*100+[1]*100+[2]*100+[3]*100

    # Perform mean-centering
    mB = np.mean(B, axis=0)
    B = B - mB

    mA = np.mean(A, axis=0)
    A = A - mA

    return A, B, labels

class PcPCA:

  def __init__(self, background, target, Ks, Ki, seed=0):
    self.Ks = Ks
    self.Ki = Ki

    self.m = background.shape[1]
    self.nx = target.shape[0]
    self.ny = background.shape[0]

    # latent variables and their corresponding inference variables
    self.zx = None #target class, shared space
    self.zy = None #background class, shared space
    self.zi = None #target class, target space
    self.qzx = None
    self.qzy = None
    self.qzi = None

    # model parameters
    self.s = None # Noise parameter
    self.w = None # shared factor loading
    self.bx = None # Target factor loading

    #Observed data RVs
    self.x = None
    self.y = None

    # Inference variable
    self.Q = None

    self.log_joint = ed.make_log_joint_fn(self.create_model_shell)
    self.log_q = ed.make_log_joint_fn(self.variational_model)
    self.x_train, self.y_train = self.build_dataset()


  def create_model_shell(self,seed=0):
    np.random.seed(seed)

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      #Noise parameter
      self.s = tf.get_variable(shape=[1], name='sigmax', initializer=tf.ones_initializer())

      #Shared factor loading
      self.w = tf.get_variable(shape=[self.Ks, self.m], name='W')

      #Target factor loading
      self.bx = tf.get_variable(shape=[self.Ki, self.m], name='Bx')

      #Latent vectors
      self.zx = ed.Normal(loc=tf.zeros([self.nx, self.Ks]), scale=tf.ones([self.nx, self.Ks]), name='zx')
      self.zy = ed.Normal(loc=tf.zeros([self.ny, self.Ks]), scale=tf.ones([self.ny, self.Ks]), name='zy')
      self.zi = ed.Normal(loc=tf.zeros([self.nx, self.Ki]), scale=tf.ones([self.nx, self.Ki]), name='zi')

      #Observed vectors
      self.x = ed.Normal(loc=tf.matmul(self.zx, self.w) + tf.matmul(self.zi, self.bx),
                      scale=self.s * tf.ones([self.nx, self.m]), name="x")
      self.y = ed.Normal(loc=tf.matmul(self.zy, self.w),
                      scale=tf.reciprocal(self.s) * tf.ones([self.ny, self.m]), name="y")
    
    return (self.x, self.y), (self.s, self.w, self.bx), (self.zx, self.zy, self.zi)

  # def probabilistic_pca(self): # (unmodeled) data
  #   w = ed.Normal(loc=tf.zeros([self.data_dim, self.latent_dim]),
  #                 scale=2.0 * tf.ones([self.data_dim, self.latent_dim]),
  #                 name="w")  # parameter N(0, 2.0), DxK
  #   z = ed.Normal(loc=tf.zeros([self.latent_dim, self.num_datapoints]),
  #                 scale=tf.ones([self.latent_dim, self.num_datapoints]), 
  #                 name="z")  # parameter N(0, 1.0) KxN
  #   x = ed.Normal(loc=tf.matmul(w, z),
  #                 scale=self.stddv_datapoints * tf.ones([self.data_dim, self.num_datapoints]),
  #                 name="x")  # (modeled) data N(?, 1.0), DxN
  #   return x, (w, z)

  # ## The Data

  # We can use the Edward2 model to generate data.

  def build_dataset(self):
    model = self.create_model_shell()
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      (x_training, y_training), (s, w, bx), (zx, zy, zi) = sess.run(model)

    # print("Principal axes:")
    # print(actual_w)
    # print("Actual Ratio:")
    # print(actual_w[0][0]/actual_w[1][0])

    # We visualize the dataset
    # plt.scatter(x_training[0, :], x_training[1, :], color='blue', alpha=0.1)
    # plt.axis([-20, 20, -20, 20])
    # plt.title("Data set")
    # plt.show()

    return x_training, y_training

  def variational_model(self, qzx_mean, qzx_stddv, qzy_mean, qzy_stddv,qzi_mean, qzi_stddv):
    self.qzx = ed.Normal(loc= qzx_mean, scale=qzx_stddv, name="qzx")
    self.qzy = ed.Normal(loc=qzy_mean, scale=qzy_stddv, name="qzy")
    self.qzi = ed.Normal(loc=qzi_mean, scale=qzi_stddv, name="qzi")
    
    return self.qzx, self.qzy, self.qzi

  def target_q(self, qzx, qzy, qzi, qzx_mean, qzx_stddv, qzy_mean, qzy_stddv,qzi_mean, qzi_stddv):
    return self.log_q(qzx_mean=qzx_mean, qzx_stddv=qzx_stddv, qzy_mean=qzy_mean, qzy_stddv=qzy_stddv,qzi_mean=qzi_mean, qzi_stddv=qzi_stddv,
                qzx=qzx, qzy=qzy, qzi=qzi)

  def target(self, zx, zy, zi):
    """Unnormalized target density as a function of the parameters."""
    return self.log_joint(Ks = self.Ks, Ki = self.Ki,
                  zx=zx, zy=zy, zi=zi, x=self.x_train, y=self.y_train)

  # def map(self, num_epochs=200, graphStepSize = 5):
  #   # tf.reset_default_graph()

  #   w = tf.Variable(np.ones([self.data_dim, self.latent_dim]), dtype=tf.float32)
  #   z = tf.Variable(np.ones([self.latent_dim, self.num_datapoints]), dtype=tf.float32)

  #   energy = -self.target(w, z)

  #   optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
  #   train = optimizer.minimize(energy)

  #   init = tf.global_variables_initializer()

  #   t = []

  #   with tf.Session() as sess:
  #     sess.run(init)

  #     for i in range(num_epochs):
  #       sess.run(train)
  #       if i % graphStepSize == 0:
  #         cE, cw, cz = sess.run([energy, w, z])
  #         t.append(cE)

  #     w_inferred_map = sess.run(w)
  #     z_inferred_map = sess.run(z)

  #     return w_inferred_map, z_inferred_map

  def variational_inference(self):
    qzx_mean = tf.get_variable("qzx/loc", [self.nx, self.Ks])
    qzx_stddv = tf.nn.softplus(tf.get_variable("qzx/scale", [self.nx, self.Ks]))
    qzy_mean= tf.get_variable("qzy/loc", [self.ny, self.Ks])
    qzy_stddv=tf.nn.softplus(tf.get_variable("qzy/scale", [self.ny, self.Ks]))
    qzi_mean=tf.get_variable("qzi/loc", [self.nx, self.Ki])
    qzi_stddv=tf.nn.softplus(tf.get_variable("qzi/scale", [self.nx, self.Ki]))

    qzx, qzy, qzi = self.variational_model(qzx_mean=qzx_mean, qzx_stddv=qzx_stddv, qzy_mean=qzy_mean, qzy_stddv=qzy_stddv,qzi_mean=qzi_mean, qzi_stddv=qzi_stddv)

    energy = self.target(qzx, qzy, qzi)
    entropy = -self.target_q(qzx, qzy, qzi, qzx_mean, qzx_stddv, qzy_mean, qzy_stddv,qzi_mean, qzi_stddv)

    elbo = energy + entropy

    return elbo, qzx_mean, qzx_stddv, qzy_mean, qzy_stddv,qzi_mean, qzi_stddv


  def learn_model(self, lrt=0.05, num_epochs=100, graphStepSize = 5, seed=0, fn='PcPCA'):
    elbo, qzx_mean, qzx_stddv, qzy_mean, qzy_stddv,qzi_mean, qzi_stddv = self.variational_inference()
    optimizer = tf.train.AdamOptimizer(learning_rate=lrt)
    train = optimizer.minimize(-elbo)
    # try:
    #   train = optimizer.minimize(-elbo)
    # except ValueError:
    #   print("shape", tf.shape(elbo))
    #   print(optimizer.variables())
    

    init = tf.global_variables_initializer()

    t = []

    with tf.Session() as sess:
      sess.run(init)

      for i in range(num_epochs):
        sess.run(train)
        if i % graphStepSize == 0:
          t.append(sess.run([elbo]))

      z_post = sess.run(qzi_mean)
      # zi_stddv_inferred = sess.run(qzi_stddv)
      zx_post = sess.run(qzx_mean)
      # zx_stddv_inferred = sess.run(qzx_stddv)
      zy_post = sess.run(qzy_mean)
      # zy_stddv_inferred = sess.run(qzy_stddv)
      # z_post = sess.run(self.qzi.mean())
      # zx_post = sess.run(self.qzx.mean())
      # zy_post = sess.run(self.qzy.mean())

      bx_post = self.bx.eval()
      w_post = self.w.eval()
      s_post = self.s.eval()

      model = {'lb': t,
                 'S': w_post,
                 'W': bx_post,
                 'sigma': s_post,
                 'zx': zx_post,
                 'zy': zy_post,
                 'zi': z_post}

    bx_col = LA.norm(bx_post, axis=1)
    order_b = np.argsort(bx_col)
    c = ['k', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    ms = ['o', 's', '*', '^', 'v', ',', '<', '>', '8', 'p']
    plt.figure()
    for i, l in enumerate(np.sort(np.unique(labels))):
        idx = np.where(labels==l)
        plt.scatter(z_post[idx, order_b[-1]], z_post[idx, order_b[-2]], marker=ms[i], color=c[i])
    plt.title("Target Latent Space")
    plt.savefig('./experiments/Edward/' + 't' + str(seed) + '.png')

    plt.figure()
    plt.subplot(1, 2, 1)
    for i, l in enumerate(np.sort(np.unique(labels))):
        idx = np.where(labels == l)
        plt.scatter(zx_post[idx, 0], zx_post[idx, 1], marker=ms[i], color=c[i])
    plt.title("Target Class Shared Space")

    plt.subplot(1, 2, 2)
    plt.scatter(zy_post[:, 0], zy_post[:, 1])
    plt.title("Background Class Shared Space")
    plt.savefig('./experiments/Edward/z' + str(seed) + '.png')

    save_name = 'experiments/Edward/' + fn + 'obj' + str(seed) + '.pkl'
    joblib.dump(model, save_name)

    return z_post, zx_post, zy_post, bx_post, w_post, s_post

if __name__ == '__main__':
    Shared = False
    Target = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0)
    parser.add_argument("--ARDTarget", default=False)
    parser.add_argument("--ARDShared", default=False)
    parser.add_argument("--Ks", default=10)
    parser.add_argument("--Ki", default=2)
    args = parser.parse_args()
    #A, B, labels = build_dataset()
    #A, B, labels = build_ARD_dataset()
    A, B, labels = build_dataset()
    if args.ARDShared:
        Shared = True
    if args.ARDTarget:
        Target = True
    scca = PcPCA(background=B, target=A, Ks=int(args.Ks), Ki=int(args.Ki), seed=int(args.seed))
    # scca = PcPCA(B, A, int(args.Ks), int(args.Ki), int(args.seed))
    scca.learn_model(B, A, labels, seed=int(args.seed))
