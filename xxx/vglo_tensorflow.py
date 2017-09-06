import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random

#=============  Some Util Function  =============
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def latent_init(size):
    return tf.random_normal(shape=size, stddev=0.1)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def mnist_next_batch(imgs, labels, size):
    img_samp = np.ndarray(shape=(size, imgs.shape[1]))
    label_samp = np.ndarray(shape=(size, labels.shape[1]))
    for i in range(size):
        rand_num = random.randint(0,len(imgs)-1)
        img_samp[i] = imgs[rand_num]
        label_samp[i] = labels[rand_num]
    return [img_samp, label_samp]

def sample_v(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def sample_z(m, n):
    z = []
    for i in range(m):
        scale = random.uniform(0., 1.)
        z_temp = np.random.uniform(-1., 1., size=[n])
        z_temp /= np.sqrt(z_temp.dot(z_temp))
        z_temp *= scale
        z.append(z_temp.tolist())
    return np.asarray(z)

#=============  Read Data & Parameter  =============
# Read file and make training set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_train = mnist.train.images[0:10000]
k_train = np.identity(x_train.shape[0])

# Some parameter
samp_size = x_train.shape[0]
v_size = 16
latent_size = 64
batch_size = 128

# Placeholder
v_ = tf.placeholder(tf.float32, shape=[None, v_size])
k_ = tf.placeholder(tf.float32, shape=[None, samp_size])
z_ = tf.placeholder(tf.float32, shape=[None, latent_size])
x_ = tf.placeholder(tf.float32, shape=[None,784])

#=============  Generative Latent Optimize  =============
# Variable
W_latent = tf.Variable(latent_init([samp_size, latent_size]))

W_g1 = tf.Variable(xavier_init([latent_size,10]))
b_g1 = tf.Variable(tf.zeros(shape=[10]))
W_g2 = tf.Variable(xavier_init([10,128]))
b_g2 = tf.Variable(tf.zeros(shape=[128]))
W_g3 = tf.Variable(xavier_init([128,784]))
b_g3 = tf.Variable(tf.zeros(shape=[784]))

# Implement
def OneHotEncoder(k):
    z_digit = tf.matmul(k, W_latent)
    z = tf.nn.l2_normalize(z_digit, dim=1, epsilon=1, name=None)
    return z

def Generator(z_l):
    h_g1 = tf.nn.relu(tf.matmul(z_l, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    x_re_logit = tf.matmul(h_g2, W_g3) + b_g3
    x_re_prob = tf.nn.relu(x_re_logit)
    return x_re_logit, x_re_prob

def LatentRescale(z):
    for i in range(z.shape[0]):
        length = z[i].dot(z[i])
        if length > 1.0:
            z[i] /= length
    return z

z_latent = OneHotEncoder(k_)
x_logits, x_prob = Generator(z_latent)
_, x_samp = Generator(z_)

z_r_ = tf.placeholder(tf.float32, shape=[samp_size, latent_size])
z_rescale_op = tf.assign(W_latent, z_r_)

#Loss and optimizer
loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_prob - x_), reduction_indices=[1]))
#loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x_in), 1)
train = tf.train.AdamOptimizer().minimize(loss)

#=============  Variational Learn Latent  =============
# Encoder
W_e1 = tf.Variable(xavier_init([latent_size, 32]))
b_e1 = tf.Variable(tf.zeros(shape=[32]))
W_e2_mu = tf.Variable(xavier_init([32, v_size]))
b_e2_mu = tf.Variable(tf.zeros(shape=[v_size]))
W_e2_sig = tf.Variable(xavier_init([32, v_size]))
b_e2_sig = tf.Variable(tf.zeros(shape=[v_size]))

def ZEncoder(z):
    h_e1 = tf.nn.relu(tf.matmul(z, W_e1) + b_e1)
    v_mu = tf.matmul(h_e1, W_e2_mu) + b_e2_mu
    v_logvar = tf.matmul(h_e1, W_e2_sig) + b_e2_sig

    return v_mu, v_logvar

def VRepara(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

# Decoder
W_d1 = tf.Variable(xavier_init([v_size, 32]))
b_d1 = tf.Variable(tf.zeros(shape=[32]))
W_d2 = tf.Variable(xavier_init([32,latent_size]))
b_d2 = tf.Variable(tf.zeros(shape=[latent_size]))

def ZDecoder(v):
    h_d1 = tf.nn.relu(tf.matmul(v, W_d1) + b_d1)
    logits = tf.matmul(h_d1, W_d2) + b_d2
    prob = tf.nn.sigmoid(logits)
    return prob, logits

v_mu, v_logvar = ZEncoder(z_)
v_sample = VRepara(v_mu, v_logvar)
_, logits = ZDecoder(v_sample)
z_samp_v, _ = ZDecoder(v_)
_, x_samp_v = Generator(z_samp_v)

recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=z_), 1)
kl_loss = 0.5 * tf.reduce_sum(tf.exp(v_logvar) + v_mu**2 - 1. - v_logvar, 1)
vae_loss = tf.reduce_mean(recon_loss + kl_loss)
v_solver = tf.train.AdamOptimizer().minimize(vae_loss)

# Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#=============  GLO Training Phase  =============
if not os.path.exists('out/'):
    os.makedirs('out/')

#Main start point
print("Training GLO")
i=0
for it in range(2000):
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        #samples = sess.run(x_samp, feed_dict={z_: sample_z(16, latent_size)})
        #k_test = sample_Z(16,10000)
        k_test = k_train[0:16]
        samples = sess.run(x_prob, feed_dict={k_: k_test})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    x_t, k_t = mnist_next_batch(x_train, k_train, batch_size)
    _loss = sess.run(train, feed_dict={x_: x_t, k_: k_t})
    #W_z = sess.run(W_latent)
    #z_re = LatentRescale(W_z)
    #sess.run(z_rescale_op, feed_dict={z_r_: z_re})

#=============  Adversarial Training Phase  =============
def LatentNextBatch(z, b_size):
    z_batch = []
    for i in range(b_size):
        r = random.randint(0,b_size-1)
        z_batch.append(z[r])
    return z_batch
i=0
print("Training Latent")
z_latent_train = sess.run(W_latent)
mb_size = 128
for it in range(50000):
    z_batch = LatentNextBatch(z_latent_train, mb_size)
    _, v_loss = sess.run([v_solver, vae_loss], feed_dict={z_: z_batch})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        samples = sess.run(x_samp_v, feed_dict={v_: np.random.randn(16, v_size)})
        fig = plot(samples)
        plt.savefig('out/ad_{}.png'.format(str(i).zfill(4)), bbox_inches='tight')
        i += 1
        plt.close(fig)


