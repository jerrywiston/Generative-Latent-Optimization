import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random

#Some util function
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

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

#def sample_Z(m, n):
#    return np.random.uniform(-1., 1., size=[m, n])

def sample_Z(m, n):
    z = []
    for i in range(m):
        z_temp = np.random.uniform(-1., 1., size=[n])
        z_temp /= np.sqrt(z_temp.dot(z_temp))
        z.append(z_temp.tolist())
    return np.asarray(z)

#Read file and make training set
mnist = input_data.read_data_sets('MNIST_fashion', one_hot=True)
X_train = mnist.train.images[0:10000]
Z_train = np.identity(X_train.shape[0])

#Some parameter
samp_size = X_train.shape[0]
latent_size = 100
batch_size = 128

#Placeholder
k_ = tf.placeholder(tf.float32, shape=[None, samp_size])
z_ = tf.placeholder(tf.float32, shape=[None, latent_size])
x_ = tf.placeholder(tf.float32, shape=[None,784])

#Model Variable
W_mu = tf.Variable(xavier_init([samp_size, latent_size]))
b_mu = tf.Variable(tf.zeros(shape=[latent_size]))
W_sigma = tf.Variable(xavier_init([samp_size, latent_size]))
b_sigma = tf.Variable(tf.zeros(shape=[latent_size]))

W_g1 = tf.Variable(xavier_init([latent_size,10]))
b_g1 = tf.Variable(tf.zeros(shape=[10]))
W_g2 = tf.Variable(xavier_init([10,128]))
b_g2 = tf.Variable(tf.zeros(shape=[128]))
W_g3 = tf.Variable(xavier_init([128,784]))
b_g3 = tf.Variable(tf.zeros(shape=[784]))

#Model Implement
def OneHotEncoder(k):
    z_mu = tf.matmul(k, W_mu) + b_mu
    z_logvar = tf.matmul(k, W_sigma) + b_sigma
    return z_mu, z_logvar

def Generator(z):
    h_g1 = tf.nn.relu(tf.matmul(z, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    x_logits = tf.matmul(h_g2, W_g3) + b_g3
    x_prob = tf.nn.sigmoid(x_logits)
    return x_prob, x_logits

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

z_mu, z_logvar = OneHotEncoder(k_)
z_sample = sample_z(z_mu, z_logvar)
_, x_logits = Generator(z_sample)
x_sample, _ = Generator(z_)

#Loss and optimizer
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x_), 1)
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

#Main start point
i=0
for it in range(100000):
    if it % 1000 == 0:
        samples = sess.run(x_sample, feed_dict={z_: np.random.randn(16, latent_size)})
        #print(samples.shape)
        #Z_test = sample_Z(16,10000)
        #Z_test = Z_train[0:16]
        #samples = sess.run(GLO_re, feed_dict={Z_one_hot: Z_test})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_, K_ = mnist_next_batch(X_train, Z_train, 50)
    G_loss = sess.run(solver, feed_dict={x_: X_, k_: K_})
    if it % 1000 == 0:
        print('Iter: {}'.format(it))

