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

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

#Read file and make training set
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
X_train = mnist.train.images[0:10000]
Z_train = np.identity(X_train.shape[0])

#Some parameter
samp_size = X_train.shape[0]
latent_size = 100
batch_size = 128

#Placeholder
Z_one_hot = tf.placeholder(tf.float32, shape=[None, samp_size])
Z_samp = tf.placeholder(tf.float32, shape=[None, latent_size])
X_in = tf.placeholder(tf.float32, shape=[None,784])

#Model Variable
W_latent = tf.Variable(xavier_init([samp_size, latent_size]))
b_latent = tf.Variable(tf.zeros(shape=[latent_size]))

W_g1 = tf.Variable(xavier_init([latent_size,10]))
b_g1 = tf.Variable(tf.zeros(shape=[10]))
W_g2 = tf.Variable(xavier_init([10,128]))
b_g2 = tf.Variable(tf.zeros(shape=[128]))
W_g3 = tf.Variable(xavier_init([128,784]))
b_g3 = tf.Variable(tf.zeros(shape=[784]))

#Model Implement
def OneHotEncoder(Z_o):
    #Z_latent = tf.nn.tanh(tf.matmul(Z_o,W_latent) + b_latent, name='latent')
    Z_latent = tf.matmul(Z_o,W_latent)
    Z_latent_normal = tf.nn.l2_normalize(Z_latent, dim=1, epsilon=1, name=None)
    return Z_latent_normal

def Generator(Z_l):
    h_g1 = tf.nn.relu(tf.matmul(Z_l, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    X_re = tf.nn.relu(tf.matmul(h_g2, W_g3) + b_g3)
    return X_re

Z_latent = OneHotEncoder(Z_one_hot)
GLO_re = Generator(Z_latent)
GLO_samp = Generator(Z_samp)

# ???


#Loss and optimizer
loss = tf.reduce_mean(tf.reduce_sum(tf.square(GLO_re - X_in), reduction_indices=[1]))
train = tf.train.AdamOptimizer().minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Main start point
i=0
for it in range(100000):
    if it % 1000 == 0:
        samples = sess.run(GLO_samp, feed_dict={Z_samp: sample_Z(16, latent_size)})
        #Z_test = sample_Z(16,10000)
        #Z_test = Z_train[0:16]
        #samples = sess.run(GLO_re, feed_dict={Z_one_hot: Z_test})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_, Z_ = mnist_next_batch(X_train, Z_train, 50)
    G_loss = sess.run(train, feed_dict={X_in: X_, Z_one_hot: Z_})
    if it % 1000 == 0:
        print('Iter: {}'.format(it))

