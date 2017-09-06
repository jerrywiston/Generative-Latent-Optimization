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

def latent_init(size):
    return tf.random_normal(shape=size, stddev=1.0)

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

def sample_z(m, n):
    z = []
    for i in range(m):
        scale = random.uniform(0., 1.)
        z_temp = np.random.uniform(-1., 1., size=[n])
        z_temp /= np.sqrt(z_temp.dot(z_temp))
        z_temp *= scale
        z.append(z_temp.tolist())
    return np.asarray(z)

#Read file and make training set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_train = mnist.train.images[0:10000]
z_train = np.identity(x_train.shape[0])

#Some parameter
samp_size = x_train.shape[0]
latent_size = 25
batch_size = 128

#Placeholder
z_one_hot = tf.placeholder(tf.float32, shape=[None, samp_size])
z_samp = tf.placeholder(tf.float32, shape=[None, latent_size])
x_in = tf.placeholder(tf.float32, shape=[None,784])

#Model Variable
W_latent = tf.Variable(latent_init([samp_size, latent_size]))

W_g1 = tf.Variable(xavier_init([latent_size,10]))
b_g1 = tf.Variable(tf.zeros(shape=[10]))
W_g2 = tf.Variable(xavier_init([10,128]))
b_g2 = tf.Variable(tf.zeros(shape=[128]))
W_g3 = tf.Variable(xavier_init([128,784]))
b_g3 = tf.Variable(tf.zeros(shape=[784]))

#Model Implement
def OneHotEncoder(k):
    z = tf.matmul(k, W_latent)
    return z

def Generator(z_l):
    h_g1 = tf.nn.relu(tf.matmul(z_l, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    x_re_logit = tf.matmul(h_g2, W_g3) + b_g3
    x_re_prob = tf.nn.relu(x_re_logit)
    return x_re_logit, x_re_prob

def LatentRescale(z):
    #z = sess.run(W_latent)
    #print(z[0])
    for i in range(z.shape[0]):
        length = z[i].dot(z[i])
        if length > 1.0:
            z[i] /= length

    return z

z_latent = OneHotEncoder(z_one_hot)
x_logits, x_prob = Generator(z_latent)
_, x_samp = Generator(z_samp)

z_re = tf.placeholder(tf.float32, shape=[samp_size, latent_size])
z_rescale = tf.assign(W_latent, z_re)

#Loss and optimizer
loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_prob - x_in), reduction_indices=[1]))
#loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x_in), 1)
train = tf.train.AdamOptimizer().minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

#Main start point
i=0
for it in range(100000):
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        samples = sess.run(x_samp, feed_dict={z_samp: sample_z(16, latent_size)})
        #Z_test = sample_Z(16,10000)
        #Z_test = Z_train[0:16]
        #samples = sess.run(GLO_re, feed_dict={Z_one_hot: Z_test})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    x_, z_ = mnist_next_batch(x_train, z_train, 50)
    G_loss = sess.run(train, feed_dict={x_in: x_, z_one_hot: z_})
    W_z = sess.run(W_latent)
    z = LatentRescale(W_z)
    sess.run(z_rescale, feed_dict={z_re: z})


