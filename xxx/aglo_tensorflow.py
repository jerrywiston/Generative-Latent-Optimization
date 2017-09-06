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

#=============  Adversarial Learn Latent  =============
# Genarator
W_ag1 = tf.Variable(latent_init([v_size, 64]))
b_ag1 = tf.Variable(tf.zeros(shape=[64]))
W_ag2 = tf.Variable(latent_init([64,latent_size]))
b_ag2 = tf.Variable(tf.zeros(shape=[latent_size]))

var_g = [W_ag1, b_ag1, W_ag2, b_ag2]

def ZGenerator(v):
    h_ag1 = tf.nn.relu(tf.matmul(v, W_ag1) + b_ag1)
    G_log_prob = tf.matmul(h_ag1, W_ag2) + b_ag2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob 

# Discriminator
W_ad1 = tf.Variable(xavier_init([latent_size, 64]))
b_ad1 = tf.Variable(tf.zeros(shape=[64]))
W_ad2 = tf.Variable(xavier_init([64, 32]))
b_ad2 = tf.Variable(tf.zeros(shape=[32]))
W_ad3 = tf.Variable(xavier_init([32, 1]))
b_ad3 = tf.Variable(tf.zeros(shape=[1]))

var_d = [W_ad1, b_ad1, W_ad2, b_ad2, W_ad3, b_ad3]

def ZDiscriminator(z):
    h_ad1 = tf.nn.relu(tf.matmul(z, W_ad1) + b_ad1)
    h_ad2 = tf.nn.relu(tf.matmul(h_ad1, W_ad2) + b_ad2)
    D_logit = tf.matmul(h_ad2, W_ad3) + b_ad3
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

G_samp = ZGenerator(v_)
D_real, D_logit_real = ZDiscriminator(z_)
D_fake, D_logit_fake = ZDiscriminator(G_samp)
_, x_samp_adv = Generator(G_samp)

'''
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=var_d)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=var_g)
'''

D_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean(D_fake**2))
G_loss = 0.5 * tf.reduce_mean((D_fake - 1)**2)

D_solver = (tf.train.AdamOptimizer().minimize(D_loss, var_list=var_d))
G_solver = (tf.train.AdamOptimizer().minimize(G_loss, var_list=var_g))
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
    W_z = sess.run(W_latent)
    z_re = LatentRescale(W_z)
    sess.run(z_rescale_op, feed_dict={z_r_: z_re})

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

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={z_: z_batch, v_: sample_v(mb_size, v_size)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={v_: sample_v(mb_size, v_size)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
        samples = sess.run(x_samp_adv, feed_dict={v_: sample_v(16, v_size)})
        fig = plot(samples)
        plt.savefig('out/ad_{}.png'.format(str(i).zfill(4)), bbox_inches='tight')
        i += 1
        plt.close(fig)


