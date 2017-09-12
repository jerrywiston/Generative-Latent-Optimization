import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random

#================================= Unit Function =================================
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

def plot_x(id, type, samp):
    fig = plot(samp)
    plt.savefig('out/{}_{}.png'.format(str(id).zfill(4), type), bbox_inches='tight')
    plt.close(fig)

def plot_z(z, dim1, dim2, id, samp):
	ax_lim = 2.0#1.2
	x = z[0:1000, dim1]
	y = z[0:1000, dim2]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_aspect('equal')
	ax.set_xlim([-ax_lim, ax_lim])
	ax.set_ylim([-ax_lim, ax_lim])
	ax.plot(x,y,'b.')
	ax.plot(samp[:,dim1], samp[:,dim2], 'ro')

	#an = np.linspace(0, 2*np.pi, 100)
	#ax.plot(np.cos(an), np.sin(an), 'r')

	plt.savefig('out/{}_dist.png'.format(str(id).zfill(4)), bbox_inches='tight')
	plt.close(fig)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def mnist_next_batch(imgs, labels, size):
    id_samp = np.ndarray(shape=(size), dtype=np.int32)
    img_samp = np.ndarray(shape=(size, imgs.shape[1]))
    label_samp = np.ndarray(shape=(size, labels.shape[1]))
    for i in range(size):
        r = random.randint(0,imgs.shape[0]-1)
        img_samp[i] = imgs[r]
        label_samp[i] = labels[r]
        id_samp[i] = r
    return [img_samp, label_samp, id_samp]

#================================= Latent Training Function =================================
def train_latent(grad, id_list, z_train, rate):
    for i in range(id_list.shape[0]):
        z_train[id_list[i]] -= rate * grad[i]

#================================= Data & Parameter =================================
#Some parameter
latent_size = 256
repar_size = 100
batch_size = 256

#mnist = input_data.read_data_sets('MNIST_fashion', one_hot=True)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_train = mnist.train.images
#std = 1. / tf.sqrt(x_train.shape[0] / 2.)
k_train = np.random.normal(0., 0.001, [x_train.shape[0], latent_size])

#================================= Network model =================================
#Placeholder
k_ = tf.placeholder(tf.float32, shape=[None, latent_size])
z_ = tf.placeholder(tf.float32, shape=[None, repar_size])
x_ = tf.placeholder(tf.float32, shape=[None, 784])

#Model Variable
W_mu = tf.Variable(xavier_init([latent_size, repar_size]))
b_mu = tf.Variable(tf.zeros(shape=[repar_size]))
W_sigma = tf.Variable(xavier_init([latent_size, repar_size]))
b_sigma = tf.Variable(tf.zeros(shape=[repar_size]))

W_g1 = tf.Variable(xavier_init([repar_size,32]))
b_g1 = tf.Variable(tf.zeros(shape=[32]))
W_g2 = tf.Variable(xavier_init([32,128]))
b_g2 = tf.Variable(tf.zeros(shape=[128]))
W_g3 = tf.Variable(xavier_init([128,784]))
b_g3 = tf.Variable(tf.zeros(shape=[784]))

#Model Implement
def Reparameter(k):
	z_mu = tf.matmul(k, W_mu) + b_mu
	z_logvar = tf.matmul(k, W_sigma) + b_sigma
	return z_mu, z_logvar

def SampleZ(mu, log_var):
	eps = tf.random_normal(shape=tf.shape(mu))
	return mu + tf.exp(log_var / 2) * eps

def Generator(z):
    h_g1 = tf.nn.relu(tf.matmul(z, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    x_re_logit = tf.matmul(h_g2, W_g3) + b_g3
    x_re_prob = tf.nn.sigmoid(x_re_logit)
    return x_re_logit, x_re_prob

z_mu, z_logvar = Reparameter(k_)
z_sample = SampleZ(z_mu, z_logvar)
x_logits, x_prob = Generator(z_sample)
_, x_sample = Generator(z_)

#Loss and optimizer
#recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_prob, labels=x_), 1)
recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_prob - x_), reduction_indices=[1]))
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
loss = tf.reduce_mean(recon_loss + kl_loss)

z_gradients = tf.gradients(loss, k_)
solver = tf.train.AdamOptimizer().minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#================================= Main =================================
if not os.path.exists('out/'):
    os.makedirs('out/')

i=0
for it in range(20001):
    #Train weight & latent
    x_batch, k_batch, id_batch = mnist_next_batch(x_train, k_train, batch_size)
    _, grad = sess.run([solver, z_gradients], feed_dict={x_: x_batch, k_: k_batch})
    grad_np = np.asarray(grad[0])
    train_latent(grad_np, id_batch, k_train, 1.)

    #Print message
    if it % 100 == 0:
        loss_ = sess.run(loss, feed_dict={x_: x_batch, k_: k_batch})
        print('Iter: {}, loss: {}'.format(it, loss_))
        
        #Reconstruct x test
        _, k_test, _ = mnist_next_batch(x_train, k_train, 16)
        samples_re = sess.run(x_prob, feed_dict={k_: k_test})
        plot_x(i,'recon', samples_re)

        #Random sample z test
        z_samp = np.random.normal(0., 1., [16, repar_size])
        samples_samp = sess.run(x_sample, feed_dict={z_: z_samp})
        plot_x(i,'samp', samples_samp)
        
        #Ouput figure
        z_plot = sess.run(z_sample, feed_dict={k_: k_train[0:1000]})
        plot_z(z_plot, 0, 1, i, z_samp)
        i += 1

#mu_look, sigma_look = sess.run([z_mu, z_logvar], feed_dict={k_: k_train[0:20]})
#print(mu_look)
#print(sigma_look)
#print(k_train)

