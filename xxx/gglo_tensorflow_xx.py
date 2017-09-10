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

	an = np.linspace(0, 2*np.pi, 100)
	ax.plot(np.cos(an), np.sin(an), 'r')

	plt.savefig('out/{}_dist.png'.format(str(id).zfill(4)), bbox_inches='tight')
	plt.close(fig)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def train_next_batch(imgs, labels1, labels2, size):
    id_samp = np.ndarray(shape=(size), dtype=np.int32)
    img_samp = np.ndarray(shape=(size, imgs.shape[1]))
    label1_samp = np.ndarray(shape=(size, labels1.shape[1]))
    label2_samp = np.ndarray(shape=(size, labels2.shape[1]))
    for i in range(size):
        r = random.randint(0,imgs.shape[0]-1)
        img_samp[i] = imgs[r]
        label1_samp[i] = labels1[r]
        label2_samp[i] = labels2[r]
        id_samp[i] = r
    return [img_samp, label1_samp, label2_samp, id_samp]

#================================= Latent Training Function =================================
def latent_rescale(z):
    length = np.sqrt(z.dot(z))
    if length > 1.0:
        z /= length
    return z

def train_latent(mu_grad, sigma_grad, id_list, z_mu_train, z_sigma_train, rate):
    for i in range(id_list.shape[0]):
        z_mu_train[id_list[i]] -= rate * mu_grad[i]
        z_sigma_train[id_list[i]] -= rate * sigma_grad[i]

#================================= Data & Parameter =================================
#Some parameter
latent_size = 100
batch_size = 256

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_train = mnist.train.images
z_mu_train = np.random.normal(0., 0.001, [x_train.shape[0], latent_size])
z_sigma_train = np.random.normal(0., 0.001, [x_train.shape[0], latent_size])

#================================= Network model =================================
#Placeholder
z_mu_ = tf.placeholder(tf.float32, shape=[None, latent_size])
z_sigma_ = tf.placeholder(tf.float32, shape=[None, latent_size])
z_ = tf.placeholder(tf.float32, shape=[None, latent_size])
x_ = tf.placeholder(tf.float32, shape=[None, 784])

#Model Variable
W_g1 = tf.Variable(xavier_init([latent_size,32]))
b_g1 = tf.Variable(tf.zeros(shape=[32]))
W_g2 = tf.Variable(xavier_init([32,128]))
b_g2 = tf.Variable(tf.zeros(shape=[128]))
W_g3 = tf.Variable(xavier_init([128,784]))
b_g3 = tf.Variable(tf.zeros(shape=[784]))

#Model Implement
def SampleZ(mu, log_var):
	eps = tf.random_normal(shape=tf.shape(mu))
	return mu + tf.exp(log_var / 2) * eps

def Generator(z):
    h_g1 = tf.nn.relu(tf.matmul(z, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    x_re_logit = tf.matmul(h_g2, W_g3) + b_g3
    x_re_prob = tf.nn.sigmoid(x_re_logit)
    return x_re_logit, x_re_prob

z_sample = SampleZ(z_mu_, z_sigma_)
x_logits, x_prob = Generator(z_sample)
_, x_sample = Generator(z_)

#Loss and optimizer
#recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_prob, labels=x_), 1)
recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_prob - x_), reduction_indices=[1]))
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_sigma_) + z_mu_**2 - 1. - z_sigma_, 1)
loss = tf.reduce_mean(recon_loss + kl_loss)

z_mu_grad = tf.gradients(loss, z_mu_)
z_sigma_grad = tf.gradients(loss, z_sigma_)
solver = tf.train.AdamOptimizer(0.0001).minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#================================= Main =================================
if not os.path.exists('out/'):
    os.makedirs('out/')

i=0
for it in range(50001):
    #Train weight & latent
    x_batch, z_mu_batch, z_sigma_batch, id_batch = train_next_batch(x_train, z_mu_train, z_sigma_train, batch_size)
    _, mu_grad, sigma_grad = sess.run([solver, z_mu_grad, z_sigma_grad], 
                            feed_dict = {x_: x_batch, z_mu_: z_mu_batch, z_sigma_: z_sigma_batch})
    mu_grad_np = np.asarray(mu_grad[0])
    sigma_grad_np = np.asarray(sigma_grad[0])
    train_latent(mu_grad_np, sigma_grad_np, id_batch, z_mu_train, z_sigma_train, 1.)

    #Print message
    if it % 100 == 0:
        loss_ = sess.run(loss, feed_dict={x_: x_batch, z_mu_: z_mu_batch, z_sigma_: z_sigma_batch})
        print('Iter: {}, loss: {}'.format(it, loss_))
        
        #Reconstruct x test
        _, mu_test, sigma_test, _ = train_next_batch(x_train, z_mu_train, z_sigma_train, 16)
        samples_re = sess.run(x_prob, feed_dict={z_mu_: mu_test, z_sigma_: sigma_test})
        plot_x(i,'recon', samples_re)

        #Random sample z test
        z_samp = np.random.normal(0., 1., [16, latent_size])
        samples_samp = sess.run(x_sample, feed_dict={z_: z_samp})
        plot_x(i,'samp', samples_samp)
        
        #Ouput figure
        zplot_size = 1000
        z_plot = sess.run(z_sample, feed_dict={z_mu_: z_mu_train[0:zplot_size], z_sigma_: z_sigma_train[0:zplot_size]})
        plot_z(z_plot, 0, 1, i, z_samp)
        i += 1
