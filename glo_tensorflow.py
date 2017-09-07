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
    x = z[0:1000, dim1]
    y = z[0:1000, dim2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
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
def latent_rescale(z):
    length = np.sqrt(z.dot(z))
    if length > 1.0:
        z /= length
    return z

def train_latent(grad, id_list, z_train, rate):
    for i in range(id_list.shape[0]):
        z_update = z_train[id_list[i]] - rate * grad[i]
        z_train[id_list[i]] = latent_rescale(z_update)
    
#================================= Data & Parameter =================================
#Some parameter
latent_size = 10
batch_size = 256

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_train = mnist.train.images
z_train = np.random.normal(0., 0.5, [x_train.shape[0], latent_size])

#================================= Network model =================================
#Placeholder
z_ = tf.placeholder(tf.float32, shape=[None, latent_size])
x_ = tf.placeholder(tf.float32, shape=[None, 784])

#Model Variable
W_g1 = tf.Variable(xavier_init([latent_size,16]))
b_g1 = tf.Variable(tf.zeros(shape=[16]))
W_g2 = tf.Variable(xavier_init([16,128]))
b_g2 = tf.Variable(tf.zeros(shape=[128]))
W_g3 = tf.Variable(xavier_init([128,784]))
b_g3 = tf.Variable(tf.zeros(shape=[784]))

#Model Implement
def Generator(z):
    h_g1 = tf.nn.relu(tf.matmul(z, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    x_re_logit = tf.matmul(h_g2, W_g3) + b_g3
    x_re_prob = tf.nn.sigmoid(x_re_logit)
    return x_re_logit, x_re_prob

x_logits, x_prob = Generator(z_)

#Loss and optimizer
loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_prob - x_), reduction_indices=[1]))
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x_))

z_gradients = tf.gradients(loss, z_)
solver = tf.train.AdamOptimizer().minimize(loss)
#solver = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#================================= Main =================================
if not os.path.exists('out/'):
    os.makedirs('out/')

i=0
for it in range(50001):
    #Train weight & latent
    x_batch, z_batch, id_batch = mnist_next_batch(x_train, z_train, batch_size)
    _, grad = sess.run([solver, z_gradients], feed_dict={x_: x_batch, z_: z_batch})
    grad_np = np.asarray(grad[0])
    train_latent(grad_np, id_batch, z_train, 1.)

    #Print message
    if it % 100 == 0:
        loss_ = sess.run(loss, feed_dict={x_: x_batch, z_: z_batch})
        print('Iter: {}, loss: {}'.format(it, loss_))
        
        #Reconstruct x test
        _, z_test, _ = mnist_next_batch(x_train, z_train, 16)
        samples_re = sess.run(x_prob, feed_dict={z_: z_test})

        #Random sample z test
        z_samp = np.random.normal(0., 0.3, [16, latent_size])
        samples_samp = sess.run(x_prob, feed_dict={z_: z_samp})
        
        #Ouput figure
        plot_x(i,'recon', samples_re)
        plot_x(i,'samp', samples_samp)
        plot_z(z_train, 0, 1, i, z_samp)
        i += 1
