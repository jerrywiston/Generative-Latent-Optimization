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
    #return tf.random_uniform(shape=size, minval=-1.0, maxval=1.0)
    return tf.random_normal(shape=size, stddev=1)

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

def plot_x(id, type, samp):
    fig = plot(samp)
    plt.savefig('out/{}_{}.png'.format(str(id).zfill(4), type), bbox_inches='tight')
    plt.close(fig)

def plot_z(z, dim1, dim2, id, show_img=False, img=None):
    x = z[0:1000, dim1]
    y = z[0:1000, dim2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
    ax.plot(x,y,'b.')
    an = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(an), np.sin(an), 'r')
    plt.savefig('out/{}_dist.png'.format(str(id).zfill(4)), bbox_inches='tight')
    plt.close(fig)


#Read file and make training set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_train = mnist.train.images[0:10000]
k_train = np.identity(x_train.shape[0])

#Some parameter
samp_size = x_train.shape[0]
latent_size = 8
batch_size = 256

#Placeholder
k_ = tf.placeholder(tf.float32, shape=[None, samp_size])
z_ = tf.placeholder(tf.float32, shape=[None, latent_size])
x_ = tf.placeholder(tf.float32, shape=[None,784])

#Model Variable
W_latent = tf.Variable(latent_init([samp_size, latent_size]))

W_g1 = tf.Variable(xavier_init([latent_size,10]))
b_g1 = tf.Variable(tf.zeros(shape=[10]))
W_g2 = tf.Variable(xavier_init([10,128]))
b_g2 = tf.Variable(tf.zeros(shape=[128]))
W_g3 = tf.Variable(xavier_init([128,784]))
b_g3 = tf.Variable(tf.zeros(shape=[784]))

var_g = [W_g1, b_g1, W_g2, b_g2, W_g3, b_g3]

#Model Implement
def OneHotEncoder(k):
    z_digit = tf.matmul(k, W_latent)
    #z = tf.nn.l2_normalize(z_digit, dim=1, epsilon=1, name=None)
    return z_digit

def Generator(z_l):
    h_g1 = tf.nn.relu(tf.matmul(z_l, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    x_re_logit = tf.matmul(h_g2, W_g3) + b_g3
    x_re_prob = tf.nn.sigmoid(x_re_logit)
    return x_re_logit, x_re_prob

def LatentRescale(z):
    for i in range(z.shape[0]):
        length = np.sqrt(z[i].dot(z[i]))
        #z[i] /= length
        if length > 1.0:
            z[i] /= length

    return z

z_latent = OneHotEncoder(k_)
x_logits, x_prob = Generator(z_latent)
_, x_samp = Generator(z_)

z_re_ = tf.placeholder(tf.float32, shape=[samp_size, latent_size])
z_rescale_op = tf.assign(W_latent, z_re_)

#Loss and optimizer
loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_prob - x_), reduction_indices=[1]))
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x_))#, reduction_indices=[1])

train = tf.train.AdamOptimizer().minimize(loss)
train_generator = tf.train.AdamOptimizer().minimize(loss, var_list=var_g)
train_latent = tf.train.AdamOptimizer(0.01).minimize(loss, var_list=[W_latent])
#train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

#Main start point
i=0
for it in range(100000):
    #Optimize
    x_batch, k_batch = mnist_next_batch(x_train, k_train, batch_size)
    #sess.run(train, feed_dict={x_: x_batch, k_: k_batch})
    sess.run(train_generator, feed_dict={x_: x_batch, k_: k_batch})
    sess.run(train_latent, feed_dict={x_: x_batch, k_: k_batch})   

    #Rescale
    W_z = sess.run(W_latent)
    z = LatentRescale(W_z)
    sess.run(z_rescale_op, feed_dict={z_re_: z})

    #Print message
    if it % 100 == 0:
        loss_ = sess.run(loss, feed_dict={x_: x_batch, k_: k_batch})
        print('Iter: {}, loss: {}'.format(it, loss_))
        samples_samp = sess.run(x_samp, feed_dict={z_: sample_z(16, latent_size)})
        _, k_test = mnist_next_batch(x_train, k_train, 16)
        samples_re = sess.run(x_prob, feed_dict={k_: k_test})

        plot_x(i,'recon',samples_re)
        plot_x(i,'samp',samples_samp)
        plot_z(W_z, 0, 1, i)
        i += 1

        
        


