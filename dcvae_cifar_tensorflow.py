import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import random
import os

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
        plt.imshow(sample.reshape(32, 32, 3))

    return fig

def plot_x(id, type, samp):
    fig = plot(samp)
    plt.savefig('out/{}_{}.png'.format(str(id).zfill(4), type), bbox_inches='tight')
    plt.close(fig)

def plot_z(z, dim1, dim2, id, samp):
	ax_lim = 2.
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
    if len(size) == 4:
        n_inputs = size[0]*size[1]*size[2]
        n_outputs = size[3]
    else:
        n_inputs = size[0]
        n_outputs = size[1]
    
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    #stddev = 0.002
    return tf.truncated_normal(size, stddev=stddev)

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

#================================= Cifar Handling =================================
def cifar_read(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pinckle = pickle.load(fo, encoding='bytes')
    return dict

def cifar_data_extract(dict):
    imgs_raw = dict[b'data'].astype("uint8")
    imgs = np.array(imgs_raw, dtype=float) / 255.0   
    imgs = imgs.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    print(imgs.shape)
    return imgs

def cifar_next_batch(imgs, size):
    img_samp = np.ndarray(shape=(size, 32, 32 ,3))
    for i in range(size):
        r = random.randint(0,len(imgs)-1)
        img_samp[i] = imgs[r]
    return img_samp

#================================= Data & Parameter =================================
#Some parameter
latent_size = 512
batch_size = 256

dict = cifar_read("cifar-10-batches-py/data_batch_1")
x_train = cifar_data_extract(dict)

#================================= Network model =================================
#Placeholder
x_ = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
z_ = tf.placeholder(tf.float32, shape=[None, latent_size])

#Model Variable
W_e_conv1 = tf.Variable(xavier_init([4,4,3,16]))
b_e_conv1 = tf.Variable(tf.zeros(shape=[16]))

W_e_conv2 = tf.Variable(xavier_init([4,4,16,64]))
b_e_conv2 = tf.Variable(tf.zeros(shape=[64]))

W_e_conv3 = tf.Variable(xavier_init([4,4,64,128]))
b_e_conv3 = tf.Variable(tf.zeros(shape=[128]))

W_e_conv4 = tf.Variable(xavier_init([4,4,128,384]))
b_e_conv4 = tf.Variable(tf.zeros(shape=[384]))

W_e_conv5 = tf.Variable(xavier_init([4,4,384,400]))
b_e_conv5 = tf.Variable(tf.zeros(shape=[400]))

W_mu = tf.Variable(xavier_init([400, latent_size]))
b_mu = tf.Variable(tf.zeros(shape=[latent_size]))
W_sigma = tf.Variable(xavier_init([400, latent_size]))
b_sigma = tf.Variable(tf.zeros(shape=[latent_size]))

W_d_fc = tf.Variable(xavier_init([latent_size, 400]))
b_d_fc = tf.Variable(tf.zeros(shape=[400]))

W_d_dconv1 = tf.Variable(xavier_init([4,4,384,400]))
b_d_dconv1 = tf.Variable(tf.zeros(shape=[384]))

W_d_dconv2 = tf.Variable(xavier_init([4,4,128,384]))
b_d_dconv2 = tf.Variable(tf.zeros(shape=[128]))

W_d_dconv3 = tf.Variable(xavier_init([4,4,64,128]))
b_d_dconv3 = tf.Variable(tf.zeros(shape=[64]))

W_d_dconv4 = tf.Variable(xavier_init([4,4,16,64]))
b_d_dconv4 = tf.Variable(tf.zeros(shape=[16]))

W_d_dconv5 = tf.Variable(xavier_init([4,4,3,16]))
b_d_dconv5 = tf.Variable(tf.zeros(shape=[3]))

#Model Implement
def conv2d(x, W, stride=[1,2,2,1]):
    return tf.nn.conv2d(x ,W ,strides=stride, padding='SAME')

def deconv2d(x, W, output_shape, stride=[1, 2, 2, 1]):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=stride, padding='SAME')

def Reparameter(k):
    z_mu = tf.matmul(k, W_mu) + b_mu
    z_logvar = tf.matmul(k, W_sigma) + b_sigma
    return z_mu, z_logvar

def SampleZ(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

def Encoder(x):
    h_e_conv1 = tf.nn.relu(conv2d(x, W_e_conv1) + b_e_conv1)
    h_e_conv2 = tf.nn.relu(conv2d(h_e_conv1, W_e_conv2) + b_e_conv2)
    h_e_conv3 = tf.nn.relu(conv2d(h_e_conv2, W_e_conv3) + b_e_conv3)
    h_e_conv4 = tf.nn.relu(conv2d(h_e_conv3, W_e_conv4) + b_e_conv4)
    h_e_conv5 = tf.nn.relu(conv2d(h_e_conv4, W_e_conv5) + b_e_conv5)
    h_e_re5 = tf.reshape(h_e_conv5, [-1, 400])
    
    z_mu = tf.matmul(h_e_re5, W_mu) + b_mu
    z_logvar = tf.matmul(h_e_re5, W_sigma) + b_sigma
    return z_mu, z_logvar

def Decoder(z):
    h_d_fc = tf.nn.relu(tf.matmul(z, W_d_fc) + b_d_fc)
    h_d_re = tf.reshape(h_d_fc, [-1, 1, 1, tf.shape(h_d_fc)[1]])

    output_shape_d1 = tf.stack([tf.shape(z)[0], 2, 2, 384])
    h_d_conv1 = tf.nn.relu(deconv2d(h_d_re, W_d_dconv1, output_shape_d1) + b_d_dconv1)
    
    output_shape_d2 = tf.stack([tf.shape(z)[0], 4, 4, 128])
    h_d_conv2 = tf.nn.relu(deconv2d(h_d_conv1, W_d_dconv2, output_shape_d2) + b_d_dconv2)

    output_shape_d3 = tf.stack([tf.shape(z)[0], 8, 8, 64])
    h_d_conv3 = tf.nn.relu(deconv2d(h_d_conv2, W_d_dconv3, output_shape_d3) + b_d_dconv3)

    output_shape_d4 = tf.stack([tf.shape(z)[0], 16, 16, 16])
    h_d_conv4 = tf.nn.relu(deconv2d(h_d_conv3, W_d_dconv4, output_shape_d4) + b_d_dconv4)

    output_shape_d5 = tf.stack([tf.shape(z)[0], 32, 32, 3])
    h_d_conv5 = tf.nn.sigmoid(deconv2d(h_d_conv4, W_d_dconv5, output_shape_d5) + b_d_dconv5)

    return h_d_conv5

#z_mu, z_logvar = Encoder(x_)
#z_sample = SampleZ(z_mu, z_logvar)
z_sample = Encoder(x_)
x_prob = Decoder(z_sample)
x_sample = Decoder(z_)

#Loss and optimizer
recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_prob - x_), reduction_indices=[1]))
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
loss = tf.reduce_mean(recon_loss + 0.01*kl_loss)

solver = tf.train.AdamOptimizer().minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#================================= Main =================================
if not os.path.exists('out/'):
    os.makedirs('out/')

i=0
for it in range(100001):
    #Train weight & latent
    x_batch = cifar_next_batch(x_train, batch_size)
    sess.run(solver, feed_dict={x_: x_batch})

    #Print message
    if it % 100 == 0:
        loss_ = sess.run(loss, feed_dict={x_: x_batch})
        print('Iter: {}, loss: {}'.format(it, loss_))
        
        #Reconstruct x test
        x_test = cifar_next_batch(x_train, 16)
        samples_re = sess.run(x_prob, feed_dict={x_: x_test})
        plot_x(i,'recon', samples_re)

        #Random sample z test
        z_samp = np.random.normal(0., 1., [16, latent_size])
        samples_samp = sess.run(x_sample, feed_dict={z_: z_samp})
        plot_x(i,'samp', samples_samp)
        
        #Ouput figure
        z_plot = sess.run(z_sample, feed_dict={x_: x_train[0:100]})
        plot_z(z_plot, 0, 1, i, z_samp)
        i += 1
