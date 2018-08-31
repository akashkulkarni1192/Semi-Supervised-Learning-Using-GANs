import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec


def initialize_weight(dimensions):
    std_dev = 1. / tf.sqrt(dimensions[0] / 2.)
    return tf.random_normal(shape=dimensions, stddev=std_dev)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for iter, single_sample in enumerate(samples):
        ax = plt.subplot(gs[iter])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(single_sample.reshape(28, 28), cmap='Greys_r')
    return fig


X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
Dis_weight1 = tf.Variable(initialize_weight([784, 128]))
Dis_bias1 = tf.Variable(tf.zeros(shape=[128]))
Dis_weight2 = tf.Variable(initialize_weight([128, 1]))
Dis_bias2 = tf.Variable(tf.zeros(shape=[1]))
Dis_all_weights = [Dis_weight1, Dis_weight2, Dis_bias1, Dis_bias2]

Z = tf.placeholder(tf.float32, shape=[None, 100])
Gen_weight1 = tf.Variable(initialize_weight([100, 128]))
Gen_bias1 = tf.Variable(tf.zeros(shape=[128]))
Gen_weight2 = tf.Variable(initialize_weight([128, 784]))
Gen_bias2 = tf.Variable(tf.zeros(shape=[784]))
Gen_all_weights = [Gen_weight1, Gen_weight2, Gen_bias1, Gen_bias2]


def generator(z):
    Gen_hidden1 = tf.nn.relu(tf.matmul(z, Gen_weight1) + Gen_bias1)
    Gen_a1 = tf.matmul(Gen_hidden1, Gen_weight2) + Gen_bias2
    Gen_output = tf.nn.sigmoid(Gen_a1)
    return Gen_output


def discriminator(x):
    Dis_hidden1 = tf.nn.relu(tf.matmul(x, Dis_weight1) + Dis_bias1)
    Dis_a1 = tf.matmul(Dis_hidden1, Dis_weight2) + Dis_bias2
    Dis_output = tf.nn.sigmoid(Dis_a1)
    return Dis_output, Dis_a1


Gen_sample = generator(Z)

Dis_real, Dis_real_logit = discriminator(X)
Dis_fake, Dis_fake_logit = discriminator(Gen_sample)

Dis_loss = -tf.reduce_mean(tf.log(Dis_real) + tf.log(1. - Dis_fake))
Gen_loss = -tf.reduce_mean(tf.log(Dis_fake))

Dis_optimizer = tf.train.AdamOptimizer().minimize(Dis_loss, var_list=Dis_all_weights)
Gen_optimizer = tf.train.AdamOptimizer().minimize(Gen_loss, var_list=Gen_all_weights)


def sample_noise(row, col):
    return np.random.uniform(-1., 1., size=[row, col])


batch_size = 128
noise_dim = 100
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
if not os.path.exists('out/'):
    os.makedirs('out/')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
for it_counter in range(200000):
    if it_counter % 1000 == 0:
        generator_samples = sess.run(Gen_sample, feed_dict={Z: sample_noise(16, noise_dim)})
        fig = plot(generator_samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_batch, _ = mnist.train.next_batch(batch_size)

    _, discriminator_loss = sess.run([Dis_optimizer, Dis_loss], feed_dict={X: X_batch, Z: sample_noise(batch_size, noise_dim)})
    _, generator_loss = sess.run([Gen_optimizer, Gen_loss], feed_dict={Z: sample_noise(batch_size, noise_dim)})

    if it_counter % 1000 == 0:
        print('Iter: {}'.format(it_counter))
        print('D loss: {:.4}'. format(discriminator_loss))
        print('G_loss: {:.4}'.format(generator_loss))
        print()