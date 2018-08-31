import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
batch_size = 64
noise_dimension = 100
X_dimension = mnist.train.images.shape[1]
y_dimension = mnist.train.labels.shape[1]
hidden_dimension = 128


def initialize_weight(dimensions):
    std_dev = 1. / tf.sqrt(dimensions[0] / 2.)
    return tf.random_normal(shape=dimensions, stddev=std_dev)


X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, y_dimension])

Dis_weight1 = tf.Variable(initialize_weight([X_dimension + y_dimension, hidden_dimension]))
Dis_bias1 = tf.Variable(tf.zeros(shape=[hidden_dimension]))

Dis_weight2 = tf.Variable(initialize_weight([hidden_dimension, 1]))
Dis_bias2 = tf.Variable(tf.zeros(shape=[1]))

Dis_all_weights = [Dis_weight1, Dis_weight2, Dis_bias1, Dis_bias2]


def discriminator(x, y):
    Dis_input = tf.concat(axis=1, values=[x, y])
    Dis_hidden1 = tf.nn.relu(tf.matmul(Dis_input, Dis_weight1) + Dis_bias1)
    Dis_a1 = tf.matmul(Dis_hidden1, Dis_weight2) + Dis_bias2
    Dis_output = tf.nn.sigmoid(Dis_a1)
    return Dis_output, Dis_a1


Z = tf.placeholder(tf.float32, shape=[None, noise_dimension])

Gen_weight1 = tf.Variable(initialize_weight([noise_dimension + y_dimension, hidden_dimension]))
Gen_bias1 = tf.Variable(tf.zeros(shape=[hidden_dimension]))

Gen_weight2 = tf.Variable(initialize_weight([hidden_dimension, X_dimension]))
Gen_bias2 = tf.Variable(tf.zeros(shape=[X_dimension]))

Gen_all_weights = [Gen_weight1, Gen_weight2, Gen_bias1, Gen_bias2]


def generator(z, y):
    Gen_input = tf.concat(axis=1, values=[z, y])
    Gen_hidden1 = tf.nn.relu(tf.matmul(Gen_input, Gen_weight1) + Gen_bias1)
    Gen_a1 = tf.matmul(Gen_hidden1, Gen_weight2) + Gen_bias2
    Gen_output = tf.nn.sigmoid(Gen_a1)
    return Gen_output


def sample_noise(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


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


Gen_sample = generator(Z, y)
Dis_real, Dis_real_logitl = discriminator(X, y)
Dis_fake, Dis_fake_logit = discriminator(Gen_sample, y)

Dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_real_logitl, labels=tf.ones_like(Dis_real_logitl)))
Dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_fake_logit, labels=tf.zeros_like(Dis_fake_logit)))
Dis_loss = Dis_loss_real + Dis_loss_fake
Gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_fake_logit, labels=tf.ones_like(Dis_fake_logit)))

Dis_optimizer = tf.train.AdamOptimizer().minimize(Dis_loss, var_list=Dis_all_weights)
Gen_optimizer = tf.train.AdamOptimizer().minimize(Gen_loss, var_list=Gen_all_weights)

if not os.path.exists('out/final/cgan/'):
    os.makedirs('out/final/cgan/')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
digit_to_generate = 7
for it in range(200000):
    if it % 1000 == 0:
        n_sample = 16

        Z_sample = sample_noise(n_sample, noise_dimension)
        y_sample = np.zeros(shape=[n_sample, y_dimension])
        y_sample[:, digit_to_generate] = 1

        samples = sess.run(Gen_sample, feed_dict={Z: Z_sample, y:y_sample})

        fig = plot(samples)
        plt.savefig('out/final/cgan/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, y_mb = mnist.train.next_batch(batch_size)

    Z_sample = sample_noise(batch_size, noise_dimension)
    _, D_loss_curr = sess.run([Dis_optimizer, Dis_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([Gen_optimizer, Gen_loss], feed_dict={Z: Z_sample, y:y_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Discriminator loss: {:.4}'. format(D_loss_curr))
        print('Generator loss: {:.4}'.format(G_loss_curr))
        print()