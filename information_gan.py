import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
batch_size = 64
noise_dim = 100
X_dimension = mnist.train.images.shape[1]
# latent_dimension = mnist.train.labels.shape[1]
hidden_dimension = 128
latent_dimension = 10

def initialize_weight(dimensions):
    std_dev = 1. / tf.sqrt(dimensions[0] / 2.)
    return tf.random_normal(shape=dimensions, stddev=std_dev)


X = tf.placeholder(tf.float32, shape=[None, X_dimension], name='X')
latent_code = tf.placeholder(tf.float32, shape=[None, latent_dimension])

Dis_weight1 = tf.Variable(initialize_weight([X_dimension , hidden_dimension]))
Dis_bias1 = tf.Variable(tf.zeros(shape=[hidden_dimension]))
Dis_weight2 = tf.Variable(initialize_weight([hidden_dimension, 1]))
Dis_bias2 = tf.Variable(tf.zeros(shape=[1]))
Dis_all_weights = [Dis_weight1, Dis_weight2, Dis_bias1, Dis_bias2]

Z = tf.placeholder(tf.float32, shape=[None, noise_dim])
Gen_weight1 = tf.Variable(initialize_weight([noise_dim + latent_dimension, hidden_dimension]))
Gen_bias1 = tf.Variable(tf.zeros(shape=[hidden_dimension]))
Gen_weight2 = tf.Variable(initialize_weight([hidden_dimension, X_dimension]))
Gen_bias2 = tf.Variable(tf.zeros(shape=[X_dimension]))
Gen_all_weights = [Gen_weight1, Gen_weight2, Gen_bias1, Gen_bias2]

Q_weigh1 = tf.Variable(initialize_weight([X_dimension, hidden_dimension]))
Q_bias1 = tf.Variable(tf.zeros(shape=[hidden_dimension]))
Q_weight2 = tf.Variable(initialize_weight([hidden_dimension, latent_dimension]))
Q_bias2 = tf.Variable(tf.zeros(shape=[latent_dimension]))
Q_all_weights = [Q_weigh1, Q_weight2, Q_bias1, Q_bias2]


def Q_network(x):
    Q_hidden1 = tf.nn.relu(tf.matmul(x, Q_weigh1) + Q_bias1)
    Q_output = tf.nn.softmax(tf.matmul(Q_hidden1, Q_weight2) + Q_bias2)
    return Q_output


def discriminator(x):
    Dis_hidden1 = tf.nn.relu(tf.matmul(x, Dis_weight1) + Dis_bias1)
    Dis_a1 = tf.matmul(Dis_hidden1, Dis_weight2) + Dis_bias2
    Dis_output = tf.nn.sigmoid(Dis_a1)
    return Dis_output, Dis_a1


def generator(z, y):
    Gen_input = tf.concat(axis=1, values=[z, y])
    Gen_hidden1 = tf.nn.relu(tf.matmul(Gen_input, Gen_weight1) + Gen_bias1)
    Gen_a1 = tf.matmul(Gen_hidden1, Gen_weight2) + Gen_bias2
    Gen_output = tf.nn.sigmoid(Gen_a1)
    return Gen_output


def sample_latent_code(m):
    return np.random.multinomial(1, 10*[0.1], size=m)


def sample_noise(row, col):
    return np.random.uniform(-1., 1., size=[row, col])


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


Gen_sample = generator(Z, latent_code)

Q_c_posterior = Q_network(Gen_sample)
conditional_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_c_posterior + 1e-8) * latent_code, 1))
entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(latent_code + 1e-8) * latent_code, 1))
Q_loss = conditional_entropy + entropy

Dis_real, Dis_real_logit = discriminator(X)
Dis_fake, Dis_fake_logit = discriminator(Gen_sample)

Dis_loss = -tf.reduce_mean(tf.log(Dis_real + 1e-8) + tf.log(1 - Dis_fake + 1e-8))

Gen_loss = -tf.reduce_mean(tf.log(Dis_fake + 1e-8))

# Dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_real_logit, labels=tf.ones_like(Dis_real_logit)))
# Dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_fake_logit, labels=tf.zeros_like(Dis_fake_logit)))
# Dis_loss = Dis_loss_real + Dis_loss_fake
#
# Gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_fake_logit, labels=tf.zeros_like(Dis_fake_logit)))

Dis_optimizer = tf.train.AdamOptimizer().minimize(Dis_loss, var_list=Dis_all_weights)
Gen_optimizer = tf.train.AdamOptimizer().minimize(Gen_loss, var_list=Gen_all_weights)
Q_optimizer = tf.train.AdamOptimizer().minimize(Q_loss, var_list=Gen_all_weights + Q_all_weights)

if not os.path.exists('out/final/infogan/'):
    os.makedirs('out/final/infogan/')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
digit_to_generate = 7
num_of_images = 16
for it in range(200000):
    if it % 1000 == 0:
        Z_sample = sample_noise(num_of_images, noise_dim)
        y_sample = np.zeros(shape=[num_of_images, latent_dimension])
        y_sample[:, digit_to_generate] = 1

        samples = sess.run(Gen_sample, feed_dict={Z: Z_sample, latent_code:y_sample})

        fig = plot(samples)
        plt.savefig('out/final/infogan/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    batch_X, _ = mnist.train.next_batch(batch_size)
    c_noise = sample_latent_code(batch_size)
    Z_sample = sample_noise(batch_size, noise_dim)
    _, discriminator_loss = sess.run([Dis_optimizer, Dis_loss], feed_dict={X: batch_X, Z: Z_sample, latent_code: c_noise})
    _, generator_loss = sess.run([Gen_optimizer, Gen_loss], feed_dict={Z: Z_sample, latent_code:c_noise})
    sess.run([Q_optimizer], feed_dict={Z: Z_sample, latent_code: c_noise})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Discriminator loss: {:.4}'. format(discriminator_loss))
        print('Generator loss: {:.4}'.format(generator_loss))
        print()