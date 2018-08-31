import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn.model_selection import train_test_split

def initialize_weight(dimensions):
    std_dev = 1. / tf.sqrt(dimensions[0] / 2.)
    return tf.random_normal(shape=dimensions, stddev=std_dev)


X = tf.placeholder(tf.float32, shape=[None, 784])

Dis_weight1 = tf.Variable(initialize_weight([784, 128]))
Dis_bias1 = tf.Variable(tf.zeros(shape=[128]))

Dis_weight2 = tf.Variable(initialize_weight([128, 1]))
Dis_bias2 = tf.Variable(tf.zeros(shape=[1]))

Dis_all_weights = [Dis_weight1, Dis_weight2, Dis_bias1, Dis_bias2]


Z = tf.placeholder(tf.float32, shape=[None, 16])
latent_code = tf.placeholder(tf.float32, shape=[None, 10])

Gen_weight1 = tf.Variable(initialize_weight([26, 256]))
Gen_bias1 = tf.Variable(tf.zeros(shape=[256]))

Gen_weight2 = tf.Variable(initialize_weight([256, 784]))
Gen_bias2 = tf.Variable(tf.zeros(shape=[784]))

Gen_all_weights = [Gen_weight1, Gen_weight2, Gen_bias1, Gen_bias2]


Q_weight1 = tf.Variable(initialize_weight([784, 128]))
Q_bias1 = tf.Variable(tf.zeros(shape=[128]))

Q_weight2 = tf.Variable(initialize_weight([128, 10]))
Q_bias2 = tf.Variable(tf.zeros(shape=[10]))

Q_all_weights = [Q_weight1, Q_weight2, Q_bias1, Q_bias2]


def sample_noise(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def sample_latent_code(m):
    return np.random.multinomial(1, 10*[0.1], size=m)


def generator(z, c):
    Gen_input = tf.concat(axis=1, values=[z, c])
    Gen_hidden1 = tf.nn.relu(tf.matmul(Gen_input, Gen_weight1) + Gen_bias1)
    Gen_a1 = tf.matmul(Gen_hidden1, Gen_weight2) + Gen_bias2
    Gen_output = tf.nn.sigmoid(Gen_a1)
    return Gen_output


def discriminator(x):
    Dis_hidden1 = tf.nn.relu(tf.matmul(x, Dis_weight1) + Dis_bias1)
    Dis_a1 = tf.matmul(Dis_hidden1, Dis_weight2) + Dis_bias2
    Dis_output = tf.nn.sigmoid(Dis_a1)
    return Dis_output


def Q_network(x):
    Q_hidden1 = tf.nn.relu(tf.matmul(x, Q_weight1) + Q_bias1)
    Q_output = tf.nn.softmax(tf.matmul(Q_hidden1, Q_weight2) + Q_bias2)
    return Q_output


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


Gen_sample = generator(Z, latent_code)
Dis_real = discriminator(X)
Dis_fake = discriminator(Gen_sample)
Q_c_posterior = Q_network(Gen_sample)
Q_semi_c_posterior = Q_network(X)

Dis_loss = -tf.reduce_mean(tf.log(Dis_real + 1e-8) + tf.log(1 - Dis_fake + 1e-8))
Gen_loss = -tf.reduce_mean(tf.log(Dis_fake + 1e-8))

conditional_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_c_posterior + 1e-8) * latent_code, 1))
entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(latent_code + 1e-8) * latent_code, 1))
Q_loss = conditional_entropy + entropy

conditional_semi_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_semi_c_posterior + 1e-8) * latent_code, 1))
entropy_semi = tf.reduce_mean(-tf.reduce_sum(tf.log(latent_code + 1e-8) * latent_code, 1))
Q_semi_loss = conditional_semi_entropy + entropy_semi

gen_semi_conditional_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_c_posterior + 1e-8) * latent_code, 1))
gen_semi_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(latent_code + 1e-8) * latent_code, 1))
Q_gen_semi_loss = gen_semi_conditional_entropy + gen_semi_entropy

Dis_optimizer = tf.train.AdamOptimizer().minimize(Dis_loss, var_list=Dis_all_weights)
Gen_optimizer = tf.train.AdamOptimizer().minimize(Gen_loss, var_list=Gen_all_weights)
Q_optimizer = tf.train.AdamOptimizer().minimize(Q_loss, var_list=Gen_all_weights + Q_all_weights)
Q_semi_optimizer = tf.train.AdamOptimizer().minimize(Q_semi_loss, var_list=Q_all_weights)
Q_gen_semi_optimizer = tf.train.AdamOptimizer().minimize(Q_loss, var_list=Gen_all_weights + Q_all_weights)

batch_size = 32
noise_dimension = 16

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
if not os.path.exists('out/infogan/'):
    os.makedirs('out/infogan/')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_num = 0
i = 0
count = 0
print('Loading Data')
Xunsup, Xsup, Yunsup, Ysup = train_test_split(mnist.train.images, mnist.train.labels, test_size=0.1, stratify=mnist.train.labels)
print('Data Loaded')
num_batch = len(Xsup) / batch_size
sup_on = True
digit_to_generate = 7

print('Supervised Data on Run')
for it in range(1000000):
    if it % 1000 == 0:
        Z_sample = sample_noise(16, noise_dimension)
        c_noise = np.zeros([16, 10])
        c_noise[range(16), digit_to_generate] = 1

        samples = sess.run(Gen_sample,
                           feed_dict={Z: Z_sample, latent_code: c_noise})

        fig = plot(samples)
        plt.savefig('out/infogan/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)


    Z_sample = sample_noise(batch_size, noise_dimension)
    c_noise = sample_latent_code(batch_size)

    if(sup_on==True and (batch_num*batch_size+batch_size) > Xsup.shape[0]):
        batch_num=0
        sup_on = False
        print('Unsupervised Data on Run')

    if(sup_on==False and (batch_num*batch_size+batch_size) > Xunsup.shape[0]):
        batch_num=0
        sup_on = True
        print('Supervised Data on Run')

    if(sup_on == True):
        _, D_loss_curr = sess.run([Dis_optimizer, Dis_loss],
                                  feed_dict={X: Xsup[batch_num * batch_size: batch_num * batch_size + batch_size, :], Z: Z_sample,
                                             latent_code: Ysup[batch_num * batch_size: batch_num * batch_size + batch_size, :]})

        _, G_loss_curr = sess.run([Gen_optimizer, Gen_loss],
                                  feed_dict={Z: Z_sample,
                                             latent_code: Ysup[batch_num * batch_size: batch_num * batch_size + batch_size, :]})

        sess.run([Q_semi_optimizer], feed_dict={X: Xsup[batch_num * batch_size: batch_num * batch_size + batch_size, :],
                                                latent_code: Ysup[batch_num * batch_size: batch_num * batch_size + batch_size, :]})
        sess.run([Q_gen_semi_optimizer], feed_dict={Z: Z_sample,
                                                    latent_code: Ysup[batch_num * batch_size: batch_num * batch_size + batch_size, :]})

    else:
        _, D_loss_curr = sess.run([Dis_optimizer, Dis_loss],
                                  feed_dict={X: Xunsup[batch_num * batch_size: batch_num * batch_size + batch_size, :], Z: Z_sample, latent_code: c_noise})

        _, G_loss_curr = sess.run([Gen_optimizer, Gen_loss],
                                  feed_dict={Z: Z_sample, latent_code: c_noise})

        sess.run([Q_optimizer], feed_dict={Z: Z_sample, latent_code: c_noise})

    batch_num = batch_num + 1

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()