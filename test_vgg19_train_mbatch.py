import time
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

import utils
from vgg import vgg19_trainable_skin as vgg19


# LECTURA DE DATOS
path = '../data/ISB2016/'
data = pd.read_csv(path + 'ISB_Train_short.csv', header=None)
train_images = data[0]
train_labels = data[2]
epoch = 4
num_class = 2
learning_rate = 0.01
mini_batch = 20
total_images = len(train_images)

assert (total_images/mini_batch).is_integer(), 'El minibatch debe ser multiplo del total de datos de entrada'


def get_images_batch(index=0, indexless=0):

    start = index * mini_batch
    end = mini_batch * (index + 1)
    batch_list = []
    label_list = []

    # print(start, end)
    for i in range(start, end):
        # print(train_images[i])
        img = utils.load_image(path + "images/" + train_images[i] + '.jpg')[:, :, :3]
        batch_list.append(img.reshape((1, 224, 224, 3)))
        label_list.append(train_labels[i])

    if (end/total_images) == 1:
        indexless = indexless + index + 1

    return np.concatenate([block for block in batch_list], 0), label_list, indexless


# ESQUEMA GENERAL
# ---------------

with tf.Session() as sess:
    # Container of the input data for the model
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, num_class])
    train_mode = tf.placeholder(tf.bool)

    # Initialize of the model VGG19
    vgg = vgg19.Vgg19('./weight/vgg19.npy', load_weight_fc=False)
    vgg.build(images, train_mode)
    sess.run(tf.global_variables_initializer())

    #
    # PHASE #1 Test classification
    # ----------------------------
    print('\n# PHASE #1 Test classification')
    batch, label, _ = get_images_batch()
    prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
    utils.print_prob_all(prob, path + 'synset_skin.txt', top=0)
    utils.print_accuracy(label, prob)

    #
    # PHASE #2 Training model
    # -----------------------
    print('\n# PHASE #2 Training model')
    cost = tf.reduce_mean((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    t0 = time.time()
    idx_aux = 0
    for idx in range(epoch):
        i = idx - idx_aux
        batch, label, idx_aux = get_images_batch(i, idx_aux)

        # Generate the 'one hot' or labels
        label = tf.one_hot([li for li in label], on_value=1, off_value=0, depth=num_class)
        label = list(sess.run(label))

        t_start = time.time()
        sess.run(train, feed_dict={images: batch, true_out: label, train_mode: True})
        t_end = time.time()
        print("    Iteration: %d train on batch time: %7.3f ms." % (idx, (t_end - t_start) * 1000))

    t1 = time.time()
    print("    Batch size: %d" % len(batch))
    print("    Iterations: %d" % epoch)
    print("    Time per iteration: %7.3f ms" % ((t1 - t0) * 1000 / epoch))

    #
    # PHASE #3 Post-Test classification
    # ---------------------------------
    print('\n# PHASE #1 Test classification')
    batch, label, _ = get_images_batch()
    prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
    utils.print_prob_all(prob, path + 'synset_skin.txt', top=0)
    utils.print_accuracy(label, prob)

    # SAVE WEIGHT
    # vgg.save_npy(sess, './weight/save-skin-vgg19-8.npy')